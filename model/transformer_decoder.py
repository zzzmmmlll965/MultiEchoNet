import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange as o_rearrange
from timm.models.layers import DropPath, trunc_normal_
import math

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()



class ConvBlock(nn.Module):
    """conv3 -- bn -- relu"""
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STP(nn.Module):
    """
    encoder feature aggregation
    """
    def __init__(self, stage_idx, task_chan, fea_chan, norm_layer=nn.BatchNorm2d):
        super(STP, self).__init__()

        self.stage_idx = stage_idx

        self.embed_chan = task_chan if stage_idx == 0 else int(task_chan // 2)

        self.task_embed = nn.Sequential(
            nn.Conv2d(in_channels=task_chan, out_channels=self.embed_chan, kernel_size=3, padding=1),
            norm_layer(self.embed_chan),
            nn.ReLU(inplace=True)
        )

        self.fea_embed = nn.Sequential(
            nn.Conv2d(in_channels=fea_chan, out_channels=self.embed_chan, kernel_size=3, padding=1),
            norm_layer(self.embed_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, task_x, fea_x):
        """
        task_x: [B, C, H, W]
        fea_x: [B, C, H, W]
        """
        # B, N, _C = task_x.shape
        # task_x = task_x.permute(0, 2, 1).reshape(B, _C, h, w).contiguous()
        # backbone feature conv-bn-relu
        fea_x = self.fea_embed(fea_x)       # 为了调整通道数
        # task feature up
        if self.stage_idx != 0:
            task_x = F.interpolate(task_x, fea_x.shape[-2:], mode='bilinear', align_corners=False)
        # task feature conv-bn-relu
        task_x = self.task_embed(task_x)    # 调整通道数
        # add_out: [B, C, H, W]
        add_out = fea_x + task_x
        # add_out = add_out.flatten(2).transpose(1, 2)

        return add_out


class task_pattern_propagation(nn.Module):
    """
    构建各自任务注意力模式，形成共享任务模式，然后再进行任务模式分配
    """
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super(task_pattern_propagation, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_qkv1 = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj_qkv2 = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        # channel reduction
        self.attn_conv = nn.Conv2d(in_channels=2*num_heads, out_channels=num_heads, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x1.shape
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv1 = self.proj_qkv1(x1).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                 4).contiguous()
        qkv2 = self.proj_qkv2(x2).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                 4).contiguous()
        q1, k1, v1 = qkv1.unbind(0)
        q2, k2, v2 = qkv2.unbind(0)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # x1的任务模式注意力
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale  # x2的任务模式注意力
        # concat: -> [batch, 2*num_heads, num_patches, num_patches]
        cat_attn = torch.cat((attn1, attn2), dim=1)
        # channel reduction: -> [batch, num_heads, num_patches, num_patches]
        cat_attn = self.attn_conv(cat_attn)                 # 共享任务模式

        cat_attn = cat_attn.softmax(dim=-1)
        cat_attn = self.attn_drop(cat_attn)
        # share task pattern
        x1 = (cat_attn @ v1).transpose(1, 2).reshape(B, N, C).contiguous()
        x1 = self.proj1(x1)
        x1 = self.proj_drop(x1)

        x2 = (cat_attn @ v2).transpose(1, 2).reshape(B, N, C).contiguous()
        x2 = self.proj2(x2)
        x2 = self.proj_drop(x2)

        return x1, x2


class TPP(nn.Module):
    """
    任务模式传播模块，解决任务模式纠缠问题
    """
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TPP, self).__init__()
        self.norm11 = norm_layer(dim)
        self.norm12 = norm_layer(dim)

        self.tpp_attn = task_pattern_propagation(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 attn_drop=attn_drop, proj_drop=drop)
        self.norm21 = norm_layer(dim)
        self.norm22 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # 暂且先设置为两个分支独立于norm和mlp
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x1, x2):
        offset_x1, offset_x2 = self.tpp_attn(self.norm11(x1), self.norm12(x2))

        x1 = x1 + offset_x1
        x2 = x2 + offset_x2

        x1 = x1 + self.mlp1(self.norm21(x1))
        x2 = x2 + self.mlp2(self.norm22(x2))

        return x1, x2


class PreliminaryDecoder(nn.Module):
    """
    to generate init-task feature
    """
    def __init__(self, p):
        super(PreliminaryDecoder, self).__init__()

        self.all_task = p.TASKS.NAMES
        self.embed_dim = p.decoder_embed_dim

        # intermediate supervision
        input_channels = p.backbone_channels[-1]
        task_channels = self.embed_dim

        self.intermediate_head = nn.ModuleDict()
        self.preliminary_decoder = nn.ModuleDict()

        for t in p.TASKS.NAMES:
            self.intermediate_head[t] = nn.Conv2d(task_channels, p.TASKS.NUM_OUTPUT[t], 1)
            self.preliminary_decoder[t] = nn.Sequential(
                ConvBlock(input_channels, input_channels),
                ConvBlock(input_channels, task_channels),
            )
        # TPP没有设置为通用多任务类型，需要手动输入两个任务特征
        self.task_pattern_pro = TPP(task_channels, num_heads=4, qkv_bias=True, qk_scale=None)

    def forward(self, x):
        """ 传入编码器多尺度特征中最小尺度特征, 当前仅支持双任务"""
        h, w = x.shape[-2:]
        pre_x_list = []     # 任务特定特征list
        inter_pred = {}     # 中间初始预测 此处没有与task-feature做融合

        for task in self.all_task:
            _x = self.preliminary_decoder[task](x)
            pre_x_list.append(_x)
        # 任务模式传播 传入Tensor格式: [B, N, C]
        pre_x_list = [rearrange(_x, 'b c h w -> b (h w) c') for _x in pre_x_list]
        pre_x_list[0], pre_x_list[1] = self.task_pattern_pro(pre_x_list[0], pre_x_list[1])
        pre_x_list = [rearrange(_x, 'b (h w) c -> b c h w', h=h, w=w) for _x in pre_x_list]
        # 获取初始预测，但是并没有利用初始预测
        for idx, task in enumerate(self.all_task):
            _inter_p = self.intermediate_head[task](pre_x_list[idx])
            inter_pred[task] = _inter_p

        return pre_x_list, inter_pred


class SelfAttention(nn.Module):
    def __init__(self,
                 fea_no,
                 dim_in,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_method='dw_bn',
                 kv_method='dw_bn',
                 kernel_size_q=3,
                 kernel_size_kv=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_in
        self.num_heads = num_heads
        self.scale = dim_in ** -0.5
        self.fea_no = fea_no

        self.conv_proj_q = self._build_projection(
            dim_in, kernel_size_q, padding_q,
            stride_q, q_method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size_kv, padding_kv,
            stride_kv, kv_method
        )

        self.proj_q = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_in, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_in, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_in)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, kernel_size, padding, stride, method):
        """for decrease the self-attention computation to downsample the qkv"""
        if method == 'dw_bn':
            proj = [nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                                            stride=stride, bias=False, groups=dim_in),
                                  nn.BatchNorm2d(dim_in),
                                  Rearrange('b c h w -> b (h w) c')) for _ in range(self.fea_no)]
            proj = nn.ModuleList(proj)
        elif method == 'avg':
            proj = [nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride, ceil_mode=True),
                                  Rearrange('b c h w -> b (h w) c')) for _ in range(self.fea_no)]
            proj = nn.ModuleList(proj)

        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def split_x(self, x, h, w):
        res = h*w
        x_list = []
        for i in range(self.fea_no):
            _x = rearrange(x[:, res*i:res*(i+1), :], 'b (h w) c -> b c h w', h=h, w=w)
            x_list.append(_x)
        return x_list

    def forward_conv(self, x, h, w):

        x_list = self.split_x(x, h, w)

        if self.conv_proj_q is not None:
            q_list = [self.conv_proj_q[i](x_list[i]) for i in range(self.fea_no)]
            q = torch.cat(q_list, dim=1)
        else:
            q_list = [rearrange(x, 'b c h w -> b (h w) c') for x in x_list]
            q = torch.cat(q_list, dim=1)

        if self.conv_proj_k is not None:
            k_list = [self.conv_proj_k[i](x_list[i]) for i in range(self.fea_no)]
            k = torch.cat(k_list, dim=1)
        else:
            k_list = [rearrange(x, 'b c h w -> b (h w) c') for x in x_list]
            k = torch.cat(k_list, dim=1)

        if self.conv_proj_v is not None:
            v_list = [self.conv_proj_v[i](x_list[i]) for i in range(self.fea_no)]
            v = torch.cat(v_list, dim=1)
        else:
            v_list = [rearrange(x, 'b c h w -> b (h w) c') for x in x_list]
            v = torch.cat(v_list, dim=1)

        return q, k, v

    def forward(self, x, h, w):
        """
        Args:
        x:multi-task tenser --> [B, task_num*N, C]
        h:task feature height
        w:task feature width
        """
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)       # 同样是多任务特征，但是有被进行下采样了

        return x


class MT_Block(nn.Module):

    def __init__(self,
                 task_no,
                 dim_in,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.stride_q = kwargs['stride_q']
        self.embed_dim = dim_in
        self.task_no = task_no

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(self.embed_dim)
        self.norm2 = norm_layer(self.embed_dim)

        dim_mlp_hidden = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.embed_dim, hidden_features=dim_mlp_hidden, act_layer=act_layer,drop=drop)

        self.attn = SelfAttention(task_no,self.embed_dim, num_heads, qkv_bias, attn_drop, drop, **kwargs)

    def split_x(self, x, h, w):
        res = h*w
        x_list = []
        for i in range(self.task_no):
            _x = x[:, res*i:res*(i+1), :]
            x_list.append(_x)
        return x_list

    def forward(self, x_list):
        """
        x_list: task_feature-->[B, C, H, W]
        """
        h, w = x_list[0].shape[2:]
        x_list = [rearrange(_x, 'b c h w -> b (h w) c') for _x in x_list]
        x = torch.cat(x_list, dim=1) # cat on space dim

        res = x
        attn = self.attn(self.norm1(x), h, w)

        # interpolate output of attention to previous resolution
        sh, sw = h//self.stride_q, w//self.stride_q
        attn_list = self.split_x(attn, sh, sw)
        attn_list = [rearrange(_it, 'b (h w) c -> b c h w', h=sh, w=sw) for _it in attn_list]
        attn_list = [F.interpolate(_it, size=(h, w), mode='bilinear', align_corners=False) for _it in attn_list]
        attn_list = [rearrange(_it, 'b c h w -> b (h w) c') for _it in attn_list]
        attn = torch.cat(attn_list, dim=1)

        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_list = self.split_x(x, h, w)
        x_list = [rearrange(_it, 'b (h w) c -> b c h w', h=h, w=w) for _it in x_list]

        return x_list


class MT_Stage(nn.Module):
    def __init__(self,
                 p,
                 stage_idx,
                 embed_dim=768,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()

        self.stage_idx = stage_idx
        self.fea_chan_idx = 3 - stage_idx  # 编码器特征索引
        self.embed_dim = embed_dim
        self.task_no = len(p.TASKS.NAMES)

        self.block = MT_Block(
                    dim_in=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )

        # 多尺度特征聚合
        task_chan = embed_dim if stage_idx == 0 else (embed_dim*2)        # 第一个阶段不需要上采样也不需要改变通道数

        self.en_decoder_fuse = nn.ModuleList()
        for _ in range(self.task_no):
            self.en_decoder_fuse.append(STP(stage_idx, task_chan, fea_chan=p.backbone_channels[self.fea_chan_idx]))

        # 初始化
        self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_list, back_fea):
        """
        x_list: [b, c, h, w]
        back_fea: [b, c, h, w]
        """
        for i in range(self.task_no):
            x_list[i] = self.en_decoder_fuse[i](x_list[i], back_fea[self.fea_chan_idx]) # STP的输入还需要x的h和w

        x_list = self.block(x_list)

        return x_list


class MTT(nn.Module):
    def __init__(self,
                 p,
                 embed_dim=3,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()

        self.p = p

        self.all_tasks = p.TASKS.NAMES
        task_no = len(self.all_tasks)
        self.task_no = task_no

        self.num_stages = spec['NUM_STAGES']
        self.embed_dim = embed_dim

        mt_in_chans = embed_dim

        self.mt_embed_dims = []
        target_channel = embed_dim
        self.redu_chan = nn.ModuleList()
        self.MT_stages = nn.ModuleList()


        for i in range(self.num_stages):
            cur_mt_embed_dim = spec['DIM_EMBED'][i]
            kwargs = {
                'task_no': task_no,
                'embed_dim': cur_mt_embed_dim,
                'depth': 1,
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': 0,
                'attn_drop_rate': 0,
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'q_method': spec['Q_PROJ_METHOD'][i],
                'kv_method': spec['KV_PROJ_METHOD'][i],
                'kernel_size_q': spec['KERNEL_Q'][i],
                'kernel_size_kv': spec['KERNEL_KV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }
            stage = MT_Stage(
                p=p,
                stage_idx=i,
                in_chans=mt_in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            self.MT_stages.append(stage)

            mt_in_chans = cur_mt_embed_dim
            self.mt_embed_dims.append(mt_in_chans)

            _redu_chan = nn.ModuleList([nn.Conv2d(mt_in_chans, target_channel, 1) for _ in range(task_no)])
            self.redu_chan.append(_redu_chan)

        # Final convs
        self.mt_embed_dim = target_channel      # 就是第一个阶段的embed_dim
        self.mt_proj = nn.ModuleDict()
        for task in self.all_tasks:
            self.mt_proj[task] = nn.Sequential(nn.Conv2d(self.mt_embed_dim, self.mt_embed_dim, 3, padding=1),
                                               nn.BatchNorm2d(self.mt_embed_dim), nn.ReLU(True))
            trunc_normal_(self.mt_proj[task][0].weight, std=0.02)

    def forward(self, x_list, back_fea):
        '''
        Input:
        x_dict: dict of feature map lists {task: [torch.tensor([B, H*W, embed_dim]), xxx]}
        '''
        h, w = self.p.mtt_resolution
        th = h * 2**(self.num_stages-1) * 2
        tw = w * 2**(self.num_stages-1) * 2
        multi_scale_task_feature = {_t: 0 for _t in self.all_tasks}

        for i in range(self.num_stages):
            x_list = self.MT_stages[i](x_list, back_fea)        # [b, c, h, w]
            # _x_list = [rearrange(_x, 'b c h w -> b (h w) c') for _x in x_list]

            for ii, task in enumerate(self.all_tasks):
                task_x = x_list[ii]
                if i > 0:
                    task_x = self.redu_chan[i][ii](task_x)
                task_x = F.interpolate(task_x, size=(th, tw), mode='bilinear', align_corners=False)  # 上采样到1/4
                # add feature from all the scales
                multi_scale_task_feature[task] += task_x

        x_dict = {}
        for i, task in enumerate(self.all_tasks):
            x_dict[task] = self.mt_proj[task](multi_scale_task_feature[task])

        return x_dict


class TransformerDecoder(nn.Module):
    """
    MTT decoder
    """
    def __init__(self, p):
        super().__init__()

        self.embed_dim = p.decoder_embed_dim
        p.mtt_resolution = [int(_ // 32) for _ in p.TRAIN.SCALE]    # resolution for input feature
        self.p = p

        spec = {
            'ori_embed_dim': self.embed_dim,
            'NUM_STAGES': 3,
            'DIM_EMBED': [self.embed_dim, self.embed_dim//2, self.embed_dim//4],
            'NUM_HEADS': [2, 2, 2],
            'MLP_RATIO': [4., 4., 4.],
            'DROP_PATH_RATE': [0.15, 0.15, 0.15],
            'QKV_BIAS': [True, True, True],
            'KV_PROJ_METHOD': ['avg', 'avg', 'avg'],
            'KERNEL_KV': [2, 4, 8],
            'PADDING_KV': [0, 0, 0],
            'STRIDE_KV': [2, 4, 8],
            'Q_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
            'KERNEL_Q': [3, 3, 3],
            'PADDING_Q': [1, 1, 1],
            'STRIDE_Q': [2, 2, 2],
        }

        self.pre_decoder = PreliminaryDecoder(p)

        self.SDFormer_pp = MTT(p, embed_dim=self.embed_dim, spec=spec)

    def forward(self, fea_x_list):
        '''
        Input:
        Backbone multi-scale feature list: 4 * x: tensor [B, embed_dim, h, w]
        '''

        task_x_list, inter_pred = self.pre_decoder(fea_x_list[-1])
        # 传入Tensor均为[b, c, h, w]
        x_dict = self.SDFormer_pp(task_x_list, fea_x_list) # multi-scale input
        return x_dict, inter_pred