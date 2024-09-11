import torch
import torchvision
import numpy as np
import tqdm
import math
from echonet.EF import cal_ef
from medpy.metric.binary import hd95
def run_epoch(model, dataloader, train, optim, device, block_size=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    total1 = 0.

    n = 0

    pos = 0   #统计正样本的数量
    neg = 0   #统计负样本的数量
    pos_pix = 0   #统计正样本像素的数量
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []
    hd95_large_list=[]
    hd95_small_list = []
    oks_l_list = []
    oks_s_list = []


    with torch.set_grad_enabled(train):       #是否开启梯度计算
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace,large_point, small_point)) in dataloader: #,large_point,small_point
                # 样本级别的统计：统计large_trace和small_trace中值为1的元素数量来累加。
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                # 像素级别的统计
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # 对大图像帧(large_frame)和小图像帧(small_frame)进行预测，并计算损失(loss_large和loss_small)。
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                large_point = large_point.float().to(device)

                #预测结果(y_large和y_small)和标签(large_trace和small_trace)之间的逻辑操作，
                # 计算了像素级别的交集(logical_and)和并集(logical_or)，并将其累加到large_inter、large_union、small_inter和small_union中。
                y_large = model(large_frame)["out"]
           ##########################################################point_loss###########################################################################
                large_point =large_point.permute(0, 3, 1, 2)
                y_large_p = model(large_frame)["out1"]
                loss_large_p = torch.nn.functional.mse_loss(y_large_p[:, 0, :, :], large_point[:, 0, :, :], reduction='sum')
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_large[:,0,:,:].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:,0,:,:].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:,0,:,:].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:,0,:,:].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))


                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                small_point = small_point.float().to(device)
                y_small = model(small_frame)["out"]
                y_small_p= model(small_frame)["out1"]

                small_point = small_point.permute(0, 3, 1, 2)
                loss_small_p = torch.nn.functional.mse_loss(y_small_p[:, 0, :, :], small_point[:, 0, :, :], reduction='sum')

                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                ###########################################################keypoints loss######################################################################

                # Take gradient step if training
                loss1 = (loss_large + loss_small) / 2      #seg_loss
                loss2 = (loss_large_p+loss_small_p) / 2    #point_loss

                if train:
                    optim.zero_grad()
                    loss1.backward()
                    loss2.backward()
                    optim.step()

                # 累积的损失
                total += loss1.item()
                total1 += loss2.item()
                n += large_trace.size(0)  #n是样本数量的累加器
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)
                hd95_large = hd95(y_large[:, 0, :, :], large_trace)
                hd95_small = hd95(y_small[:, 0, :, :], small_trace)
                hd95_large_list.extend([hd95_large])
                hd95_small_list.extend([hd95_small])

                thr = 0.2 *11.2
                pck_l= PCK_metric(y_large_p[:, 0, :, :], large_point[:, 0, :, :],thr)
                pck_s= PCK_metric(y_small_p[:, 0, :, :], small_point[:, 0, :, :],thr)


                # Show info on process bar
                # pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss1.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                # pbar.update()
                pbar.set_postfix_str("{:.4f} {:.4f}({:.4f}) / {:.4f} {:.4f}, {:.4f},{:.4f}, {:.4f}".format(total / n / 112 / 112, total1 / (n * 3) / 112 / 112, loss1.item() / large_trace.size(0) / 112 / 112, loss2.item() / large_trace.size(0) / 3 / 2, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()


    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)
    hd95_large_list = np.array(hd95_large_list)
    hd95_small_list = np.array(hd95_small_list)


    if n == 0:
        return 0, large_inter_list, large_union_list, small_inter_list, small_union_list
    else:
        return (total / n / 112 / 112,
                total1 / (n * 3) / 112 / 112,
                hd95_large,
                hd95_small,
                hd95_large_list,
                hd95_small_list,
                pck_l,
                pck_s,
                large_inter_list,
                large_union_list,
                small_inter_list,
                small_union_list,
                y_large_p,
                y_small_p,
                y_large,
                y_small,
                )



def _video_collate_fn(x):
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i


# def PCK_metric(pred, gt, thr):
#     num_points = pred.shape[1]
#     num_samples = pred.shape[0]
#     results = np.full((num_samples, num_points), 0, dtype=np.float32)
#
#     for i in range(num_samples):
#         for j in range(num_points):
#             distance = torch.norm(pred[i, j, :] - gt[i, j, :])
#             if distance <= thr:
#                 results[i, j] = 1
#
#     mean_points = np.mean(results, axis=0)
#     mean_all = np.mean(mean_points)
#     return mean_all
# def PCK_metric(pred, gt, thr):
#     # 计算预测坐标与真实坐标之间的欧式距离
#     distances = np.linalg.norm(pred - gt, axis=2)
#
#     # 计算每个点是否在阈值范围内
#     results = (distances <= thr).astype(np.float32)
#
#     # 计算每个样本中符合要求的点的平均值
#     mean_points = np.mean(results, axis=1)
#
#     # 计算所有样本的平均值
#     mean_all = np.mean(mean_points)
#
#     return mean_all
def PCK_metric(pred, gt, thr):
    # 计算预测坐标与真实坐标之间的欧式距离
    distances = np.linalg.norm(pred.astype(np.float32) - gt.astype(np.float32), axis=2)
    # 计算每个点是否在阈值范围内
    results = (distances <= thr)
    # 计算每个样本中符合要求的点的平均值
    mean_points = np.mean(results, axis=1)
    # 计算所有样本的平均值
    mean_all = np.mean(mean_points)

    return mean_all
if __name__ == "__main__":
    a=torch.randn(8,3,2)
    b=torch.randn(8,3,2)
    thr=0.5*6.12
    pck=PCK_metric(a.detach().cpu().numpy() , b.detach().cpu().numpy() , thr)
    print(pck)
