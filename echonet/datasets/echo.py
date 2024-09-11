"""EchoNet-Dynamic Dataset."""

import os
import collections
import pandas

import numpy as np
import skimage.draw
import torchvision
import echonet


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

  参数:
        root (字符串): 数据集的根目录（默认为`echonet.config.DATA_DIR`）
        split (字符串): 选择之一 {``train'', ``val'', ``test'', ``all'', 或 ``external_test''}
        target_type (字符串或列表, 可选): 要使用的目标类型，
            ``Filename''、``EF''、``EDV''、``ESV''、``LargeIndex'',
            ``SmallIndex''、``LargeFrame''、``SmallFrame''、``LargeTrace'',
            或 ``SmallTrace''
            也可以是列表，以输出包含所有指定目标类型的元组。
            目标表示：
                ``Filename'' (字符串): 视频的文件名
                ``EF'' (浮点数): 射血分数
                ``EDV'' (浮点数): 舒张末期容积
                ``ESV'' (浮点数): 收缩末期容积
                ``LargeIndex'' (整数): 视频中大（舒张期）帧的索引
                ``SmallIndex'' (整数): 视频中小（收缩期）帧的索引
                ``LargeFrame'' (np.array 形状=(3, 高度, 宽度)): 归一化的大（舒张期）帧
                ``SmallFrame'' (np.array 形状=(3, 高度, 宽度)): 归一化的小（收缩期）帧
                ``LargeTrace'' (np.array 形状=(高度, 宽度)): 左心室大（舒张期）分割
                    像素值为0表示像素在左心室外部
                             1表示像素在左心室内部
                ``SmallTrace'' (np.array 形状=(高度, 宽度)): 左心室小（收缩期）分割
                    像素值为0表示像素在左心室外部
                             1表示像素在左心室内部
            默认为 ``EF''。
        mean (整数、浮点数或 np.array 形状=(3,), 可选): 所有通道的均值（如果是标量）或每个通道的均值（如果是 np.array）。
            用于对视频进行归一化。默认为 0（不进行平移）。
        std (整数、浮点数或 np.array 形状=(3,), 可选): 所有通道的标准差（如果是标量）或每个通道的标准差（如果是 np.array）。
            用于对视频进行归一化。默认为 0（不进行缩放）。
        length (整数或 None, 可选): 从视频中裁剪的帧数。如果为 ``None''，则返回尽可能长的剪辑。
            默认为 16。
        period (整数, 可选): 从视频中采样的间隔（即每隔 ``period'' 帧取一帧）。
            默认为 2。
        max_length (整数或 None, 可选): 裁剪视频的最大帧数（主要用于缩短过长的视频，当 ``length'' 设置为 None 时）。
            如果为 ``None''，则不对任何视频进行缩短。
            默认为 250。
        clips (整数, 可选): 要采样的剪辑数量。主要用于测试时利用随机剪辑进行数据增强。
            默认为 1。
        pad (整数或 None, 可选): 在每个帧的每一侧填充像素数（用于数据增强）。
            并且取原始尺寸的窗口。如果为 ``None''，则不进行填充。
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)    #使用pandas库的read_csv函数读取CSV文件，并将其保存到data变量中
            data["Split"].map(lambda x: x.upper())  #对data中"Split"列的每个元素应用lambda函数，将元素的值转换为大写字母

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()   #将data中"FileName"列的值作为列表赋值给对象的fnames属性
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()   ## 将data中的数据部分转换为二维列表，并将结果赋值给对象的outcome属性

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            ############################################keypoints##############################################################
            self.pframes = collections.defaultdict(list)
            self.ptrace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "keypoints.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "X3", "Y3", "Frame"]
                for line in f:
                    filename, x1, y1, x2, y2, x3, y3, frame = line.strip().split(',')
                    px1 = float(x1)
                    py1 = float(y1)
                    px2 = float(x2)
                    py2 = float(y2)
                    px3 = float(x3)
                    py3 = float(y3)
                    frame = int(frame)
                    if frame not in self.ptrace[filename]:
                        self.pframes[filename].append(frame)
                    self.ptrace[filename][frame].append((px1, py1, px2, py2, px3, py3))
            for filename in self.pframes:
                for frame in self.pframes[filename]:
                    self.ptrace[filename][frame] = np.array(self.ptrace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video).astype(np.float32)
        #添加模拟噪声（随机将像素点变为黑色）
        # 如果noise属性不为None，则根据指定的噪声比例，在视频中随机选择一定比例的像素点，并将其值设置为0，表示黑色。
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # 归一化
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # 设置视频长度
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # 收集目标数据
        target = []
        for t in self.target_type:           #根据target_type列表中的每个元素t收集目标数据
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":   #最大帧索引
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(int(self.frames[key][0]))
            elif t == "LargeFrame":  #将视频中的最大帧添加到目标列表中
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
           #####################################################################################################################
            elif t in ["LargeKeypoint", "SmallKeypoint"]:
                if t == "LargeKeypoint":
                    t = self.ptrace[key][self.frames[key][-1]]
                else :
                    t = self.ptrace[key][self.frames[key][0]]

                px1, py1, px2, py2, px3, py3 = t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5]
                target.append(np.array([[px1, py1], [px2, py2], [px3, py3]]))  #target 存储的形式是一个(3,2)形式的数组，其中每一行表示一个关键点的坐标

            elif t in ["LargeTrace", "SmallTrace"]:        #根据t选择相应的跟踪轨迹
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))
                #根据坐标数组 x 和 y 绘制多边形
                # r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int),(video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1

                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))   #将每列添加，按表中列名找到对应值 EF\EDV\ESV

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
