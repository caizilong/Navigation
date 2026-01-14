import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple, List

import torch
from torch.utils.data import Sampler, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training


class InterleavedSampler(Sampler):
    """
    交错采样器：将多个数据集的样本按块（chunk）交错采样，
    而不是完全随机采样。这样可以显著减少数据集之间的切换频率，
    提高页面缓存命中率，从而提升多数据集联合训练时的 GPU 利用率。

    工作原理：
    1. 将每个数据集的索引分成多个 chunk
    2. 随机打乱 chunk 的顺序
    3. 在每个 chunk 内部随机打乱样本顺序
    4. 这样每个 chunk 内的样本来自同一个数据集，减少 LMDB 切换
    """

    def __init__(self, concat_dataset: ConcatDataset, chunk_size: int = 256, shuffle: bool = True):
        """
        Args:
            concat_dataset: ConcatDataset 对象
            chunk_size: 每个块的大小，越大则数据集切换越少，但随机性越差
            shuffle: 是否打乱
        """
        self.concat_dataset = concat_dataset
        self.chunk_size = chunk_size
        self.shuffle = shuffle

        # 获取每个子数据集的大小和起始索引
        self.dataset_sizes = []
        self.dataset_offsets = [0]

        cumulative_size = 0
        for dataset in concat_dataset.datasets:
            size = len(dataset)
            self.dataset_sizes.append(size)
            cumulative_size += size
            self.dataset_offsets.append(cumulative_size)

        self.total_size = cumulative_size

    def __iter__(self):
        # 为每个数据集创建 chunk
        all_chunks = []

        for dataset_idx, (offset, size) in enumerate(zip(self.dataset_offsets[:-1], self.dataset_sizes)):
            # 生成该数据集的所有索引
            indices = list(range(offset, offset + size))

            if self.shuffle:
                # 先打乱该数据集内部的索引
                np.random.shuffle(indices)

            # 分成 chunks
            for i in range(0, len(indices), self.chunk_size):
                chunk = indices[i:i + self.chunk_size]
                all_chunks.append(chunk)

        if self.shuffle:
            # 打乱 chunks 的顺序（保持 chunk 内部顺序）
            np.random.shuffle(all_chunks)

        # 展平所有 chunks
        for chunk in all_chunks:
            for idx in chunk:
                yield idx

    def __len__(self):
        return self.total_size


def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    # 处理yaw可能是数组的情况
    if isinstance(yaw, np.ndarray):
        yaw = float(yaw.item()) if yaw.size == 1 else float(yaw[0])
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        # crop to the right ratio
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        # crop to the right ratio
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)
