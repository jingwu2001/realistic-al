# Code adapted from: https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/classification/data/ImbalanceCIFAR.py
from typing import List, Tuple

import numpy as np

from .utils import ActiveSubset


def create_imbalanced_dataset(
    dataset, imb_type: str, imb_factor: float
) -> ActiveSubset:
    """Creates an imbalanced dataset from dataset.

    Args:
        dataset (Dataset): Dataset with attribute dataset.targets
        imb_type (str): in [exp, step, balanced]
        imb_factor (float): #samples_min/ #samples_max

    Returns:
        ActiveSubset: Imbalanced dataset class
    """
    targets = np.asarray(dataset.targets)
    _, counts = np.unique(targets, return_counts=True)
    img_num_per_cls = _get_samples_per_cls(counts, imb_type, imb_factor)
    imb_dataset, class_dict = gen_imbalanced_data(dataset, img_num_per_cls)
    for key in class_dict:
        print("Class {} : #Samples {}".format(key, class_dict[key]))
    return imb_dataset


def _get_samples_per_cls(
    counts: np.ndarray, imb_type: str, imb_factor: float
) -> List[int]:
    """Computes the amount of samples for each class given an imbalance setting.

    Args:
        counts (np.ndarray): number of available samples per class
        imb_type (str): exp, step, balanced
        imb_factor (float): #samples_min/ #samples_max

    Returns:
        List[int]: [#samples_max, ... #samples_min]
    """
    cls_num = len(counts)
    img_max = int(counts.max())
    img_min = int(counts.min())
    img_num_per_cls = []
    if imb_type == "exp":
        for cls_idx in range(cls_num):
            exponent = cls_idx / (cls_num - 1.0) if cls_num > 1 else 0
            num = img_max * (imb_factor ** exponent)
            img_num_per_cls.append(min(max(1, int(num)), int(counts[cls_idx])))
    elif imb_type == "step":
        for cls_idx in range(cls_num):
            num = img_max if cls_idx < cls_num // 2 else img_max * imb_factor
            img_num_per_cls.append(min(max(1, int(num)), int(counts[cls_idx])))
    elif imb_type == "balanced":
        img_num_per_cls.extend([img_min] * cls_num)
    else:
        raise NotImplementedError
    return img_num_per_cls


def gen_imbalanced_data(
    dataset: ActiveSubset, img_num_per_cls: List[int]
) -> Tuple[ActiveSubset, dict]:
    targets = [y for x, y in dataset]
    targets_np = np.array(targets, dtype=int)
    classes = np.unique(targets_np)

    num_per_cls_dict = dict()
    selec_idxs = []
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        selec_idxs.append(selec_idx)
        print(
            "Class {} - Needed Samples: {} - Real Samples {}".format(
                the_class, the_img_num, len(selec_idx)
            )
        )
    selec_idxs = np.concatenate(selec_idxs)
    return ActiveSubset(dataset, selec_idxs), num_per_cls_dict
