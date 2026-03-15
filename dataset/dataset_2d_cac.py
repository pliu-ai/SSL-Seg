"""
2D CAC (Context-Aware Consistency) Dataset for C3PS training.
Loads 2D h5 slices and generates overlapping patch pairs.
"""
import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class ACDCDatasetCAC(Dataset):
    """
    2D CAC dataset for ACDC.
    Each item returns two overlapping 2D patches from the same slice,
    along with overlap coordinates and condition labels.

    Expected directory structure:
        base_dir/data/{case_name}_slice_{idx}.h5   (train slices)
        base_dir/data/{case_name}.h5               (val volumes)
        base_dir/train_slices.list
        base_dir/val.list
    """

    def __init__(self, base_dir, patch_size=(224, 224), num_class=4,
                 stride=2, iou_bound=(0.1, 0.9), labeled_num=68,
                 con_list=None, addi_con_list=None):
        self.base_dir = base_dir
        self.patch_size = patch_size
        self.num_class = num_class
        self.stride = stride
        self.iou_bound = iou_bound
        self.labeled_num = labeled_num
        self.con_list = con_list or list(range(1, num_class))
        self.addi_con_list = addi_con_list or []

        list_path = os.path.join(base_dir, 'train_slices.list')
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()
                                if line.strip()]

        print(f"[ACDCDatasetCAC] {len(self.sample_list)} slices, "
              f"labeled_num={labeled_num}, patch_size={patch_size}, "
              f"con_list={self.con_list}")

    def __len__(self):
        return len(self.sample_list)

    def _load_slice(self, idx):
        case = self.sample_list[idx]
        h5_path = os.path.join(self.base_dir, 'data', 'slices',
                               f'{case}.h5')
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)
        return image, label

    def _pad_if_needed(self, image, label):
        h, w = image.shape
        ph = max(self.patch_size[0] - h, 0)
        pw = max(self.patch_size[1] - w, 0)
        if ph > 0 or pw > 0:
            image = np.pad(image, ((0, ph), (0, pw)), mode='constant')
            label = np.pad(label, ((0, ph), (0, pw)), mode='constant')
        return image, label

    def _get_overlap_patches(self, image, label):
        """Crop two overlapping 2D patches and compute overlap coordinates."""
        H, W = image.shape
        ph, pw = self.patch_size

        ub_y = H - ph
        ub_x = W - pw

        y1 = np.random.randint(0, max(ub_y, 0) + 1)
        x1 = np.random.randint(0, max(ub_x, 0) + 1)

        max_iters = 50
        y2, x2 = y1, x1
        for _ in range(max_iters):
            y2_cand = np.random.randint(0, max(ub_y, 0) + 1)
            x2_cand = np.random.randint(0, max(ub_x, 0) + 1)
            y2_cand = (y2_cand - y1) // self.stride * self.stride + y1
            x2_cand = (x2_cand - x1) // self.stride * self.stride + x1
            if y2_cand < 0:
                y2_cand += self.stride
            if x2_cand < 0:
                x2_cand += self.stride

            oy = ph - abs(y2_cand - y1)
            ox = pw - abs(x2_cand - x1)
            if oy <= 0 or ox <= 0:
                continue
            inter = oy * ox
            union = 2 * ph * pw - inter
            iou = inter / union
            if self.iou_bound[0] <= iou <= self.iou_bound[1]:
                y2, x2 = y2_cand, x2_cand
                break

        img1 = image[y1:y1 + ph, x1:x1 + pw]
        img2 = image[y2:y2 + ph, x2:x2 + pw]
        lbl1 = label[y1:y1 + ph, x1:x1 + pw]
        lbl2 = label[y2:y2 + ph, x2:x2 + pw]

        # overlap coordinates in each patch's local frame
        ol1_ul = [max(0, y2 - y1), max(0, x2 - x1)]
        ol1_br = [min(ph, ph + y2 - y1), min(pw, pw + x2 - x1)]
        ol2_ul = [max(0, y1 - y2), max(0, x1 - x2)]
        ol2_br = [min(ph, ph + y1 - y2), min(pw, pw + x1 - x2)]

        return img1, img2, lbl1, lbl2, ol1_ul, ol1_br, ol2_ul, ol2_br

    def _pick_condition(self, label_arr):
        labels = list(np.unique(label_arr))
        if 0 in labels:
            labels.remove(0)
        inter = list(set(labels) & set(self.con_list))
        if not inter:
            inter = self.con_list
        return int(np.random.choice(inter))

    def __getitem__(self, idx):
        image, label = self._load_slice(idx)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image, label = self._pad_if_needed(image, label)

        img1, img2, lbl1, lbl2, ul1, br1, ul2, br2 = \
            self._get_overlap_patches(image, label)

        if idx >= self.labeled_num:
            lbl1 = np.zeros_like(lbl1)
            lbl2 = np.zeros_like(lbl2)

        cond1 = self._pick_condition(lbl1)
        cond_choices = list(set(self.con_list) | set(self.addi_con_list))
        labels2 = list(np.unique(lbl2))
        if 0 in labels2:
            labels2.remove(0)
        inter2 = list(set(labels2) & set(cond_choices))
        if not inter2:
            inter2 = cond_choices
        cond2 = int(np.random.choice(inter2))

        # (2, 1, H, W) and (2, H, W) matching the 3D CAC format
        img_tensor = torch.FloatTensor(
            np.stack([img1[np.newaxis], img2[np.newaxis]], axis=0)
        )
        lbl_tensor = torch.LongTensor(
            np.stack([lbl1, lbl2], axis=0)
        )
        condition = torch.LongTensor([cond1, cond2])

        return {
            'image': img_tensor,
            'label': lbl_tensor,
            'ul1': ul1,
            'br1': br1,
            'ul2': ul2,
            'br2': br2,
            'condition': condition,
        }


class ACDCDatasetVal(Dataset):
    """Validation dataset: loads full volumes for slice-by-slice evaluation."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        list_path = os.path.join(base_dir, 'val.list')
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()
                                if line.strip()]
        print(f"[ACDCDatasetVal] {len(self.sample_list)} validation cases")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5_path = os.path.join(self.base_dir, 'data', f'{case}.h5')
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:].astype(np.float32)
            label = f['label'][:].astype(np.uint8)
        return {'image': image, 'label': label}
