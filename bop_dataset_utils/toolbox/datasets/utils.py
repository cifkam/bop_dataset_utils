"""Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Library
from typing import Dict, List

# Third Party
import numpy as np


def make_detections_from_segmentation(
    segmentations: np.ndarray,
) -> List[Dict[int, np.ndarray]]:
    """segmentations: (n, h, w) int np.ndarray."""
    assert segmentations.ndim == 3
    detections = []
    for segmentation_n in segmentations:
        dets_n = {}
        for unique_id in np.unique(segmentation_n):
            ids = np.where(segmentation_n == unique_id)
            x1, y1, x2, y2 = (
                np.min(ids[1]),
                np.min(ids[0]),
                np.max(ids[1]),
                np.max(ids[0]),
            )
            dets_n[int(unique_id)] = np.array([x1, y1, x2, y2])
        detections.append(dets_n)
    return detections



def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !.
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4,)
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.tensor(orig_size, dtype=torch.float)
    crop_resize = torch.tensor(crop_resize, dtype=torch.float)

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K



def crop_to_aspect_ratio(images, box, masks=None, K=None):
    assert images.dim() == 4
    bsz, _, h, w = images.shape
    assert box.dim() == 1
    assert box.shape[0] == 4
    w_output, h_output = box[[2, 3]] - box[[0, 1]]
    boxes = torch.cat(
        (
            torch.arange(bsz).unsqueeze(1).to(box.device).float(),
            box.unsqueeze(0).repeat(bsz, 1).float(),
        ),
        dim=1,
    ).to(images.device)
    images = torchvision.ops.roi_pool(images, boxes, output_size=(h_output, w_output))
    if masks is not None:
        assert masks.dim() == 4
        masks = torchvision.ops.roi_pool(masks, boxes, output_size=(h_output, w_output))
    if K is not None:
        assert K.dim() == 3
        assert K.shape[0] == bsz
        K = get_K_crop_resize(
            K,
            boxes[:, 1:],
            orig_size=(h, w),
            crop_resize=(h_output, w_output),
        )
    return images, masks, K


def make_detections_from_segmentation(masks):
    detections = []
    if masks.dim() == 4:
        assert masks.shape[0] == 1
        masks = masks.squeeze(0)

    for mask_n in masks:
        dets_n = {}
        for uniq in torch.unique(mask_n, sorted=True):
            ids = np.where((mask_n == uniq).cpu().numpy())
            x1, y1, x2, y2 = (
                np.min(ids[1]),
                np.min(ids[0]),
                np.max(ids[1]),
                np.max(ids[0]),
            )
            dets_n[int(uniq.item())] = torch.tensor([x1, y1, x2, y2]).to(mask_n.device)
        detections.append(dets_n)
    return detections


def make_masks_from_det(detections, h, w):
    n_ids = len(detections)
    detections = torch.as_tensor(detections)
    masks = torch.zeros((n_ids, h, w)).byte()
    for mask_n, det_n in zip(masks, detections):
        x1, y1, x2, y2 = det_n.cpu().int().tolist()
        mask_n[y1:y2, x1:x2] = True
    return masks
