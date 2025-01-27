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
import io
import json
from collections import defaultdict
from functools import partial
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Third Party
import imageio
import numpy as np
import pandas as pd
import webdataset as wds
from bop_dataset_utils._bop_toolkit_lib.dataset import bop_webdataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from bop_dataset_utils.datasets.bop_scene_dataset import data_from_bop_obs

# MegaPose
from bop_dataset_utils.datasets.samplers import ListSampler
from bop_dataset_utils.datasets.scene_dataset import (
    CameraData,
    DataJsonType,
    IterableSceneDataset,
    ObjectData,
    ObservationInfos,
    SceneDataset,
    SceneObservation,
)
from bop_dataset_utils.utils.webdataset import tarfile_to_samples


def simple_collate_fn(batch: Any) -> Any:
    return batch


def write_scene_ds_as_wds(
    scene_ds: SceneDataset,
    wds_dir: Path,
    n_reading_workers: int = 8,
    maxcount: int = 1000,
    shard_format: str = "shard-%08d.tar",
    keep_labels_set: Optional[Set] = None,
    n_max_frames: Optional[int] = None,
    frame_ids: Optional[List[int]] = None,
    depth_scale: int = 1000,
) -> None:
    assert scene_ds.frame_index is not None

    wds_dir.mkdir(exist_ok=True, parents=True)
    frame_index = scene_ds.frame_index.copy()
    shard_writer = wds.ShardWriter(
        str(wds_dir / shard_format),
        maxcount=maxcount,
        start_shard=0,
    )

    sampler = None
    n_frames = len(scene_ds)
    if frame_ids is not None:
        sampler = ListSampler(frame_ids)
        n_frames = len(frame_ids)

    iterator = DataLoader(
        scene_ds,
        num_workers=n_reading_workers,
        batch_size=1,
        collate_fn=simple_collate_fn,
        shuffle=False,
        sampler=sampler,
    )

    n_frames = 0
    for idx, data in tqdm(enumerate(iterator), total=n_frames):
        obs: SceneObservation = data[0]
        assert obs.rgb is not None

        if keep_labels_set is not None:
            assert obs.object_datas is not None
            object_labels = {obj.label for obj in obs.object_datas}
            n_objects_valid = len(object_labels.intersection(keep_labels_set))
            if n_objects_valid == 0:
                continue

        key = sha1(obs.rgb.data).hexdigest()
        sample: Dict[str, Any] = {
            "__key__": key,
        }
        if obs.rgb is not None:
            sample["rgb.png"] = obs.rgb
        if obs.segmentation is not None:
            sample["segmentation.png"] = obs.segmentation
        if obs.depth is not None:
            sample["depth.png"] = np.array(obs.depth * depth_scale, dtype=np.int32)
        if obs.infos is not None:
            sample["infos.json"] = obs.infos.to_json()
        if obs.object_datas is not None:
            sample["object_datas.json"] = [obj.to_json() for obj in obs.object_datas]
        if obs.camera_data is not None:
            sample["camera_data.json"] = obs.camera_data.to_json()

        shard_writer.write(sample)
        n_frames += 1
        frame_index.loc[idx, "key"] = key
        frame_index.loc[idx, "shard_fname"] = Path(shard_writer.fname).name
        if n_max_frames is not None and n_frames > n_max_frames:
            break
    frame_index = frame_index.loc[:, ["scene_id", "view_id", "key", "shard_fname"]]
    shard_writer.close()
    frame_index.to_feather(wds_dir / "frame_index.feather")
    ds_infos = {
        "depth_scale": depth_scale,
    }
    (wds_dir / "infos.json").write_text(json.dumps(ds_infos))
    return


def load_scene_ds_obs(
    sample: Dict[str, Union[bytes, str]],
    depth_scale: float = 1000.0,
    load_depth: bool = False,
    label_format: str = "{label}",
) -> SceneObservation:
    assert isinstance(sample["rgb.png"], bytes)
    assert isinstance(sample["segmentation.png"], bytes)
    assert isinstance(sample["depth.png"], bytes)
    assert isinstance(sample["camera_data.json"], bytes)
    assert isinstance(sample["infos.json"], bytes)

    rgb = np.array(imageio.imread(io.BytesIO(sample["rgb.png"])))
    segmentation = np.array(imageio.imread(io.BytesIO(sample["segmentation.png"])))
    segmentation = np.asarray(segmentation, dtype=np.uint32)
    depth = None
    if load_depth:
        depth = imageio.imread(io.BytesIO(sample["depth.png"]))
        depth = np.asarray(depth, dtype=np.float32)
        depth /= depth_scale

    object_datas_json: List[DataJsonType] = json.loads(sample["object_datas.json"])
    object_datas = [ObjectData.from_json(d) for d in object_datas_json]
    for obj in object_datas:
        obj.label = label_format.format(label=obj.label)

    camera_data = CameraData.from_json(sample["camera_data.json"])
    infos = ObservationInfos.from_json(sample["infos.json"])

    return SceneObservation(
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
        infos=infos,
        object_datas=object_datas,
        camera_data=camera_data,
    )


class WebSceneDataset(SceneDataset):
    def __init__(
        self,
        wds_dir: Path,
        load_depth: bool = True,
        load_segmentation: bool = True,
        label_format: str = "{label}",
        load_frame_index: bool = False,
    ):
        self.label_format = label_format
        self.wds_dir = wds_dir

        frame_index = None
        if load_frame_index:
            key_to_shard = json.loads((wds_dir / "key_to_shard.json").read_text())
            frame_index = defaultdict(list)
            for key, shard_id in key_to_shard.items():
                image_id, scene_id = map(int, key.split("_"))
                frame_index["image_id"].append(image_id)
                frame_index["scene_id"].append(scene_id)
                frame_index["key"].append(key)
                frame_index["shard_id"].append(shard_id)
            frame_index = pd.DataFrame(frame_index)

        super().__init__(
            frame_index=frame_index,
            load_depth=load_depth,
            load_segmentation=load_segmentation,
        )

    def get_tar_list(self) -> List[str]:
        tar_files = [str(x) for x in self.wds_dir.iterdir() if x.suffix == ".tar"]
        tar_files.sort()
        return tar_files

    def __getitem__(self, idx: int) -> SceneObservation:
        assert self.frame_index is not None
        row = self.frame_index.iloc[idx]
        shard_id, key = row.shard_id, row.key
        shard_path = self.wds_dir / f"shard-{shard_id:06d}.tar"

        bop_obs = bop_webdataset.load_image_data(
            shard_path,
            key,
            load_rgb=True,
            load_mask_visib=True,
            load_gt=True,
            load_gt_info=True,
        )
        obs = data_from_bop_obs(bop_obs, use_raw_object_id=True)
        return obs


class IterableWebSceneDataset(IterableSceneDataset):
    def __init__(self, web_scene_dataset: WebSceneDataset, buffer_size: int = 1):
        self.web_scene_dataset = web_scene_dataset

        load_scene_ds_obs_ = partial(
            load_scene_ds_obs,
            depth_scale=self.web_scene_dataset.depth_scale,
            load_depth=self.web_scene_dataset.load_depth,
            label_format=self.web_scene_dataset.label_format,
        )

        def load_scene_ds_obs_iterator(
            samples,
        ):
            for sample in samples:
                yield load_scene_ds_obs_(sample)

        self.datapipeline = wds.DataPipeline(
            wds.ResampledShards(self.web_scene_dataset.get_tar_list()),
            tarfile_to_samples(),
            load_scene_ds_obs_iterator,
            wds.shuffle(buffer_size),
        )

    def __iter__(self):
        return iter(self.datapipeline)
