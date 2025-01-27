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
import datetime
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


# Third Party
import torch
import torch.distributed as dist

# MegaPose
from bop_dataset_utils.utils.logging import get_logger

logger = get_logger(__name__)


def get_tmp_dir() -> Path:
    if "JOB_DIR" in os.environ:
        tmp_dir = Path(os.environ["JOB_DIR"]) / "tmp"
    else:
        tmp_dir = Path("/tmp/megapose_job")
    tmp_dir.parent.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def get_rank() -> int:
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()
    return rank


def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        world_size = 1
    else:
        world_size = torch.distributed.get_world_size()
    return world_size


def reduce_dict(
    input_dict: Dict[str, Any],
    average: bool = True,
) -> Dict[str, Any]:
    """https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.tensor(values).float().cuda()
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    return reduced_dict


def init_distributed_mode() -> None:
    assert torch.cuda.device_count() == 1
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(12345)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"Rank: {rank}, World size: {world_size}")
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=4 * 1800),  # 2 hours
    )
    torch.distributed.barrier()
