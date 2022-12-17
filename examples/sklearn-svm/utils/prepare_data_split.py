# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from enum import Enum
from typing import List, Optional

import numpy as np


class SplitMethod(Enum):
    UNIFORM = "uniform"
    LINEAR = "linear"
    SQUARE = "square"
    EXPONENTIAL = "exponential"


class StoreMethod(Enum):
    STORE_INDEX = "store_index"
    STORE_DATA = "store_data"


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate data split")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--site_num", type=int, help="Total number of sites")
    parser.add_argument(
        "--site_name_prefix", type=str, default="site-", help="Name prefix"
    )
    parser.add_argument("--size_total", type=int, help="Total instance number")
    parser.add_argument(
        "--size_valid",
        type=int,
        help="Validation size, the first N to be treated as validation data. "
             "We allow size_valid = size_total, where all data will be used"
             "for both training and validation. Should be used with caution.",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="uniform",
        choices=["uniform", "linear", "square", "exponential"],
        help="How to split the dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="~/dataset",
        help="Output path for the data split json file",
    )
    return parser


def get_split_ratios(site_num: int, split_method: SplitMethod):
    if split_method == SplitMethod.UNIFORM:
        ratio_vec = np.ones(site_num)
    elif split_method == SplitMethod.LINEAR:
        ratio_vec = np.linspace(1, site_num, num=site_num)
    elif split_method == SplitMethod.SQUARE:
        ratio_vec = np.square(np.linspace(1, site_num, num=site_num))
    elif split_method == SplitMethod.EXPONENTIAL:
        ratio_vec = np.exp(np.linspace(1, site_num, num=site_num))
    else:
        raise ValueError(f"Split method {split_method.name} not implemented!")

    return ratio_vec


def split_num_proportion(n, site_num, split_method: SplitMethod) -> List[int]:
    split = []
    ratio_vec = get_split_ratios(site_num, split_method)
    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def assign_data_index_to_sites(data_size: int,
                               valid_fraction: float,
                               num_sites: int,
                               site_name_prefix: str,
                               split_method: SplitMethod = SplitMethod.UNIFORM) -> dict:
    if valid_fraction == 1.0:
        raise ValueError("validation percent should be less than 100% of the total data")

    valid_size = int(round(data_size * valid_fraction, 0))
    train_size = data_size - valid_size

    site_sizes = split_num_proportion(train_size, num_sites, split_method)
    split_data_indices = {
        "valid": {"start": 0, "end": valid_size},
    }
    for site in range(num_sites):
        site_id = site_name_prefix + str(site + 1)
        idx_start = valid_size + sum(site_sizes[:site])
        idx_end = valid_size + sum(site_sizes[: site + 1])
        split_data_indices[site_id] = {"start": idx_start, "end": idx_end}

    return split_data_indices


def get_lines(fp, r: range):
    # read by line numbers
    return [x for i, x in enumerate(fp) if i in r]


def get_file_line_count(input_path: str) -> int:
    count = 0
    with open(input_path, "r") as fp:
        for i, _ in enumerate(fp):
            count += 1
    return count


def save_lines(input_path, output_file: str, site_range: range):
    with open(input_path, 'r') as fp:
        lines = get_lines(fp, site_range)
        with open(output_file, "w") as ofp:
            for one_line in lines:
                ofp.write(one_line)


def save_indices(data_path: str,
                 site_indices: dict,
                 output_dir: str,
                 filename: str = "data_split.json"):
    json_data = {
        "data_path": data_path,
    }
    json_data.update({"data_index": site_indices})

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, filename)
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=4)


def save_sites_data(input_path: str,
                    output_dir: str,
                    filename: str,
                    sites: list,
                    site_indices: dict,
                    output_file_format: str = "csv"):
    for site in sites:
        di = site_indices[site]
        output_file = os.path.join(output_dir, f"{site}/{filename}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if output_file_format == "csv":
            site_range = range(di["start"], di["end"])
            save_lines(input_path, output_file, site_range)
        else:
            raise NotImplementedError


def save_split_data(site_indices: dict,
                    input_path: str,
                    output_dir: str,
                    filename: str,
                    store_method: StoreMethod = StoreMethod.STORE_DATA,
                    output_file_format="csv"):
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.rmdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # sites and "valid"
    index_keys = [x for x in site_indices]
    if store_method == StoreMethod.STORE_DATA:
        save_sites_data(input_path, output_dir, filename, index_keys, site_indices, output_file_format)
    elif store_method == StoreMethod.STORE_INDEX:
        save_indices(input_path, site_indices, output_dir)
    else:
        raise NotImplementedError


def get_file_format(ext: str) -> str:
    if ext is None or ext == "" or ext.isspace():
        return "csv"
    elif ext.startswith("."):
        return ext[1:]
    else:
        return ext


def split_data(data_path: str,
               output_dir: str,
               site_num: int,
               valid_frac: float,
               site_name_prefix: str = "site-",
               split_method: SplitMethod = SplitMethod.UNIFORM,
               store_method: StoreMethod = StoreMethod.STORE_INDEX,
               ):
    size_total = get_file_line_count(data_path)
    site_indices = assign_data_index_to_sites(size_total,
                                              valid_frac,
                                              site_num,
                                              site_name_prefix,
                                              split_method)

    from nvflare.app_common.utils.file_utils import get_file_format
    file_format = get_file_format(data_path)
    filename = os.path.basename(data_path)

    save_split_data(site_indices=site_indices,
                    input_path=data_path,
                    output_dir=output_dir,
                    filename=filename,
                    store_method=store_method,
                    output_file_format=file_format
                    )


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    valid_frac = args.size_valid / args.size_total
    split_method = SplitMethod(args.split_method)

    split_data(args.data_path,
               args.out_path,
               args.site_num,
               valid_frac,
               args.site_name_prefix,
               split_method,
               StoreMethod.STORE_INDEX)


if __name__ == "__main__":
    main()
