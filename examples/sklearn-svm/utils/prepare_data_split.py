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
        raise ValueError("Split method not implemented!")

    return ratio_vec


def split_num_proportion(n, site_num, split_method: SplitMethod) -> list[int]:
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
                               site_name_prefix: str = "site-",
                               split_method: SplitMethod = SplitMethod.UNIFORM) -> dict:
    if valid_fraction == 1.0:
        raise ValueError("validation percent should be less than 100% of the total data")

    valid_size = round(data_size * valid_fraction, 0)

    train_size = data_size - valid_size

    site_sizes = split_num_proportion(train_size, num_sites, split_method)

    split_data_indices = {
        "data_index": {"valid": {"start": 0, "end": valid_size}},
    }
    for site in range(num_sites):
        site_id = site_name_prefix + str(site + 1)
        idx_start = valid_size + sum(site_sizes[:site])
        idx_end = valid_size + sum(site_sizes[: site + 1])
        split_data_indices["data_index"][site_id] = {"start": idx_start, "end": idx_end}

    return split_data_indices


def save_split_data(data_indices: dict,
                    output_dir: str,
                    store_method: StoreMethod = StoreMethod.STORE_DATA,
                    file_forma="csv"):
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.rmdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sites = [x for x in data_indices["data_index"] if x != "valid"]
    if store_method == StoreMethod.STORE_DATA:
        if file_forma == "csv":
            for site in sites:
                output_file = os.path.join(output_dir, f"data_{site}.json")
                with open(output_file, "w") as f:
                     f.readline()
        else:
            raise  NotImplementedError
    elif store_method == StoreMethod.STORE_DATA:
        pass
    else:
        raise NotImplementedError


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    json_data = {
        "data_path": args.data_path,
        "data_index": {"valid": {"start": 0, "end": args.size_valid}},
    }

    valid_frac = args.size_valid / args.size_total
    split_method = SplitMethod(args.split_method)

    r = assign_data_index_to_sites(args.size_total,
                                   valid_frac,
                                   args.site_num,
                                   args.site_name_prefix,
                                   split_method)
    json_data.update(r)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    for site in range(args.site_num):
        output_file = os.path.join(
            args.out_path, f"data_{args.site_name_prefix}{site + 1}.json"
        )
        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()
