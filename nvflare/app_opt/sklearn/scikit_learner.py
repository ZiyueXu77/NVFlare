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

# We will move to this app_common when it gets matured
from abc import ABC
from typing import Optional

import pandas as pd
import numpy as np
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.sklearn.data_loader import load_data


class SKLearner(FLComponent, ABC):
    def __init__(self, train_data_path: str, valid_data_path: Optional[str] = None):
        self.fl_ctx = None
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        super().__init__()

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx

    def load_data(self) -> dict:
        train_data = load_data(self.train_data_path)
        valid_data = load_data(self.valid_data_path)
        return {"train": train_data, "valid": valid_data}

    def get_parameters(self, global_param: Optional[dict] = None) -> dict:
        pass

    def train(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        pass

    def evaluate(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        pass

    def log(self, curr_round: int, log_data: Optional[dict] = None) -> None:
        pass

    def save_model(self, curr_round: int, model: Optional[dict] = None) -> None:
        pass
