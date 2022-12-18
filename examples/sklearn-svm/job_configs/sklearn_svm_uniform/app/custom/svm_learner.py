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
from typing import Optional

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.sklearn.sklearner import SKLearner


class SVMLearner(SKLearner):

    def __init__(self, root_data_path: str, filename: str):
        super().__init__(root_data_path, filename)
        self.train_data = None
        self.valid_data = None

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx

        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]

    def get_parameters(self, global_param: Optional[dict] = None) -> dict:
        if global_param:
            global_support_x = global_param["support_x"]
            global_support_y = global_param["support_y"]
            return {"model": (global_support_x, global_support_y)}
        else:
            return {}

    def train(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        svm = SVC(kernel="rbf")
        (x_train, y_train, train_size) = self.train_data
        svm.fit(x_train, y_train)
        index = svm.support_
        local_support_x = x_train[index]
        local_support_y = y_train[index]
        return {"model": (local_support_x, local_support_y, svm)}

    def evaluate(self, curr_round: int, global_param: Optional[dict] = None) -> dict:
        # local validation with global center
        # fit a standalone SVM with the global support vectors
        svm_global = SVC(kernel="rbf")
        global_model = self.get_parameters(global_param)
        metrics = {}
        if "model" in global_model:
            (support_x, support_y) = global_model["model"]
            svm_global.fit(support_x, support_x)
            (x_valid, y_valid, valid_size) = self.valid_data
            y_pred = svm_global.predict(x_valid)
            auc = roc_auc_score(y_valid, y_pred)
            metrics = {"AUC", auc}

        return metrics

    def save_model(self, curr_round: int, model: Optional[dict] = None) -> None:
        # # save global model
        # dump(svm_global, self.global_model_path)
        pass

    def finalize(self) -> None:
        # freeing resources in finalize
        del self.train_data
        del self.valid_data
        self.log_info(self.fl_ctx, "Freed training resources")
