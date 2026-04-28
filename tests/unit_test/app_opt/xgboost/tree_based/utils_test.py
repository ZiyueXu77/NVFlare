# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.app_opt.xgboost.tree_based.utils import update_model


def _make_xgb_model(num_parallel_tree, tree_info):
    num_trees = len(tree_info)
    return {
        "learner": {
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": str(num_parallel_tree),
                        "num_trees": str(num_trees),
                    },
                    "iteration_indptr": [0, num_trees],
                    "tree_info": list(tree_info),
                    "trees": [{"id": i, "marker": f"tree-{i}"} for i in range(num_trees)],
                }
            }
        }
    }


class TestUpdateModel:
    def test_appends_all_trees_when_num_trees_exceeds_num_parallel_tree(self):
        prev_model = _make_xgb_model(num_parallel_tree=2, tree_info=[0, 1, 2, 0, 1, 2])
        model_update = _make_xgb_model(num_parallel_tree=2, tree_info=[0, 1, 2, 0, 1, 2])

        updated_model = update_model(prev_model, model_update)

        xgb_model = updated_model["learner"]["gradient_booster"]["model"]
        assert xgb_model["gbtree_model_param"]["num_trees"] == "12"
        assert xgb_model["iteration_indptr"] == [0, 6, 12]
        assert xgb_model["tree_info"] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        assert [tree["id"] for tree in xgb_model["trees"]] == list(range(12))
