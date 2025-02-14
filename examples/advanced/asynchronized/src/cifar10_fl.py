# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os

import numpy as np
import torch
import torch.optim as optim
from cifar10_nets import ModerateCNN
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from utils.cifar10_data_utils import CIFAR10_ROOT
from utils.cifar10_dataset import CIFAR10_Idx

import nvflare.client as flare

CIFAR10_IDX_ROOT = "/tmp/nvflare/dataset/cifar10_idx"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_idx_root",
        type=str,
        default=CIFAR10_IDX_ROOT,
    )
    parser.add_argument(
        "--epoch_local",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    return parser.parse_args()


def main():
    args = define_parser()

    # Training params
    train_idx_root = args.train_idx_root
    epoch_local = args.epoch_local
    lr = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Local model file paths
    local_model_file = "local_model.pt"
    best_global_model_file = "best_global_model_file.pt"
    best_acc = 0.0

    # Training components
    model = ModerateCNN()
    writer = SummaryWriter("./")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )
    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    # Initializes NVFlare client API and get site_name from flare
    flare.init()
    sys_info = flare.system_info()
    site_name = sys_info["site_name"]

    # Dataset
    # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
    site_idx_file_name = os.path.join(train_idx_root, site_name + ".npy")
    print(f"IndexList Path: {site_idx_file_name}")
    if os.path.exists(site_idx_file_name):
        print("Loading subset index")
        site_idx = np.load(site_idx_file_name).tolist()
    else:
        raise ValueError(f"No subset index found! File {site_idx_file_name} does not exist!")
    train_dataset = CIFAR10_Idx(
        root=CIFAR10_ROOT,
        data_idx=site_idx,
        train=True,
        download=False,
        transform=transform_train,
    )
    valid_dataset = datasets.CIFAR10(
        root=CIFAR10_ROOT,
        train=False,
        download=False,
        transform=transform_valid,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Train federated rounds
    # start with global model at the beginning of each round
    while flare.is_running():
        # receives FLModel from NVFlare
        global_model = flare.receive()
        curr_round = global_model.current_round
        epoch_global = epoch_local * curr_round
        print(f"current_round={curr_round}")

        # Load global model params to local model
        model.load_state_dict(global_model.params)
        model = model.to(DEVICE)

        # wraps evaluation logic into a method to re-use for
        # evaluation on both trained and received model
        def evaluate(tb_id):
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for _i, (inputs, labels) in enumerate(valid_loader):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    _, pred_label = torch.max(outputs.data, 1)

                    total += inputs.data.size()[0]
                    correct += (pred_label == labels.data).sum().item()
                metric = correct / float(total)
                if tb_id:
                    writer.add_scalar(tb_id, metric, epoch_global)
            return metric

        # evaluate on received global model
        val_acc = evaluate("global_val_acc")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_global_model_file)

        for epoch in range(epoch_local):
            model.train()
            epoch_len = len(train_loader)
            print(f"Local epoch {site_name}: {epoch + 1}/{epoch_local} (lr={lr})")
            avg_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                current_step = epoch_len * epoch_global + i
                avg_loss += loss.item()
            writer.add_scalar("train_loss", avg_loss / len(train_loader), current_step)

        # evaluate on trained model
        val_acc_local = evaluate("local_val_acc")
        torch.save(model.state_dict(), local_model_file)

        # construct trained FL model
        out_param = model.cpu().state_dict()
        output_model = flare.FLModel(
            params=out_param,
            metrics={"eval_acc": val_acc_local},
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_loader) * epoch_local},
        )
        # send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
