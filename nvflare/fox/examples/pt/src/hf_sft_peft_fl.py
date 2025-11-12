# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import random

import datasets
import numpy as np
import torch
from transformers import AutoModelForCausalLM, TrainerCallback, trainer_utils
from trl import SFTConfig, SFTTrainer

import nvflare.client as flare

# Set deterministic seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class StopCallback(TrainerCallback):
    """Callback to stop training at each epoch."""

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


def format_instruction(example):
    """Format instruction for training."""
    return f"### Instruction: Generate Output according to the information and question given by Input. ### Input:{example['input']} ### Response: {example['output']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
    )
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/training.jsonl",
    )
    parser.add_argument(
        "--data_path_valid",
        type=str,
        default="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/validation.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="opt-125m-sft-fl",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="learning rate scheduler type, default to 'constant'",
    )
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=3)
    args = parser.parse_args()

    # Dataset
    dataset_train = datasets.load_dataset("json", data_files=args.data_path_train, split="train")
    dataset_valid = datasets.load_dataset("json", data_files=args.data_path_valid, split="train")
    # Print dataset info
    print(f"Dataset size: training {len(dataset_train)}, validation {len(dataset_valid)}")
    # record every 5% of the dataset
    batch_size = 4
    gra_accu_steps = 10
    logging_steps = int(len(dataset_train) / (20 * batch_size * gra_accu_steps))
    print(f"logging_steps: {logging_steps}")

    # Model configs
    model_name_or_path = args.model_name_or_path
    peft_config = None

    # Load model
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="cuda:0",
        use_cache=False,
        dtype=torch.bfloat16,
    )
    torch.set_default_dtype(default_dtype)
    model.config.pretraining_tp = 1

    # Training arguments
    train_args = SFTConfig(
        output_dir=args.output_path,
        # Using callback, stop at each epoch, so specify num_train_epochs
        # the same as the total epoch in one-call training
        num_train_epochs=args.local_epoch * args.num_rounds,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=5e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs={"num_cycles": 2} if args.lr_scheduler == "cosine_with_restarts" else {},
        disable_tqdm=True,
        max_length=1024,
        save_total_limit=2,
        save_safetensors=False,
        seed=0,
        data_seed=0,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        formatting_func=format_instruction,
        args=train_args,
        callbacks=[StopCallback()],
    )

    # initializes NVFlare client API
    flare.init()

    # Train federated rounds
    # start with global model at the beginning of each round
    while flare.is_running():
        # receives golobal model from NVFlare
        input_model = flare.receive()
        curr_round = input_model.current_round
        print(f"current_round={curr_round}")
        # Update the key name received from global model and convert numpy to tensor if needed
        global_model = input_model.params
        for key in list(global_model.keys()):
            value = global_model.pop(key)
            # Convert numpy arrays to tensors if needed
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            # Update key name (remove "model." prefix)
            global_model[key.replace("model.", "", 1)] = value

        # Load state dict
        trainer.model.load_state_dict(global_model)

        # Evaluate the global model
        eval_loss = trainer.evaluate()
        eval_loss = float(eval_loss["eval_loss"])

        # Train
        if curr_round == 0:
            # First round, start from scratch
            for epoch in range(args.local_epoch):
                print(f"Training local epoch {epoch + 1}/{args.local_epoch}")
                # train for one epoch
                if epoch == 0:
                    trainer.train()
                else:
                    # continue training
                    trainer.train(resume_from_checkpoint=True)
        else:
            # Subsequent rounds - replace local resume weights with global weights
            resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
            trainer.model.save_pretrained(
                resume_from_checkpoint_folder, state_dict=global_model, safe_serialization=False
            )

            # continue training
            for epoch in range(args.local_epoch):
                print(f"Training local epoch {epoch + 1}/{args.local_epoch}")
                trainer.train(resume_from_checkpoint=True)

        # compose output model to send back to server (only on main process)
        out_param = trainer.model.state_dict()

        # update the key name sent to global model
        for key in list(out_param.keys()):
            out_param["model." + key] = out_param.pop(key).cpu()
        # print the dict size
        print(f"In total {len(out_param.keys())} params to be sent to server.")

        # construct trained FL model
        output_model = flare.FLModel(
            params=out_param,
            metrics={"eval_loss": eval_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": trainer.train_dataset.num_rows},
        )
        # send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
