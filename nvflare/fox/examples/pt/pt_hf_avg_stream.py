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

import logging
import os
import random

import datasets
import numpy as np
import torch
from transformers import AutoModelForCausalLM, TrainerCallback, trainer_utils
from trl import SFTConfig, SFTTrainer

from nvflare.fox.api.app import ClientApp, ServerApp
from nvflare.fox.api.constants import ContextKey, EnvType
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab
from nvflare.fox.api.group import all_clients
from nvflare.fox.api.strategy import Strategy
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.utils import parse_state_dict
from nvflare.fox.sim.simulator import Simulator
from nvflare.fox.sys.tensor_downloader import download_state_dict, prepare_for_download
from nvflare.fuel.utils.log_utils import get_obj_logger

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


class _AggrResult:
    """Aggregation result container."""

    def __init__(self):
        self.total = {}
        self.count = 0
        self.metrics = {}
        self.client_names = []  # Track which clients responded


class HFFedAvgStream(Strategy):
    """HuggingFace Federated Averaging Strategy with streaming support."""

    def __init__(self, model_name_or_path, num_rounds=3, timeout=10.0, num_tensors_per_chunk=1, min_clients=None):
        self.num_rounds = num_rounds
        self.model_name_or_path = model_name_or_path
        self.timeout = timeout
        self.num_tensors_per_chunk = num_tensors_per_chunk
        self.min_clients = min_clients  # Minimum number of clients required for aggregation
        self.name = "HFFedAvgStream"
        self.logger = get_obj_logger(self)

        # Load initial model
        self.logger.info(f"Loading initial model from {model_name_or_path}")
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="cpu",
            use_cache=False,
            dtype=torch.bfloat16,
        )
        torch.set_default_dtype(default_dtype)
        model.config.pretraining_tp = 1

        initial_model = model.state_dict()
        self.logger.info(f"Loaded model with {len(initial_model)} parameters")
        self._init_model = parse_state_dict(initial_model)

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        global_model = context.get_prop(ContextKey.INPUT, self._init_model)
        for round_num in range(self.num_rounds):
            self.logger.info(f"[{context.header_str()}] Starting round {round_num + 1}/{self.num_rounds}")
            global_model = self._do_one_round(round_num, global_model, context)

        self.logger.info(f"Training completed after {self.num_rounds} rounds")

        # Save final model to server workspace
        workspace = context.workspace
        if workspace and global_model is not None:
            output_dir = os.path.join(workspace.get_work_dir(), "final_model")
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "global_model.pt")
            torch.save(global_model, model_path)
            self.logger.info(f"Saved final global model to {model_path}")

        return global_model

    def _do_one_round(self, round_num, global_model, ctx: Context):
        aggr_result = _AggrResult()

        # Get total number of clients
        total_clients = len(ctx.clients)
        min_clients_required = self.min_clients if self.min_clients is not None else total_clients

        self.logger.info(
            f"[{ctx.header_str()}] Round {round_num}: Expecting updates from {total_clients} clients "
            f"(minimum required: {min_clients_required})"
        )

        # Prepare model for streaming if in system mode
        if ctx.env_type == EnvType.SYSTEM:
            model = prepare_for_download(
                state_dict=global_model,
                ctx=ctx,
                timeout=self.timeout,
                num_tensors_per_chunk=self.num_tensors_per_chunk,
            )
            model_type = "ref"
            self.logger.info(f"[{ctx.header_str()}] Prepared model as ref for round {round_num}")
        else:
            model = global_model
            model_type = "model"

        # Send to all clients for training and wait for ALL responses
        self.logger.info(f"[{ctx.header_str()}] Round {round_num}: Sending global model to all clients...")
        all_clients(
            ctx,
            blocking=True,  # Explicitly block until responses received
            timeout=3600.0,  # Set to 1 hour to allow training to complete
            min_resps=min_clients_required,  # Require minimum number of responses
            wait_after_min_resps=0.0,  # Don't wait extra time after minimum is reached
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        ).train(round_num, model, model_type)

        self.logger.info(
            f"[{ctx.header_str()}] Round {round_num}: Received updates from {aggr_result.count}/{total_clients} clients"
        )

        # Validate we received enough updates
        if aggr_result.count < min_clients_required:
            self.logger.error(
                f"Round {round_num}: Insufficient clients returned results. "
                f"Received {aggr_result.count}, required {min_clients_required}"
            )
            return None

        if aggr_result.count == 0:
            self.logger.warning(f"Round {round_num}: No clients returned results")
            return None
        else:
            # Normalize the aggregated weights and metrics
            result = {}
            for k, v in aggr_result.total.items():
                result[k] = torch.div(v, aggr_result.count)

            # Log metrics
            avg_metrics = {}
            for metric_name, metric_values in aggr_result.metrics.items():
                avg_metrics[metric_name] = sum(metric_values) / len(metric_values)

            self.logger.info(
                f"[{ctx.header_str()}] Round {round_num}: Successfully aggregated from {aggr_result.count} clients: "
                f"{aggr_result.client_names}. Metrics: {avg_metrics}"
            )

            # Save round model to server workspace
            workspace = ctx.workspace
            if workspace:
                output_dir = os.path.join(workspace.get_work_dir(), "round_models")
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, f"model_round_{round_num}.pt")
                torch.save(result, model_path)
                self.logger.info(f"Saved round {round_num} model to {model_path}")

            return result

    def _accept_train_result(self, result, aggr_result: _AggrResult, context: Context):
        client_name = context.caller
        self.logger.info(f"[{context.header_str()}] Received train result from {client_name}")

        model, model_type, metrics = result

        # Track which client responded
        aggr_result.client_names.append(client_name)

        # Store metrics
        for metric_name, metric_value in metrics.items():
            if metric_name not in aggr_result.metrics:
                aggr_result.metrics[metric_name] = []
            aggr_result.metrics[metric_name].append(metric_value)

        # In-time aggregate the updates
        if model_type == "ref":
            # Download streamed model
            err, model = download_state_dict(
                ref=model,
                per_request_timeout=self.timeout,
                ctx=context,
                tensors_received_cb=self._aggregate_tensors,
                aggr_result=aggr_result,
                context=context,
            )
            if err:
                raise RuntimeError(f"Failed to download model from {context.caller}: {err}")
        else:
            # Direct aggregation
            for k, v in model.items():
                if k not in aggr_result.total:
                    aggr_result.total[k] = v
                else:
                    aggr_result.total[k] += v

        aggr_result.count += 1
        self.logger.info(
            f"[{context.header_str()}] Successfully aggregated update from {client_name} ({aggr_result.count} total)"
        )
        return None

    def _aggregate_tensors(self, td: dict[str, torch.Tensor], aggr_result: _AggrResult, context: Context):
        self.logger.info(f"[{context.header_str()}] Aggregating received tensors")
        for k, v in td.items():
            if k not in aggr_result.total:
                aggr_result.total[k] = v
            else:
                aggr_result.total[k] += v


class HFTrainer(ClientApp):
    """HuggingFace Trainer Client App."""

    def __init__(
        self,
        model_name_or_path="facebook/opt-125m",
        data_path_train="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/training.jsonl",
        data_path_valid="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/validation.jsonl",
        num_tensors_per_chunk=1,
        output_path="dolly-sft",
        lr_scheduler="constant",
        local_epoch=1,
        num_rounds=3,
        batch_size=4,
        gradient_accumulation_steps=10,
        learning_rate=5e-4,
        max_length=1024,
        timeout=10.0,
        device="auto",  # Can be "auto", "cpu", "cuda", "cuda:0", etc.
    ):
        ClientApp.__init__(self)
        self.model_name_or_path = model_name_or_path
        self.data_path_train = data_path_train
        self.data_path_valid = data_path_valid
        self.num_tensors_per_chunk = num_tensors_per_chunk
        self.output_path = output_path
        self.lr_scheduler = lr_scheduler
        self.local_epoch = local_epoch
        self.num_rounds = num_rounds
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.timeout = timeout
        self.device = device

        self.trainer = None
        self._initialized = False
        self.total_epochs = num_rounds * local_epoch  # Total epochs across all rounds

    def _initialize(self, context: Context):
        """Initialize model, dataset, and trainer."""
        if self._initialized:
            return

        self.logger.info(f"[{context.header_str()}] Initializing HF Trainer")

        # Resolve output_path relative to workspace work directory
        workspace = context.workspace
        if workspace:
            work_dir = workspace.get_work_dir()
            # If output_path is relative, make it relative to work_dir
            if not os.path.isabs(self.output_path):
                resolved_output_path = os.path.join(work_dir, self.output_path)
            else:
                resolved_output_path = self.output_path
            self.logger.info(f"Resolved output path: {resolved_output_path}")
        else:
            resolved_output_path = self.output_path
            self.logger.warning("No workspace available, using output_path as-is")

        self.resolved_output_path = resolved_output_path

        # Load datasets
        self.logger.info(f"Loading datasets from {self.data_path_train} and {self.data_path_valid}")
        dataset_train = datasets.load_dataset("json", data_files=self.data_path_train, split="train")
        dataset_valid = datasets.load_dataset("json", data_files=self.data_path_valid, split="train")
        self.logger.info(f"Dataset size: training {len(dataset_train)}, validation {len(dataset_valid)}")

        # Calculate logging steps (every 5% of dataset)
        logging_steps = max(1, int(len(dataset_train) / (20 * self.batch_size * self.gradient_accumulation_steps)))

        # Load model
        self.logger.info(f"Loading model {self.model_name_or_path}")
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        # Determine target device
        # Priority: explicit device parameter > auto based on env_type
        if self.device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {target_device} (auto-detected)")
        else:
            # Use explicitly specified device
            target_device = self.device
            self.logger.info(f"Using explicitly specified device: {target_device}")

        self.logger.info(f"Loading model from {self.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=target_device,
            use_cache=False,
            dtype=torch.bfloat16,
        )
        torch.set_default_dtype(default_dtype)
        model.config.pretraining_tp = 1

        # Store target device for later use
        self.target_device = target_device

        # Training arguments
        # Set num_train_epochs to total_epochs (num_rounds * local_epoch)
        # Training will be interrupted at each round by StopCallback
        train_args = SFTConfig(
            output_dir=self.resolved_output_path,
            num_train_epochs=self.total_epochs,  # Total epochs across all federated rounds
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
            logging_steps=logging_steps,
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type=self.lr_scheduler,
            lr_scheduler_kwargs={"num_cycles": 2} if self.lr_scheduler == "cosine_with_restarts" else {},
            disable_tqdm=True,
            max_length=self.max_length,
            save_total_limit=2,
            save_safetensors=False,
            seed=0,
            data_seed=0,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            no_cuda=(target_device == "cpu"),
        )

        # Create trainer
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            formatting_func=format_instruction,
            args=train_args,
            callbacks=[StopCallback()],
        )

        self._initialized = True
        self.logger.info(f"[{context.header_str()}] HF Trainer initialized successfully")

    @collab
    def train(self, current_round, global_model, model_type: str, context: Context):
        """Train the model for one round."""
        if context.is_aborted():
            self.logger.debug("Training aborted")
            return None, None, {}

        self.logger.info(f"[{context.header_str()}] Starting training for round {current_round}")

        # Initialize on first call
        if not self._initialized:
            self._initialize(context)

        # Download global model if streaming
        if global_model is not None:
            if model_type == "ref":
                self.logger.info(f"[{context.header_str()}] Downloading global model")
                err, global_model = download_state_dict(ref=global_model, per_request_timeout=self.timeout, ctx=context)
                if err:
                    raise RuntimeError(f"Failed to download model: {err}")
                self.logger.info(f"[{context.header_str()}] Successfully downloaded global model")

            # Ensure all tensors in global_model are on the correct device
            if self.target_device != "cpu":
                global_model = {
                    k: v.to(self.target_device) if isinstance(v, torch.Tensor) else v for k, v in global_model.items()
                }
                self.logger.info(f"[{context.header_str()}] Moved global model tensors to {self.target_device}")

            # Load global model into trainer
            self.trainer.model.load_state_dict(global_model)

        # Evaluate global model
        self.logger.info(f"[{context.header_str()}] Round {current_round} - Starting evaluation...")
        eval_result = self.trainer.evaluate()
        eval_loss = float(eval_result.get("eval_loss", 0.0))

        # Calculate cumulative epochs
        cumulative_start_epoch = current_round * self.local_epoch
        cumulative_end_epoch = (current_round + 1) * self.local_epoch

        self.logger.info(
            f"[{context.header_str()}] Round {current_round} - Evaluation completed. "
            f"Loss: {eval_loss:.4f}. Will train cumulative epochs {cumulative_start_epoch} to {cumulative_end_epoch - s1} "
            f"(local epochs: {self.local_epoch}, total planned: {self.total_epochs})"
        )

        # Train for local epochs
        # Note: SFTConfig.num_train_epochs is set to total_epochs (num_rounds * local_epoch)
        # StopCallback stops training after each epoch, so we train local_epoch times per round
        # The trainer tracks cumulative epochs internally across rounds
        if current_round == 0:
            # First round - start from scratch
            for epoch in range(self.local_epoch):
                local_epoch_num = epoch + 1
                cumulative_epoch_num = cumulative_start_epoch + epoch + 1
                self.logger.info(
                    f"[{context.header_str()}] Starting training local epoch {local_epoch_num}/{self.local_epoch} "
                    f"(cumulative epoch {cumulative_epoch_num}/{self.total_epochs})"
                )
                if epoch == 0:
                    self.trainer.train()
                else:
                    self.trainer.train(resume_from_checkpoint=True)
                self.logger.info(
                    f"[{context.header_str()}] Completed training local epoch {local_epoch_num}/{self.local_epoch} "
                )
        else:
            # Subsequent rounds - replace checkpoint with global model
            resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(self.trainer.args.output_dir)
            if resume_from_checkpoint_folder:
                self.trainer.model.save_pretrained(
                    resume_from_checkpoint_folder, state_dict=global_model, safe_serialization=False
                )

            # Continue training
            for epoch in range(self.local_epoch):
                local_epoch_num = epoch + 1
                cumulative_epoch_num = cumulative_start_epoch + epoch + 1
                self.logger.info(
                    f"[{context.header_str()}] Starting training local epoch {local_epoch_num}/{self.local_epoch} "
                    f"(cumulative epoch {cumulative_epoch_num}/{self.total_epochs})"
                )
                self.trainer.train(resume_from_checkpoint=True)
                self.logger.info(
                    f"[{context.header_str()}] Completed training local epoch {local_epoch_num}/{self.local_epoch} "
                )

        # Get updated model parameters
        out_param = self.trainer.model.state_dict()
        self.logger.info(f"[{context.header_str()}] Prepared {len(out_param)} parameters for transmission")

        # Prepare for streaming if in system mode
        if context.env_type == EnvType.SYSTEM:
            model = prepare_for_download(
                state_dict=out_param,
                ctx=context,
                timeout=self.timeout,
                num_tensors_per_chunk=self.num_tensors_per_chunk,
            )
            result_model_type = "ref"
            self.logger.info(f"[{context.header_str()}] Prepared model as ref for streaming")
        else:
            model = out_param
            result_model_type = "model"

        # Prepare metrics
        metrics = {
            "eval_loss": eval_loss,
            "num_samples": self.trainer.train_dataset.num_rows,
        }

        return model, result_model_type, metrics


def main():
    """Main function to run the HF federated learning example."""
    simple_logging(logging.INFO)

    # Model configuration
    model_name_or_path = "facebook/opt-125m"

    # Model transmission configuration
    num_tensors_per_chunk = 1

    # Federated learning configuration
    num_rounds = 3  # Number of federated rounds
    local_epoch = 1  # Number of local epochs per round per client
    # Total training epochs per client = num_rounds * local_epoch = 3 * 1 = 3 epochs

    # Server app with HF FedAvg strategy
    # min_clients: minimum number of client updates required before aggregation
    #   - None (default): require ALL clients to respond
    #   - N: require at least N clients to respond (allows some to fail/timeout)
    server_app = ServerApp(
        strategy_name="hf_fed_avg_stream",
        strategy=HFFedAvgStream(
            model_name_or_path=model_name_or_path,
            num_rounds=num_rounds,
            timeout=10.0,
            num_tensors_per_chunk=num_tensors_per_chunk,
            min_clients=None,  # Require ALL clients (both site-1 and site-2)
        ),
    )

    # Client app with HF trainer
    # Device options for FOX Simulator (thread-based):
    #   "cpu" - TRUE parallel execution (recommended for testing with small models)
    #   "cuda:0" - GPU training with operation serialization (faster per-client, quasi-parallel)
    #
    # NOTE: FOX Simulator uses threads (not processes), so GPU operations are serialized
    # by CUDA. For TRUE parallel GPU training, use separate processes
    # or use more GPUs.
    client_app = HFTrainer(
        model_name_or_path=model_name_or_path,
        output_path="dolly-sft",  # Relative path - will be resolved to workspace/<site>/dolly-sft
        data_path_train="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/training.jsonl",
        data_path_valid="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/validation.jsonl",
        num_tensors_per_chunk=num_tensors_per_chunk,
        lr_scheduler="constant",
        local_epoch=local_epoch,
        num_rounds=num_rounds,
        batch_size=4,
        gradient_accumulation_steps=10,
        learning_rate=5e-4,
        max_length=1024,
        device="cuda:0",
    )

    # Simulator
    simulator = Simulator(
        root_dir="/tmp/nvflare/fox_hf_sim",
        experiment_name="hf_fedavg_stream",
        server_app=server_app,
        client_app=client_app,
        num_clients=1,
    )

    simulator.run()


if __name__ == "__main__":
    main()
