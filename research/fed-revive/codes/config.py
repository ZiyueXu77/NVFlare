import json
import os
from typing import Any, Dict, Optional, Union

from utils.data_utils import get_number_of_classes


class Config:
    """
    Configuration class for FL experiments.
    Handles loading, processing, and accessing configuration parameters.
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        # Experiment settings
        "setup_random_seed": 42,
        "run_random_seed": 64,
        "log_dir": "logs",
        "train_device": "cuda",
        "model_device": "cpu",
        "data_device": "cpu",
        "gpu_id": 0,
        # Exp state save settings
        "continue_exp_from": None,  # If not a valid exp path, it will be ignored.
        "save_exp_state_interval": 10,  # in minutes. If <= 0, it will be ignored.
        # Server settings
        # If any of the following conditions are met, the server will stop:
        # If any of them is set to <= 0, it will be ignored.
        "target_accuracy": 0.9,
        "max_wait_time": 6000,  # in seconds
        "max_rounds": 2000,
        # Client settings
        "num_clients": 10,
        # Algorithm settings
        "algorithm": "async_fl",
        "async_fl_settings": {"nb_of_active_jobs": 10, "buffer_size": 2},  # For asynchronous FL
        "sync_fl_settings": {"nb_of_active_jobs": 10},
        "async_downweighting": False,  # Whether to downweight based on staleness kd_augmentation should be None if this is True
        "kd_augmentation": None,  # None or "simple_kd" (for simple knowledge distillation) or "dfkd" (for DFKD)
        "simple_kd_settings": {
            "learning_rate": 0.01,
            "nb_of_iters": 10,
            "kd_batch_size": 32,
            "synth_batch_size": 64,
            "beta": 0.5,  # Can be a float value or "adaptive"/"cosine"/"exponential"/"linear"/"stepwise" for adaptive beta
            "adaptive_beta_factor": 75,  # Parameter of the adaptive beta decay function see utils.dfkd_utils.py for more details
            "temperature": 1,
            "kd_buffer_size": 3,  # Size of the KD buffer for storing recent updates
            "kd_version": "avg_logits",  # "avg_logits", "avg_loss", or "distribution_match_batch"
            "weighted_avg": False,  # Whether to use weighted averaging based on client data distributions
            "weighted_class_loss": False,  # Whether to use weighted class loss in KL divergence (only for avg_loss)
        },
        "dfkd_settings": {
            "beta": 0.5,  # Can be a float value or "adaptive"-"cosine"/"exponential"/"linear"/"stepwise" for adaptive beta
            "adaptive_beta_factor": 75,  # Parameter of the adaptive beta decay function see utils.dfkd_utils.py for more details
            "vis_freq": 10,  # Frequency of saving visualization grids (every N synthesis runs)
            "max_images": None,  # Maximum number of images/text samples to keep in memory (None for unlimited)
            "batch_size": 64,  # Synthesis batch size
            "lr": 0.0003,  # Learning rate for student
            "T": 20,  # Temperature for knowledge distillation
            "kd_buffer_size": 5,  # Size of model buffer for teacher selection
            "lr_g": 3e-3,  # Learning rate for generator
            "lr_z": 1e-2,  # Learning rate for latent code
            "g_steps": 10,  # Number of generator steps
            "warmup_rounds": 20,  # Warmup rounds before adversarial loss
            "freeze_rounds": 100,  # Rounds to freeze certain components
            "adv": 0.1,  # Scaling factor for adversarial distillation
            "bn": 1e-2,  # Scaling factor for BN regularization
            "oh": 1.0,  # Scaling factor for one hot loss
            "kl_uniform": 0.01,  # Scaling factor for KL-from-Uniform loss (confidence suppression)
            "diversity_loss_weight": 0.0,  # Scaling factor for diversity loss
            "class_embedding_weight": 1.0,  # Weight for class embeddings in additive conditioning
            "bn_mmt": 0.0,  # Momentum when fitting batchnorm statistics
            "is_maml": False,  # Use MAML (True) or REPTILE (False)
            "nz": 256,  # Noise vector dimension
            "ngf": 64,  # Generator feature dimension
            "reset_l0": True,  # Reset l0 in generator during training
            "kd_nb_of_iters": 25,  # Number of total steps in each KD run
        },
        "news_dataset_settings": {
            "embedding_dim": 512,
            "original_max_length": 128,
            "kd_max_length": 128,
            "kd_prompt_vec_len": 4,
            "kd_generator_min_active": 64,
            "kd_generator_max_active": 128,
        },
        "server_lr": 0.1,  # If negative, interpolate the models model <- (1-server_lr) * model_new + server_lr * model_new
        # If positive, update the model with server lr, model <- model + server_lr * (model_new - model_start)
        # Good values to try: -0.5, -0.3, -0.1, -0.05, 0.03, 0.1, 0.3, 1
        "pretrained_model_path": None,  # Path to pretrained model (.pth file) or None to use default
        # Data settings
        "dataset": "cifar10",  # "cifar10", "cifar100", "femnist", "news"
        "val_split": 0.15,
        "test_split": 0.15,
        "kd_split": 0.05,
        "external_kd_dataset": None,  # Path to external KD dataset (.pth file) or None to use default
        "data_distribution": "non_iid_dirichlet",  # iid, non_iid_dirichlet
        "dirichlet_alpha": 0.3,
        "client_data_size": 1000,
        # Model settings
        "local_batch_size": 32,
        "local_iters": 20,
        "local_optimizer": "adam",  # "adam"
        "local_learning_rate": 0.01,
        "local_momentum": 0.9,
        # Delay settings - to generate random delays while creating computational heterogeneity
        "localtrain_delay": {
            "type": "exponential",
            "mean_distribution": [(0.25, 1), (0.5, 1.3), (0.25, 1.6)],
            "std": 0.2,
        },
        "download_delay": {"type": "constant", "mean_distribution": [(1, 0.1)], "std": 0.2},
        "upload_delay": {"type": "uniform", "mean_distribution": [(0.5, 0.15), (0.5, 0.25)], "std": 0.02},
        # Evaluation settings
        "eval_interval": 10,
        "save_model_interval": 1000,
        "eval_batch_size": 300,
        "save_plot": True,  # Whether to save test evaluation plots
        # TensorBoard settings (wandb_flag kept for backward compatibility)
        "wandb_flag": False,  # Set to True to enable TensorBoard logging
        "wandb_entity": None,  # Deprecated: Not used with TensorBoard
        "wandb_project": "Default-Project",  # Deprecated: Not used with TensorBoard
        "wandb_run_name": "Default-Run",  # Deprecated: Not used with TensorBoard
        "wandb_group": None,  # Deprecated: Not used with TensorBoard
        # Output settings
        "out_dir": "./experiments/default_experiment",
    }

    def __init__(self, config: Union[str, Dict[str, Any], "Config", None] = None):
        """
        Initialize configuration.

        Args:
            config: Can be one of the following:
                - Path to a JSON file (str)
                - Dictionary containing configuration (Dict)
                - None (use default configuration)
                - Another Config object
        """
        # Start with default configuration
        self.config = self.DEFAULT_CONFIG.copy()

        # Load configuration from file or dictionary
        if config is not None:
            if isinstance(config, str):
                # Load from JSON file
                try:
                    self.load_from_file(config)
                except FileNotFoundError:
                    print(f"File {config} not found. Using default configuration")
            elif isinstance(config, dict):
                # Update from dictionary
                self.update_config(config)
            elif isinstance(config, self.__class__):
                # Update from another Config object
                self.update_config(config.config)
            else:
                raise ValueError("Config must be a valid file path, a dictionary, or None")

        self.set_nb_of_classes()

        self.set_lr_values()

    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file
        """
        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)

            self.update_config(config_data)
            print(f"Configuration loaded from {file_path}")
        except Exception as e:
            print(f"Error loading configuration from {file_path}: {e}")
            print("Using default configuration")

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters to update
        """
        for key, value in config_dict.items():
            self.config[key] = value

    def save_to_file(self, file_path: str) -> None:
        """
        Save current configuration to a JSON file.

        Args:
            file_path: Path to save the configuration file
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {file_path}")
        except Exception as e:
            print(f"Error saving configuration to {file_path}: {e}")

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-like access to configuration parameters.

        Args:
            key: Configuration parameter name

        Returns:
            Value of the configuration parameter
        """
        return self.config.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-like setting of configuration parameters.

        Args:
            key: Configuration parameter name
            value: Value to set
        """
        self.config[key] = value

    def __iter__(self):
        """
        Allow iteration over configuration parameter keys.

        Returns:
            Iterator over configuration keys
        """
        return iter(self.config)

    def keys(self):
        """
        Get configuration parameter keys.

        Returns:
            View of configuration keys
        """
        return self.config.keys()

    def values(self):
        """
        Get configuration parameter values.

        Returns:
            View of configuration values
        """
        return self.config.values()

    def items(self):
        """
        Get configuration parameter key-value pairs.

        Returns:
            View of configuration key-value pairs
        """
        return self.config.items()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter with a default value if not found.

        Args:
            key: Configuration parameter name
            default: Default value to return if parameter is not found

        Returns:
            Value of the configuration parameter or default
        """
        return self.config.get(key, default)

    def get_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.

        Returns:
            Dictionary containing all configuration parameters
        """
        return self.config.copy()

    def set_nb_of_classes(self) -> int:
        """
        Get the number of classes in the dataset.
        """
        self.config["nb_of_classes"] = get_number_of_classes(self.config["dataset"])

    def set_lr_values(self) -> None:
        """
        Set the learning rate values based on the dataset.
        """
        if self.config["local_learning_rate"] == "auto":

            if self.config["dataset"] == "cifar10":
                if self.config["algorithm"] == "sync_fl":
                    self.config["local_learning_rate"] = 0.0003
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["local_learning_rate"] = 0.0003
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["local_learning_rate"] = 0.0003
                    elif self.config["kd_augmentation"] is None:
                        self.config["local_learning_rate"] = 0.0003

            elif self.config["dataset"] == "cifar100":
                if self.config["algorithm"] == "sync_fl":
                    self.config["local_learning_rate"] = 0.0003
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["local_learning_rate"] = 0.0003
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["local_learning_rate"] = 0.0001
                    elif self.config["kd_augmentation"] is None:
                        self.config["local_learning_rate"] = 0.0003

            elif self.config["dataset"] == "femnist":
                if self.config["algorithm"] == "sync_fl":
                    self.config["local_learning_rate"] = 0.001
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["local_learning_rate"] = 0.001
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["local_learning_rate"] = 0.001
                    elif self.config["kd_augmentation"] is None:
                        self.config["local_learning_rate"] = 0.001

            elif self.config["dataset"] == "news":
                if self.config["algorithm"] == "sync_fl":
                    self.config["local_learning_rate"] = 0.003
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["local_learning_rate"] = 0.001
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["local_learning_rate"] = 0.001
                    elif self.config["kd_augmentation"] is None:
                        self.config["local_learning_rate"] = 0.001

        if self.config["server_lr"] == "auto":
            if self.config["dataset"] == "cifar10":
                if self.config["algorithm"] == "sync_fl":
                    self.config["server_lr"] = 1.4
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["server_lr"] = 0.1
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["server_lr"] = 0.2
                    elif self.config["kd_augmentation"] is None:
                        self.config["server_lr"] = 0.05

            elif self.config["dataset"] == "cifar100":
                if self.config["algorithm"] == "sync_fl":
                    self.config["server_lr"] = 1.4
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["server_lr"] = 0.1
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["server_lr"] = 0.3
                    elif self.config["kd_augmentation"] is None:
                        self.config["server_lr"] = 0.1

            elif self.config["dataset"] == "femnist":
                if self.config["algorithm"] == "sync_fl":
                    self.config["server_lr"] = 1.4
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["server_lr"] = 0.05
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["server_lr"] = 0.1
                    elif self.config["kd_augmentation"] is None:
                        self.config["server_lr"] = 0.025

            elif self.config["dataset"] == "news":
                if self.config["algorithm"] == "sync_fl":
                    self.config["server_lr"] = 1
                elif self.config["algorithm"] == "async_fl":
                    if self.config["kd_augmentation"] == "dfkd":
                        self.config["server_lr"] = 0.1
                    elif self.config["kd_augmentation"] == "simple_kd":
                        self.config["server_lr"] = 0.1
                    elif self.config["kd_augmentation"] is None:
                        self.config["server_lr"] = 0.1

        assert isinstance(self.config["local_learning_rate"], (int, float)) and isinstance(
            self.config["server_lr"], (int, float)
        ), "local_learning_rate or server_lr couldn't be set to a numeric value"
