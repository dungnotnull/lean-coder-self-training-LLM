"""Distributed training with FSDP and DeepSpeed support."""

import os
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    strategy: str = "fsdp"  # "fsdp", "deepspeed", "ddp", or None
    num_gpus: int = 4
    cpu_offload: bool = True
    mixed_precision: str = "bf16"
    shard_strategy: str = "FULL_SHARD"
    deepspeed_stage: int = 2


class DistributedTrainer:
    """Distributed training manager."""

    def __init__(self, config: DistributedConfig):
        """Initialize distributed trainer.

        Args:
            config: Distributed configuration
        """
        self.config = config
        self.world_size = config.num_gpus
        self.rank = 0
        self.local_rank = 0
        self.device = None

        self._setup_distributed()

    def _setup_distributed(self):
        """Setup distributed environment."""
        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.config.num_gpus))

        if torch.cuda.is_available():
            # Try to initialize distributed
            if dist.is_available():
                # Use available GPUs
                self.world_size = min(self.config.num_gpus, torch.cuda.device_count())

                if self.world_size > 1:
                    self._initialize_process_group()
                else:
                    logger.warning(f"Only 1 GPU available, falling back to single-GPU training")
                    self.device = torch.device("cuda:0")
            else:
                logger.warning("Distributed not available, using single GPU")
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            logger.warning("CUDA not available, using CPU")
            self.device = torch.device("cpu")

    def _initialize_process_group(self):
        """Initialize distributed process group."""
        # For multi-GPU training, would need proper process spawning
        # This is a simplified version
        try:
            if self.config.strategy == "ddp":
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    world_size=self.world_size,
                )
            elif self.config.strategy == "fsdp":
                self._setup_fsdp()
            elif self.config.strategy == "deepspeed":
                self._setup_deepspeed()

            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {e}")
            logger.info("Falling back to single-GPU training")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _setup_fsdp(self):
        """Setup FSDP (Fully Sharded Data Parallel)."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            from torch.distributed.fsdp import CPUOffload

            sharding_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
            }

            self.fsdp_config = {
                "sharding_strategy": sharding_map.get(
                    self.config.shard_strategy, ShardingStrategy.FULL_SHARD
                ),
                "cpu_offload": CPUOffload(offload_params=self.config.cpu_offload),
            }

            logger.info("FSDP configured")

        except ImportError:
            logger.warning("FSDP not available, falling back to DDP")
            self.config.strategy = "ddp"

    def _setup_deepspeed(self):
        """Setup DeepSpeed."""
        try:
            import deepspeed

            self.deepspeed_config = {
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 4,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": 2e-4,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                    },
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": 2e-4,
                        "warmup_num_steps": 100,
                    },
                },
                "fp16": {
                    "enabled": self.config.mixed_precision == "fp16",
                },
                "bf16": {
                    "enabled": self.config.mixed_precision == "bf16",
                },
                "zero_optimization": {
                    "stage": self.config.deepspeed_stage,
                },
                "gradient_accumulation_steps": 4,
                "gradient_clipping": 1.0,
                "steps_per_print": 10,
            }

            logger.info("DeepSpeed configured")

        except ImportError:
            logger.warning("DeepSpeed not available, falling back to DDP")
            self.config.strategy = "ddp"

    def wrap_model(self, model):
        """Wrap model for distributed training.

        Args:
            model: Model to wrap

        Returns:
            Wrapped model
        """
        if self.config.strategy == "fsdp":
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                return FSDP(model, **self.fsdp_config)
            except Exception as e:
                logger.warning(f"Failed to wrap with FSDP: {e}")

        elif self.config.strategy == "deepspeed":
            try:
                import deepspeed

                model, _, _, _ = deepspeed.initialize(
                    model=model,
                    config=self.deepspeed_config,
                )
                return model
            except Exception as e:
                logger.warning(f"Failed to wrap with DeepSpeed: {e}")

        elif self.config.strategy == "ddp":
            try:
                from torch.nn.parallel import DistributedDataParallel as DDP

                return DDP(model, device_ids=[self.local_rank])
            except Exception as e:
                logger.warning(f"Failed to wrap with DDP: {e}")

        # Fallback: return unwrapped model
        logger.info("Using unwrapped model")
        return model

    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """Calculate effective batch size.

        Args:
            base_batch_size: Base batch size per GPU

        Returns:
            Effective batch size
        """
        return base_batch_size * self.world_size

    def save_checkpoint(self, model, path: str, rank0_only: bool = True):
        """Save distributed checkpoint.

        Args:
            model: Model to save
            path: Save path
            rank0_only: Only save on rank 0
        """
        if rank0_only and self.rank != 0:
            return

        # FSDP requires special handling
        if self.config.strategy == "fsdp":
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType

                if isinstance(model, FSDP):
                    with FSDP.state_dict_type(
                        model, StateDictType.FULL_STATE_DICT
                    ):
                        state_dict = model.state_dict()
                else:
                    state_dict = model.state_dict()
            except Exception:
                state_dict = model.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(state_dict, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, model, path: str):
        """Load distributed checkpoint.

        Args:
            model: Model to load into
            path: Checkpoint path
        """
        state_dict = torch.load(path, map_location=self.device)

        if self.config.strategy == "fsdp":
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                if isinstance(model, FSDP):
                    # FSDP load
                    model.load_state_dict(state_dict)
                else:
                    model.load_state_dict(state_dict)
            except Exception:
                model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        logger.info(f"Checkpoint loaded: {path}")

    def cleanup(self):
        """Cleanup distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.cleanup()


def setup_distributed(config: Dict) -> DistributedTrainer:
    """Setup distributed training from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        DistributedTrainer instance
    """
    dist_config = DistributedConfig(
        strategy=config.get("distributed", {}).get("strategy", "fsdp"),
        num_gpus=config.get("distributed", {}).get("num_gpus", 4),
        cpu_offload=config.get("distributed", {}).get("cpu_offload", True),
        mixed_precision=config.get("distributed", {}).get("mixed_precision", "bf16"),
        shard_strategy=config.get("fsdp", {}).get("sharding_strategy", "FULL_SHARD"),
        deepspeed_stage=config.get("deepspeed", {}).get("stage", 2),
    )

    return DistributedTrainer(dist_config)
