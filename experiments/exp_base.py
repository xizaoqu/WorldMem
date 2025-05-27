"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Dict
import pathlib
import os

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info

from omegaconf import DictConfig

from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero
from safetensors.torch import load_model
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub import model_info

torch.set_float32_matmul_precision("high")

def is_huggingface_model(path: str) -> bool:
    hf_ckpt = str(path).split('/')
    repo_id = '/'.join(hf_ckpt[:2])
    try:
        model_info(repo_id)
        return True
    except:
        return False
    
def load_custom_checkpoint(algo, checkpoint_path):
    if not checkpoint_path:
        rank_zero_info("No checkpoint path provided, skipping checkpoint loading.")
        return None

    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)

    if is_huggingface_model(str(checkpoint_path)):
        # Load from Hugging Face Hub if the path contains 'yslan'
        hf_ckpt = str(checkpoint_path).split('/')
        repo_id = '/'.join(hf_ckpt[:2])
        file_name = '/'.join(hf_ckpt[2:])
        model_path = hf_hub_download(repo_id=repo_id, filename=file_name)
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))

        filtered_state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if "frame_timestep_embedder" in k:
                new_k = k.replace("frame_timestep_embedder", "timestamp_embedding")
                filtered_state_dict[new_k] = v
            else:
                filtered_state_dict[k] = v

        algo.load_state_dict(filtered_state_dict, strict=True)

    elif checkpoint_path.suffix == ".pt":
        # Load from a .pt file
        ckpt = torch.load(checkpoint_path, weights_only=True)

        filtered_state_dict = {
            k: v for k, v in ckpt.items()
            if not k in ["data_mean", "data_std"]
        }

        algo.load_state_dict(filtered_state_dict, strict=False)

    elif checkpoint_path.suffix == ".ckpt":
        # Load from a .ckpt file
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        filtered_state_dict = {
            k: v for k, v in ckpt['state_dict'].items()
            if not k in ["data_mean", "data_std"]
        }
        algo.load_state_dict(filtered_state_dict, strict=False)

    elif checkpoint_path.suffix == ".safetensors":
        load_model(algo, checkpoint_path, strict=False)
        
    elif os.path.isdir(checkpoint_path):
        # Load the most recent .ckpt file from directory
        ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
        if not ckpt_files:
            raise FileNotFoundError("No .ckpt files found in the specified directory!")
        selected_ckpt = max(ckpt_files)
        selected_ckpt_path = os.path.join(checkpoint_path, selected_ckpt)
        print(f"Checkpoint file selected for loading: {selected_ckpt_path}")
        
        ckpt = torch.load(selected_ckpt_path, map_location=torch.device('cpu'))

        filtered_state_dict = {
            k: v for k, v in ckpt['state_dict'].items()
            if not k in ["data_mean", "data_std"]
        }

        # for k, v in filtered_state_dict.items():
        #     if "frame_timestep_embedder" in k:
        #         new_k = k.replace("frame_timestep_embedder", "timestamp_embedding")
        #         filtered_state_dict[new_k] = v

        algo.load_state_dict(filtered_state_dict, strict=False)

    else:
        raise ValueError(
            f"Unsupported checkpoint: {checkpoint_path}"
        )
    rank_zero_info("Model weights loaded.")

class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None
        self.customized_load = getattr(root_cfg, "customized_load", False)
        self.seperate_load = getattr(root_cfg, "seperate_load", False)
        self.zero_init_gate= getattr(root_cfg, "zero_init_gate", False)
        self.only_tune_memory = getattr(root_cfg, "only_tune_memory", False)
        self.diffusion_model_path = getattr(root_cfg, "diffusion_model_path", None)
        self.vae_path = getattr(root_cfg, "vae_path", None)
        self.pose_predictor_path = getattr(root_cfg, "pose_predictor_path", None)

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """
        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )

    def exec_interactive(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """
        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            return getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )

class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_validation_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        if validation_dataset:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.validation.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        shuffle = False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.test.data.shuffle
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.test.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    filename="epoch{epoch}_step{step}",
                    auto_insert_metric_name=False,
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",  # 自动选择设备
            strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
            logger=self.logger or False,  # 简化写法
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val or 0.0,  # 确保默认值
            val_check_interval=self.cfg.validation.val_every_n_step if self.cfg.validation.val_every_n_step else None,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch if not self.cfg.validation.val_every_n_step else None,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches or 1,  # 默认累积为1
            precision=self.cfg.training.precision or 32,  # 默认32位精度
            detect_anomaly=False,  # 默认关闭异常检测
            num_sanity_val_steps=int(self.cfg.debug) if self.cfg.debug else 0,
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time
        )


        if self.customized_load:
            if self.seperate_load:
                load_custom_checkpoint(algo=self.algo.diffusion_model,checkpoint_path=self.diffusion_model_path)
                load_custom_checkpoint(algo=self.algo.vae,checkpoint_path=self.vae_path)
            else:
                load_custom_checkpoint(algo=self.algo,checkpoint_path=self.ckpt_path)

            if self.zero_init_gate:
                for name, para in self.algo.diffusion_model.named_parameters():
                    if 'r_adaLN_modulation' in name:
                        para.requires_grad_(False)
                        para[2*1024:3*1024] = 0
                        para[5*1024:6*1024] = 0
                        para.requires_grad_(True)

            if self.only_tune_memory:
                for name, para in self.algo.diffusion_model.named_parameters():
                    para.requires_grad_(False)
                    if 'r_' in name or 'pose_embedder' in name or 'pose_cond_mlp' in name or 'lora_' in name:
                        para.requires_grad_(True)
                    
            trainer.fit(
                self.algo,
                train_dataloaders=self._build_training_loader(),
                val_dataloaders=self._build_validation_loader(),
                ckpt_path=None,
            )
        else:

            if self.only_tune_memory:
                for name, para in self.algo.diffusion_model.named_parameters():
                    para.requires_grad_(False)
                    if 'r_' in name or 'pose_embedder' in name or 'pose_cond_mlp' in name or 'lora_' in name:
                        para.requires_grad_(True)
            
            trainer.fit(
                self.algo,
                train_dataloaders=self._build_training_loader(),
                val_dataloaders=self._build_validation_loader(),
                ckpt_path=self.ckpt_path,
            )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        if self.customized_load:
            if self.seperate_load:
                load_custom_checkpoint(algo=self.algo.diffusion_model,checkpoint_path=self.diffusion_model_path)
                load_custom_checkpoint(algo=self.algo.vae,checkpoint_path=self.vae_path)
            else:
                load_custom_checkpoint(algo=self.algo,checkpoint_path=self.ckpt_path)

            if self.zero_init_gate:
                for name, para in self.algo.diffusion_model.named_parameters():
                    if 'r_adaLN_modulation' in name:
                        para.requires_grad_(False)
                        para[2*1024:3*1024] = 0
                        para[5*1024:6*1024] = 0
                        para.requires_grad_(True)
            
            trainer.validate(
                self.algo,
                dataloaders=self._build_validation_loader(),
                ckpt_path=None,
            )
        else:
            trainer.validate(
                self.algo,
                dataloaders=self._build_validation_loader(),
                ckpt_path=self.ckpt_path,
            )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
        )
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if split in ["training", "test", "validation"]:
            return self.compatible_datasets[self.root_cfg.dataset._name](self.root_cfg.dataset, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
