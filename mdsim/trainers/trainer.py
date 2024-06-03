"""
*
*     SOFTWARE NAME
*
*        File:  trainer.py
*
*     Authors: Deleted for purposes of anonymity
*
*     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
*
* The software and its source code contain valuable trade secrets and shall be maintained in
* confidence and treated as confidential information. The software may only be used for
* evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
* license agreement or nondisclosure agreement with the proprietor of the software.
* Any unauthorized publication, transfer to third parties, or duplication of the object or
* source code---either totally or in part---is strictly prohibited.
*
*     Copyright (c) 2024 Proprietor: Deleted for purposes of anonymity
*     All Rights Reserved.
*
* THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
* AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
* DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.
*
* NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
* IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
* LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
* FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
* OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
* ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
* TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGES.
*
* For purposes of anonymity, the identity of the proprietor is not given herewith.
* The identity of the proprietor will be given once the review of the
* conference submission is completed.
*
* THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
*
"""

import datetime
import errno
import logging
import os
import random
import subprocess
from abc import ABC
from collections import defaultdict
import time
import _pickle as cPickle

import numpy as np
import math as mt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import torch_geometric
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import mdsim
from mdsim.common import distutils
from mdsim.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from mdsim.common.registry import registry
from mdsim.common.utils import save_checkpoint

from mdsim.modules.evaluator import Evaluator
from mdsim.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from mdsim.modules.loss import DDPLoss, L2MAELoss
from mdsim.modules.normalizer import Normalizer
from mdsim.modules.scheduler import LRScheduler

from ocpmodels.models.escn.escn import eSCN

rng = np.random.Generator(np.random.PCG64())


@registry.register_trainer("trainer")
class Trainer(ABC):
    """
    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """
    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        name="s2ef",
        slurm={},
        noddp=False,
        no_energy=False,
        simulate=False,
        if_robust_loss=None,
        delta=None,
        if_adv=None,
        coef_rob=None,
        coef_cyc=None,
    ):
        self.no_energy = no_energy
        self.simulate = simulate
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0
        self.elapsed = 0  # wall time

        # for optuna
        self.delta = delta
        self.if_adv = if_adv
        self.coef_rob = coef_rob
        self.coef_cyc = coef_cyc

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        if run_dir is None:
            run_dir = os.getcwd()
        
        if timestamp_id is None:
            timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
                self.device
            )
            # create directories from master rank only
            distutils.broadcast(timestamp, 0)
            timestamp = datetime.datetime.fromtimestamp(
                timestamp.int()
            ).strftime("%Y-%m-%d-%H-%M-%S")
            if identifier:
                self.timestamp_id = f"{timestamp}-{identifier}"
            else:
                self.timestamp_id = timestamp
        else:
            self.timestamp_id = timestamp_id
        
        # compose expname.
        data_name = dataset[0]["name"]
        model_name = model["name"]
        data_size = dataset[0]["size"]
        expname = f"{data_name}_{data_size}_{model_name}"
        if identifier:
            expname = f"{expname}_{identifier}"
        self.expname = expname

        try:
            commit_hash = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        mdsim.__path__[0],
                        "describe",
                        "--always",
                    ]
                )
                .strip()
                .decode("ascii")
            )
        # catch instances where code is not being run from a git repo
        except Exception:
            commit_hash = None

        logger_name = logger if isinstance(logger, str) else logger["name"]
        self.config = {
            "task": task,
            "model": model.pop("name"),
            "model_attributes": model,
            "optim": optimizer,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            
            "cmd": {
                "expname": self.expname,
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, self.expname, "checkpoints"),
                "results_dir": os.path.join(
                    run_dir, self.expname, "results"),
                "logs_dir": os.path.join(
                    run_dir, self.expname, "logs", logger_name),
            },
            "slurm": slurm,
            "noddp": noddp,
        }

        if if_robust_loss is None:
            self.if_robust_loss = self.config["optim"].get("robust_loss", False)
        else:
            self.if_robust_loss = if_robust_loss

        #if self.config["optim"].get("if_neuralODE", False):
        #    self.rhs_ODE = rhs_func().to(self.device)

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"][
                "folder"
            ].replace("%j", self.config["slurm"]["job_id"])
            
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            self.config["dataset"] = dataset.get("train", None)
            self.config["val_dataset"] = dataset.get("val", None)
            self.config["test_dataset"] = dataset.get("test", None)
        else:
            self.config["dataset"] = dataset

        self.normalizer = normalizer
        # This supports the legacy way of providing norm parameters in dataset
        if self.config.get("dataset", None) is not None and normalizer is None:
            self.normalizer = self.config["dataset"]

        if not is_debug and distutils.is_master() and not is_hpo and not simulate:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        self.is_debug = is_debug
        self.is_hpo = is_hpo

        if self.is_hpo:
            # conditional import is necessary for checkpointing
            from ray import tune
            from mdsim.common.hpo_utils import tune_reporter
            # sets the hpo checkpoint frequency
            # default is no checkpointing
            self.hpo_checkpoint_every = self.config["optim"].get(
                "checkpoint_every", -1
            )

        if distutils.is_master():
            print(yaml.dump(self.config, default_flow_style=False))

        self.load()
        self.evaluator = Evaluator(task=name, no_energy=no_energy)

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_task()
        self.load_model()
        self.load_loss()
        self.load_optimizer()
        self.load_extras()
            
    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["cmd"]["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if (not self.is_debug 
            and distutils.is_master() 
            and not self.is_hpo 
            and not self.simulate):
            assert (
                self.config["logger"] is not None
            ), "Specify logger in config"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

            self.logger = registry.get_logger_class(logger_name)(self.config)

    def get_sampler(self, dataset, batch_size, shuffle):
        if "load_balancing" in self.config["optim"]:
            balancing_mode = self.config["optim"]["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=distutils.get_world_size(),
            rank=distutils.get_rank(),
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )
        return sampler

    def get_dataloader(self, dataset, sampler):
        loader = DataLoader(
            dataset,
            collate_fn=self.parallel_collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )
        return loader

    def load_datasets(self):
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model_attributes"].get("otf_graph", False),
        )

        self.train_loader = self.val_loader = self.test_loader = None

        if self.config.get("dataset", None):
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"]["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )
            if self.train_loader is not None:
                print('train_loader exists...')

            if self.config.get("val_dataset", None):
                self.val_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["val_dataset"])
                self.val_sampler = self.get_sampler(
                    self.val_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=True,
                )
                self.val_loader = self.get_dataloader(
                    self.val_dataset,
                    self.val_sampler,
                )
                if self.val_loader is not None:
                    print('val_loader exists...')

            if self.config.get("test_dataset", None):
                self.test_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["test_dataset"])
                self.test_sampler = self.get_sampler(
                    self.test_dataset,
                    self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.test_loader = self.get_dataloader(
                    self.test_dataset,
                    self.test_sampler,
                )
                if self.test_loader is not None:
                    print('test_loader exists...')

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", False):
            if "target_mean" in self.normalizer and (not self.no_energy):                
                self.normalizers["target"] = Normalizer(
                    mean=self.normalizer["target_mean"],
                    std=self.normalizer["target_std"],
                    device=self.device,
                )
            else:
                if not self.no_energy:
                    self.normalizers["target"] = Normalizer(
                        tensor=self.train_loader.dataset.data.y[
                            self.train_loader.dataset.__indices__
                        ],
                        device=self.device,
                    )
                else:
                    if not self.no_energy:
                        raise NotImplementedError('normalization needs to be specified.')
                    else:
                        logging.info('<no_energy> flag is <True>. no energy information is used.')

    def load_task(self):
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")

        if "relax_dataset" in self.config["task"]:
            self.relax_dataset = registry.get_dataset_class("lmdb")(
                self.config["task"]["relax_dataset"]
            )
            self.relax_sampler = self.get_sampler(
                self.relax_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.relax_loader = self.get_dataloader(
                self.relax_dataset,
                self.relax_sampler,
            )

        self.num_targets = 1

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if self.config["model_attributes"].get("regress_forces", True):
            if self.normalizer.get("normalize_labels", False):
                if "grad_target_mean" in self.normalizer:
                    self.normalizers["grad_target"] = Normalizer(
                        mean=self.normalizer["grad_target_mean"],
                        std=self.normalizer["grad_target_std"],
                        device=self.device,
                    )
                else:
                    if not self.no_energy:
                        self.normalizers["grad_target"] = Normalizer(
                            tensor=self.train_loader.dataset.data.y[
                                self.train_loader.dataset.__indices__
                            ],
                            device=self.device,
                        )
                        self.normalizers["grad_target"].mean.fill_(0)
                    else:
                        raise NotImplementedError('normalization needs to be specified.')

    def load_model(self):
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )
        loader = self.train_loader or self.val_loader or self.test_loader
        if self.config["model"] != 'escn':
            self.model = registry.get_model_class(self.config["model"])(
                loader.dataset[0].x.shape[-1]
                if loader
                and hasattr(loader.dataset[0], "x")
                and loader.dataset[0].x is not None
                else None,
                bond_feat_dim,
                self.num_targets,
                **self.config["model_attributes"],
            ).to(self.device)
        else:
            self.model = eSCN(
                loader.dataset[0].x.shape[-1]
                if loader
                and hasattr(loader.dataset[0], "x")
                and loader.dataset[0].x is not None
                else None,
                bond_feat_dim,
                self.num_targets,
                **self.config["model_attributes"],
            ).to(self.device)

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

        if self.config["optim"].get('if_SOL', False):
            import sol
            self.model = sol.optimize(self.model)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.elapsed = checkpoint.get("elapsed", 0)
        if self.elapsed > 0:
            logging.info(f"trained time: {self.elapsed}")
        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        first_key = next(iter(checkpoint["state_dict"]))
        if (
            not distutils.initialized() or self.config["noddp"]
        ) and first_key.split(".")[1] == "module":
            # No need for OrderedDict since dictionaries are technically ordered
            # since Python 3.6 and officially ordered since Python 3.7
            new_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(new_dict)
        elif distutils.initialized() and first_key.split(".")[1] != "module":
            new_dict = {
                f"module.{k}": v for k, v in checkpoint["state_dict"].items()
            }
            self.model.load_state_dict(new_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])

    def load_loss(self):
        self.loss_fn = {}
        self.loss_fn["energy"] = self.config["optim"].get("loss_energy", "mse")
        self.loss_fn["force"] = self.config["optim"].get("loss_force", "mse")
        for loss, loss_name in self.loss_fn.items():
            if loss_name in ["l1", "mae"]:
                self.loss_fn[loss] = nn.L1Loss()
            elif loss_name == "mse":
                self.loss_fn[loss] = nn.MSELoss()
            elif loss_name == "huber":
                self.loss_fn[loss] = nn.HuberLoss()
            elif loss_name == "l2mae":
                self.loss_fn[loss] = L2MAELoss()
            elif loss_name == "energy_per_atom_rmse":
                #self.loss_fn[loss] = RMSE_per_atom_Loss()
                self.loss_fn[loss] = L2MAELoss()
            else:
                raise NotImplementedError(
                    f"Unknown loss function name: {loss_name}"
                )
            if distutils.initialized():
                self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)

        if self.config["optim"].get("weight_decay", 0) > 0:

            # Do not regularize bias etc.
            params_decay = []
            params_no_decay = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "embedding" in name:
                        params_no_decay += [param]
                    elif "frequencies" in name:
                        params_no_decay += [param]
                    elif "bias" in name:
                        params_no_decay += [param]
                    else:
                        params_decay += [param]

            self.optimizer = optimizer(
                [
                    {"params": params_no_decay, "weight_decay": 0},
                    {
                        "params": params_decay,
                        "weight_decay": self.config["optim"]["weight_decay"],
                    },
                ],
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )

    def load_extras(self):
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        self.early_stopping_lr = self.config["optim"].get("early_stopping_lr")
        self.early_stopping_time = self.config["optim"].get("early_stopping_time")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    def save(
        self,
        metrics=None,
        checkpoint_file="checkpoint.pt",
        training_state=True,
    ):
        if not self.is_debug and distutils.is_master():
            if training_state:
                save_checkpoint(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.scheduler.state_dict()
                        if self.scheduler.scheduler_type != "Null"
                        else None,
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "ema": self.ema.state_dict() if self.ema else None,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema:
                    self.ema.store()
                    self.ema.copy_to()
                save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()

    def save_hpo(self, epoch, step, metrics, checkpoint_every):
        # default is no checkpointing
        # checkpointing frequency can be adjusted by setting checkpoint_every in steps
        # to checkpoint every time results are communicated to Ray Tune set checkpoint_every=1
        if checkpoint_every != -1 and step % checkpoint_every == 0:
            with tune.checkpoint_dir(  # noqa: F821
                step=step
            ) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(self.save_state(epoch, step, metrics), path)

    def hpo_update(
        self, epoch, step, train_metrics, val_metrics, test_metrics=None
    ):
        progress = {
            "steps": step,
            "epochs": epoch,
            "act_lr": self.optimizer.param_groups[0]["lr"],
        }
        # checkpointing must occur before reporter
        # default is no checkpointing
        self.save_hpo(
            epoch,
            step,
            val_metrics,
            self.hpo_checkpoint_every,
        )
        # report metrics to tune
        tune_reporter(  # noqa: F821
            iters=progress,
            train_metrics={
                k: train_metrics[k]["metric"] for k in self.metrics
            },
            val_metrics={k: val_metrics[k]["metric"] for k in val_metrics},
            test_metrics=test_metrics,
        )

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image=True,
        results_file=None,
        disable_tqdm=False,
    ):
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None:
            if not self.no_energy:
                self.normalizers["target"].to(self.device)
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": []}

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)

            if self.normalizers is not None:
                if not self.no_energy:
                    out["energy"] = self.normalizers["target"].denorm(
                        out["energy"]
                    )
                out["forces"] = self.normalizers["grad_target"].denorm(
                    out["forces"]
                )
            if per_image:
                systemids = [str(i) for i in batch_list[0].fid]
                predictions["id"].extend(systemids)
                predictions["energy"].extend(
                    out["energy"].tolist()
                    #out["energy"].to(torch.float16).tolist()
                )
                batch_natoms = torch.cat(
                    [batch.natoms for batch in batch_list]
                )
                batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                forces = out["forces"].cpu().detach()#.to(torch.float16)
                per_image_forces = torch.split(forces, batch_natoms.tolist())
                per_image_forces = [
                    force.numpy() for force in per_image_forces
                ]
                # evalAI only requires forces on free atoms
                if results_file is not None:
                    _per_image_fixed = torch.split(
                        batch_fixed, batch_natoms.tolist()
                    )
                    _per_image_free_forces = [
                        force[(fixed == 0).tolist()]
                        for force, fixed in zip(
                            per_image_forces, _per_image_fixed
                        )
                    ]
                    _chunk_idx = np.array(
                        [
                            free_force.shape[0]
                            for free_force in _per_image_free_forces
                        ]
                    )
                    per_image_forces = _per_image_free_forces
                    predictions["chunk_idx"].extend(_chunk_idx)
                predictions["forces"].extend(per_image_forces)
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["forces"] = out["forces"].detach()
                return predictions

        predictions["forces"] = np.array(predictions["forces"])
        predictions["chunk_idx"] = np.array(predictions["chunk_idx"])
        predictions["energy"] = np.array(predictions["energy"])
        predictions["id"] = np.array(predictions["id"])
        self.save_results(
            predictions, results_file, keys=["energy", "forces", "chunk_idx"]
        )

        if self.ema:
            self.ema.restore()

        return predictions
    
    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm=True,
    ):
        print('present best val:', self.best_val_metric)
        print('val:', val_metrics[primary_metric]["metric"])
        if (val_metrics[primary_metric]["metric"] < self.best_val_metric):
            #"mae" in primary_metric
            #and val_metrics[primary_metric]["metric"] < self.best_val_metric):
        #) or (val_metrics[primary_metric]["metric"] < self.best_val_metric):
            print('best validation, save model weight...')
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if (self.test_loader is not None) and not self.config["task"]["if_test_final"]:
                #self.predict(
                #    self.test_loader,
                #    results_file="predictions",
                #    disable_tqdm=disable_eval_tqdm,
                #)
                self.test_metric = self.validate(
                            split="test",
                            disable_tqdm=disable_eval_tqdm,
                        )
                if not self.is_debug:
                    print('saving test result at the best validation case...')
                    print(self.config["cmd"]["results_dir"])
                    results_file_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        #f"{self.name}_test_{self.config['task']['task_num']}.pickle",
                        f"{self.name}_"+self.expname+f"_{self.config['task']['task_num']}.pickle",
                    #    f"{self.name}_test.npz",
                    )
                    with open(results_file_path, 'wb') as f:
                        cPickle.dump(self.test_metric, f)

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        if self.test_loader is not None:
            print('test loader exists..')
            print(len(self.test_loader))
        else:
            print('test loader is ', self.test_loader)
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        warmup = self.config["optim"].get("warmup", 0)
        max_iter = self.config["optim"].get('max_iter', None)

        self.best_val_metric = 1e9 #if "mae" in primary_metric else -1.0
        self.test_metric = None
        self.metrics = {}

        if self.config["task"].get("if_pretrain", False):  # load pretrained model
            # load best weight
            chck_name = os.path.join('./pretrain/', "best_checkpoint.pt")
            self.load_checkpoint(checkpoint_path=chck_name)

        if self.config["task"].get("if_eval_fe", False):  # only evaluating force & energy predictions
            # load best weight
            chck_name = os.path.join(self.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt")
            self.load_checkpoint(checkpoint_path=chck_name)
            # predict
            self.predictions_ef = self.predict(self.test_loader, results_file="predictions", disable_tqdm=False)
            return
        # self.predict(
        #    self.test_loader,
        #    results_file="predictions",
        #    disable_tqdm=disable_eval_tqdm,
        # )

        if self.config["optim"].get("if_compile", False):  # if using torch.compile
            self.model = torch_geometric.compile(self.model)

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)
        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            epoch_start_time = time.time()
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)
            print('epochs:', epoch_int)

            if max_iter is not None:
                if self.step > max_iter:
                    break

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                if max_iter is not None:
                    if self.step > max_iter:
                        break

                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch, if_aug=self.config["optim"].get("augment", False))
                    loss = self._compute_loss(out, batch)
                    if self.if_robust_loss and self.step > warmup:  # robust loss
                        loss_rob = self._compute_robust_loss_3pt(out, batch)
                        loss = loss + loss_rob
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    })
                
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                    if self.logger is not None:
                        self.logger.log(
                            log_dict,
                            step=self.step,
                            split="train",
                        )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                    # Evaluate on val set every `eval_every` iterations.
                    if self.step % eval_every == 0:
                        if self.val_loader is not None:
                            val_metrics = self.validate(
                                split="val",
                                disable_tqdm=disable_eval_tqdm,
                            )

                            self.update_best(
                                primary_metric,
                                val_metrics,
                                disable_eval_tqdm=disable_eval_tqdm)

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics)

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()
                                 
                if self.scheduler.scheduler_type in ["ReduceLROnPlateau", "ExponentialLR"]:
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)
                
            if self.early_stopping_lr is not None:
                if self.scheduler.get_lr() <= self.early_stopping_lr:
                    logging.info("Early stopping due to low learning rate")
                    break
            
            if self.early_stopping_time is not None:
                self.elapsed += time.time() - epoch_start_time
                if self.elapsed >= self.early_stopping_time:
                    logging.info("Early stopping due to time limit")
                    break    

        ### after finishing training
        if self.config["task"].get("eval_force_rotation", False):  # evaluate ratation of forces
            chck_name = os.path.join(self.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt")
            self.load_checkpoint(checkpoint_path=chck_name)
            self.test_metric = self.validate(
                split="test",
                disable_tqdm=True,
                if_rotation=True
            )

            if not self.is_debug:
                print('saving test result at the best validation case with forces rotation...')
                print(self.config["cmd"]["results_dir"])
                results_file_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"{self.name}_"+self.expname+f"_{self.config['task']['task_num']}.pickle",
                )
                with open(results_file_path, 'wb') as f:
                    cPickle.dump(self.test_metric, f)


        # if only evaluate final time to save time
        if self.config["task"]["if_test_final"]:
            chck_name = os.path.join(self.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt")
            self.load_checkpoint(checkpoint_path=chck_name)
            self.test_metric = self.validate(
                split="test",
                disable_tqdm=True,
            )
            if not self.is_debug:
                print('saving test result at the best validation case...')
                print(self.config["cmd"]["results_dir"])
                results_file_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"{self.name}_test_{self.config['task']['task_num']}.pickle",
                    #    f"{self.name}_test.npz",
                )
                with open(results_file_path, 'wb') as f:
                    cPickle.dump(self.test_metric, f)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False, max_points=None, if_rotation=False):
        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")
        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if if_rotation:
            evaluator, metrics = Evaluator(task=self.name, forces_rotation=True), {}
        else:
            evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader
        batch_size = self.config["optim"].get(
                        "eval_batch_size", self.config["optim"]["batch_size"])
        if max_points is None:
            max_points = len(loader) * batch_size
        for i, batch in tqdm(
            enumerate(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
            total=np.ceil(max_points // batch_size)
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                if if_rotation:
                    out = self._forward(batch, if_rotation=True)
                else:
                    out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)
            
            if max_points and (i+1) * batch_size >= max_points:
                break
            
        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
            if k.split('_')[-1] == "rmse":
                aggregated_metrics[k]["metric"] = np.sqrt(aggregated_metrics[k]["metric"])


        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    def _forward(self, batch_list, if_aug=False, if_rotation=False):
        if if_aug:
            if self.delta is None:
                delta = self.config["optim"].get("delta", 1.e-3)
            else:
                delta = self.delta
            _batch_list = []
            for batch in batch_list:
                #nb = len(batch.natoms)
                #mean_bond = torch.linalg.vector_norm(batch.pos.view(nb, -1, 3)[:, 1:] \
                #                                     - batch.pos.view(nb, -1, 3)[:, :-1], dim=-1)
                #mean_bond = float(torch.min(mean_bond).data.cpu()) * delta  # 2.e-3
                mean_bond = delta  # 1 x delta [A]
                #dirc = mean_bond * torch.rand(batch.pos.shape)  # A->C
                dirc = 2. * torch.rand(batch.pos.shape) - 1. # A->C
                dirc = mean_bond * dirc  / torch.linalg.vector_norm(dirc, dim=-1, keepdims=True)  # A->C
                _batch_data = batch.detach().clone()
                _batch_data.pos = _batch_data.pos + dirc  # A->C
                _batch_list.append(_batch_data)

            batch_list = _batch_list

        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            if if_rotation:
                out_energy, out_forces, out_rotation = self.model(batch_list, if_rotation)
            else:
                out_energy, out_forces = self.model(batch_list)
        else:
            out_energy = self.model(batch_list)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        out = {
            "energy": out_energy,
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces

        if if_rotation:
            out["rotation"] = out_rotation

        return out

    def _compute_loss(self, out, batch_list):
        loss = []

        # Energy loss.
        if not self.no_energy:
            energy_target = torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            )
            _natoms = torch.cat(
                [batch.natoms.to(self.device) for batch in batch_list], dim=0
            )
            natoms = _natoms.median().item()
            if self.normalizer.get("normalize_labels", False):
                energy_target = self.normalizers["target"].norm(energy_target)
            energy_mult = self.config["optim"].get("energy_coefficient", 1)
            if self.config["optim"].get("loss_energy", "mse") == "energy_per_atom_rmse":
                loss.append(
                    #energy_mult * self.loss_fn["energy"](out["energy"], energy_target, natoms)
                    energy_mult * self.loss_fn["energy"](out["energy"], energy_target) / mt.sqrt(natoms)
                )
            else:
                loss.append(
                    energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
                )

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(
                    force_target
                )

            tag_specific_weights = self.config["task"].get(
                "tag_specific_weights", []
            )
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [
                        batch.tags.float().to(self.device)
                        for batch in batch_list
                    ],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                loss_force_list = torch.abs(out["forces"] - force_target)
                train_loss_force_unnormalized = torch.sum(
                    loss_force_list * weight.view(-1, 1)
                )
                train_loss_force_normalizer = 3.0 * weight.sum()

                # add up normalizer to obtain global normalizer
                distutils.all_reduce(train_loss_force_normalizer)

                # perform loss normalization before backprop
                train_loss_force_normalized = train_loss_force_unnormalized * (
                        distutils.get_world_size() / train_loss_force_normalizer
                )
                loss.append(train_loss_force_normalized)

            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 30)
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](
                            out["forces"][mask], force_target[mask]
                        )
                    )
                else:
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](out["forces"], force_target)
                    )
        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)

        return loss

    @staticmethod
    def _PI_extrapolate(energy, force, dltR, nb):
        return energy - torch.sum(torch.einsum("ab,ab->a", dltR, force).view(nb, -1),
                                  dim=-1, keepdim=False)  # A->B

    def _compute_robust_loss_3pt(self, out, _batch_list):   # 3pt on a triangle
        ''' assuming 1 GPU '''
        #loss_func_robust = torch.nn.L1Loss()
        loss_func_robust = torch.nn.MSELoss()

        batch_list = [_batch_list[0]]
        batch = batch_list[0]
        nb = out["energy"].shape[0]
        if self.delta is None:
            delta = self.config["optim"].get("delta", 1.e-3)
        else:
            delta = self.delta
        if self.if_adv is None:
            if_adv = self.config["optim"].get("if_adv", False)
        else:
            if_adv = self.if_adv
        if self.coef_rob is None:
            coef_rob = self.config["optim"].get("coef_rob", 1.e-1)
        else:
            coef_rob = self.coef_rob
        if self.coef_cyc is None:
            coef_cyc = self.config["optim"].get("coef_cyc", 1.e-2)
        else:
            coef_cyc = self.coef_cyc
        if_local = self.config["optim"].get("if_local", False)

        if self.config["optim"].get("cyc_reg", False):
            alpha = self.config["optim"].get("cyc_reg")
        else:
            alpha = 1.

        if self.config["model_attributes"].get("regress_forces", True):
            forces = out["forces"]
        else:
            pos = batch.pos
            pos = pos.requires_grad_(True)
            forces = - \
                grad(outputs=out["energy"],\
                     inputs=pos,\
                     grad_outputs=torch.ones_like(out["energy"]).to(self.device),\
                     create_graph=True,
                     #retain_graph=True
                     )[0]

        # prepare
        # delta < 2.5e-2 otherwise atoms would be the same possition which cause error!!
        mean_bond = delta # 1 x delta [A]
        # batch_data.pos: [batch * atoms, 3]
        if if_adv:
            #dirc1 = - mean_bond * forces / torch.linalg.vector_norm(forces, dim=-1,
            dev_force = forces - batch.force.to(self.device)
            dirc1 = - mt.sqrt(3.) * mean_bond * dev_force / torch.linalg.vector_norm(dev_force, dim=-1,
                                                                                     keepdims=True)
            # A->C , adversarial direction, negative sign
        else:
            #dirc1 = 2. * torch.rand(batch.pos.shape).to(self.device) - 1.  # A->C
            dirc1 = torch.rand(batch.pos.shape).to(self.device)  # A->C
            dirc1 = mean_bond * dirc1 #/ torch.linalg.vector_norm(dirc1, dim=-1, keepdims=True) # A->C
        #dirc2 = 2. * torch.rand(batch.pos.shape).to(self.device) - 1.  # A->B
        dirc2 = torch.rand(batch.pos.shape).to(self.device)  # A->B
        dirc2 = mean_bond * dirc2  # / torch.linalg.vector_norm(dirc2, dim=-1, keepdims=True)   # A->B
        dirc3 = dirc1 * alpha - dirc2  # B->C
        #dirc4 = 2. * torch.rand(batch.pos.shape).to(self.device) - 1.  # A->D
        dirc4 = torch.rand(batch.pos.shape).to(self.device)  # A->D
        dirc4 = mean_bond * dirc4 #/ torch.linalg.vector_norm(dirc4, dim=-1, keepdims=True)     # A->D
        dirc5 = dirc1 - dirc4 # D->C
        if if_local:
            mask = torch.ones_like(dirc1).to(self.device)
            mask_unity = rng.choice([0, 1], size=[mask.shape[0]], p=(1 - if_local, if_local))
            mask *= torch.from_numpy(mask_unity[:, None]).to(self.device)
            #
            dirc1 *= mask
            dirc2 *= mask
            dirc3 *= mask
            dirc4 *= mask
            dirc5 *= mask
        dirc1 = dirc1.detach()
        dirc2 = dirc2.detach()
        dirc3 = dirc3.detach()
        dirc4 = dirc4.detach()
        dirc5 = dirc5.detach()

        if self.config["optim"].get("cyc_reg", False):
            batch_dataC = batch.detach()  # regulation
        else:
            batch_dataC = batch.detach().clone()
        batch_dataC.pos = batch_dataC.pos + dirc1  # A->C
        outC = self._forward([batch_dataC])
        if self.config["optim"].get("cls_pseudo", False):  # if classical pseudo label with label
            V_AC = self._PI_extrapolate(batch.y.to(self.device), batch.force.to(self.device), dirc1, nb)  # A->C Euler
        else:
            V_AC = out["energy"] - torch.sum(torch.einsum("ab,ab->a", dirc1, forces).view(nb, -1),
                                             dim=-1, keepdim=False)  # A->C
        loss = coef_rob * loss_func_robust(outC["energy"], V_AC)  # robust loss at C
        if self.config["model_attributes"].get("regress_forces", True) is False:
            return loss

        if self.config["optim"].get("two_point_consistency", False):  # if two point consistency
            V_CA = outC["energy"] - torch.sum(torch.einsum("ab,ab->a", -dirc1, outC["forces"]).view(nb, -1),
                                                  dim=-1, keepdim=False)  # C->A
            loss += coef_cyc * loss_func_robust(out["energy"], V_CA)  # two point consistency loss
            return loss

        if self.config["optim"].get("cyc_reg", False):
            batch_dataB = batch.detach()  # regulation
        else:
            batch_dataB = batch.detach().clone()
        batch_dataB.pos = batch_dataB.pos + dirc2  # A->B

        outB = self._forward([batch_dataB])
        if self.config["model_attributes"].get("regress_forces", True):
            forceB = outB["forces"]
        else:
            pos = batch_dataB.pos
            pos = pos.requires_grad_(True)
            forceB = - \
                grad(outputs=outB["energy"],\
                     inputs=pos,\
                     grad_outputs=torch.ones_like(outB["energy"]).to(self.device),\
                     create_graph=True, retain_graph=True)[0]

        if self.config["optim"].get("if_neuralODE_cyc", False):
            V_BC = self._PI_extrapolate_ODE(outB["energy"], dirc3, batch_dataB, self._forward, nb,
                                            self.device,
                                            solver=self.config["optim"].get("neuralODE_solver", 'rk4')
                                            )  # via NeuralODE
        else:
            V_BC = outB["energy"] - torch.sum(torch.einsum("ab,ab->a", dirc3, forceB),
                                              dim=-1, keepdim=False) # B->C
        if self.config["optim"].get("consis_case", 'A') == 'A':  # if consistency 2, at C, basic
            if self.config["optim"].get("consis_loss", 'L2') == 'L2':
                loss += coef_cyc * loss_func_robust(outC["energy"], V_BC)  # consistency loss 2, at C, basic
            elif self.config["optim"].get("consis_loss", 'L2') == 'relu': # consistency loss with ReLU
                residual = torch.nn.functional.relu(
                        torch.abs(outC["energy"] - V_BC) - torch.abs(V_BC).detach().clone() * delta ** 2)
                loss += coef_cyc * torch.mean(residual)
        elif self.config["optim"].get("consis_case", 'A') == 'B':  # if consistency 1, at C
            loss += coef_cyc * loss_func_robust(V_AC, V_BC)  # consistency loss 1, at C
        elif self.config["optim"].get("consis_case", 'A') == 'C':  # if consistency 3, at B
            if self.config["optim"].get("if_neuralODE_cyc", False):
                V_CB = self._PI_extrapolate_ODE(outC["energy"], -dirc3, batch_dataC, self._forward, nb,
                                                self.device,
                                                solver=self.config["optim"].get("neuralODE_solver", 'rk4')
                                                )  # via NeuralODE
            else:
                V_CB = self._PI_extrapolate(outC["energy"], outC['forces'], -dirc3, nb)  # C->B for cyc3
            loss += coef_cyc * loss_func_robust(outB["energy"], V_CB)  # consistency loss 3
        elif self.config["optim"].get("consis_case", 'A') == 'D':  # if consistency 4
            batch_dataD = batch.detach().clone()
            batch_dataD.pos = batch_dataD.pos + dirc4  # A->D

            outD = self._forward([batch_dataD])
            if self.config["model_attributes"].get("regress_forces", True):
                forceD = outD["forces"]
            else:
                pos = batch_dataD.pos
                pos = pos.requires_grad_(True)
                forceD = - \
                    grad(outputs=outD["energy"],\
                         inputs=pos,\
                         grad_outputs=torch.ones_like(outD["energy"]).to(self.device),\
                         create_graph=True, retain_graph=True)[0]

            if self.config["optim"].get("if_neuralODE_cyc", False):
                V_DC = self._PI_extrapolate_ODE(outD["energy"], dirc5, batch_dataD, self._forward, nb,
                                                self.device,
                                                solver=self.config["optim"].get("neuralODE_solver", 'rk4')
                                                )  # via NeuralODE
            else:
                V_DC = self._PI_extrapolate(outD["energy"], forceD, dirc5, nb)  # D->C Euler
            loss += coef_cyc * loss_func_robust(V_DC, V_BC)  # consistency loss 4
        else:  # all
            loss += coef_cyc * loss_func_robust(outC["energy"], V_BC)   #at C
            loss += coef_cyc * loss_func_robust(V_AC, V_BC)  # consistency loss at C from A and B
            V_CB = self._PI_extrapolate(outC["energy"], outC['forces'], -dirc3, nb)  # C->B for cyc3
            loss += coef_cyc * loss_func_robust(outB["energy"], V_CB)  # consistency loss at B
            V_CA = self._PI_extrapolate(outC["energy"], outC["forces"], -dirc1, nb)  # A->C Euler
            loss += coef_cyc * loss_func_robust(out["energy"], V_CA)  # consistency loss at A

        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "forces": torch.cat([batch.force.to(self.device) for batch in batch_list], dim=0),
            "natoms": natoms,
        }
        
        if not self.no_energy:
            target.update({
                "energy": torch.cat([batch.y.to(self.device) for batch in batch_list], dim=0)
            })

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask = fixed == 0
            out["forces"] = out["forces"][mask]
            target["forces"] = target["forces"][mask]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(torch.sum(mask[s_idx:s_idx + natoms]).item())
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            out["natoms"] = torch.LongTensor(natoms_free).to(self.device)

        if self.normalizer.get("normalize_labels", False):
            if not self.no_energy:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])

        metrics = evaluator.eval(out, target, prev_metrics=metrics)

        return metrics

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model.module, "shared_parameters"):
            for p, factor in self.model.module.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )
        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None:
                self.logger.log(
                    {"grad_norm": grad_norm}, step=self.step, split="train"
                )
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()

    def save_results(self, predictions, results_file, keys):
        if results_file is None:
            return

        results_file_path = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{self.name}_{results_file}_{distutils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )

        distutils.synchronize()
        if distutils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.config["cmd"]["results_dir"],
                f"{self.name}_{results_file}.npz",
            )

            for i in range(distutils.get_world_size()):
                rank_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"{self.name}_{results_file}_{i}.npz",
                )
                rank_results = np.load(rank_path, allow_pickle=True)
                gather_results["ids"].extend(rank_results["ids"])
                for key in keys:
                    gather_results[key].extend(rank_results[key])
                os.remove(rank_path)

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            gather_results["ids"] = np.array(gather_results["ids"])[idx]
            for k in keys:
                if k == "forces":
                    gather_results[k] = np.concatenate(
                        np.array(gather_results[k])[idx]
                    )
                elif k == "chunk_idx":
                    gather_results[k] = np.cumsum(
                        np.array(gather_results[k])[idx]
                    )[:-1]
                else:
                    gather_results[k] = np.array(gather_results[k])[idx]

            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)
