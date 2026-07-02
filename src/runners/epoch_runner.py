import glob
import os
import time
from abc import ABCMeta
from importlib import import_module

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..losses.exceptions import NoGradientError
from ..models.rotation_utils import align_rotation, compute_angles
from ..utils import get_logger
from .utils import *


def _name_to_class(name):
    return "".join(n.capitalize() for n in name.split("_"))


class EpochRunner(metaclass=ABCMeta):
    """Base runner from training/testing and logging"""

    def _metrics(self, outputs_model, inputs_data, mode="train") -> dict:
        """Compute metrics that is saved in tensorboard.

        Args:
            outputs_model (dict): returned by `self.model`
            inputs_data (dict): returned by dataset

        Returns:
            A dict containing different metrics.
        """

    def _get_images(self, outputs_model, inputs_data, mode="train"):
        """Visualizing results

        Args:
            outputs_model (dict): model output
            inputs_data (dict): returned by dataset

        Returns:
            None
        """

    def test(self):
        """
        Test model after training.

        Returns:
            None
        """
        test_configs = self.configs["test_configs"]

        def get_metrics(outputs_model, inputs_data):
            with torch.no_grad():
                R_pred: torch.Tensor = outputs_model["rotations"]
                weights: torch.Tensor = outputs_model.get("weights", None)
                R_gt: torch.Tensor = inputs_data["rotations"]
                mask = inputs_data.get("mask", None)
                if mask is not None:
                    s = mask[0, :].sum()
                    R_pred = R_pred[mask].reshape((mask.shape[0], s, 3, 3))
                    R_gt = R_gt[mask].reshape((mask.shape[0], s, 3, 3))

                R_pred = align_rotation(R_pred, R_gt)
                angles = compute_angles(R_pred, R_gt).abs()
                angles3deg = (angles <= 3).float().mean()
                angles5deg = (angles <= 5).float().mean()
                angles10deg = (angles <= 10).float().mean()
                metrics = {
                    "mean": angles.mean(),
                    "acc_3": angles3deg,
                    "acc_5": angles5deg,
                    "acc_10": angles10deg,
                }
                if weights is not None:
                    std, m = torch.std_mean(weights)
                    metrics.update(
                        {
                            "weight_std": std,
                            "weight_mean": m,
                        }
                    )
            return metrics, angles

        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(test_configs["device"])
        use_amp = test_configs.get("use_amp", False)

        workspace = test_configs["workspace"]
        checkpoint_path = test_configs.get("checkpoint", "")
        if checkpoint_path != "" and checkpoint_path != "None":
            if isinstance(checkpoint_path, int):
                checkpoint_path = "ckpt_{:0>4}.pth".format(checkpoint_path)
            checkpoint_path = os.path.join(workspace, checkpoint_path)
        elif checkpoint_path != "None":
            checkpoint_path = os.path.join(workspace, "best.pth")

        if checkpoint_path != "None":
            print(f"Load checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=test_configs["device"]
            )
            if "epoch" in checkpoint.keys():
                print("Epoch: {}".format(checkpoint["epoch"]))
            else:
                print("Step: {}".format(checkpoint["step"]))
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        ####################
        # setup dataloader #
        ####################
        batch_size = self.dataset_configs["batch_size"]
        num_workers = self.dataset_configs["num_workers"]
        test_dataset = self.dataset(self.dataset_configs, "test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
        )

        self.model.eval()
        # test
        angles_all = []
        avg_meter = DictAverageMeter()
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(test_loader):
                data = to_device(data, self.device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    model_outputs = self.model(data, mode="test")

                try:
                    loss = self.loss_term(model_outputs, data, mode="test")
                except NoGradientError:
                    continue

                if isinstance(loss, tuple):
                    loss, items = loss
                else:
                    items = None

                if items is not None:
                    items.update({"loss": float(loss)})
                else:
                    items = {"loss": float(loss)}

                metrics, angles = get_metrics(model_outputs, data)
                angles_all.append(angles)
                if metrics is not None:
                    items.update(metrics)

                avg_meter.update(tensor2float(items))

        angles_all = torch.concat(angles_all).cpu()
        med = tensor2float(torch.median(angles_all).cpu())
        metrics = avg_meter.mean()
        metrics["checkpoint"] = checkpoint_path
        if checkpoint_path != "None":
            if "epoch" in checkpoint.keys():
                metrics["epoch"] = checkpoint["epoch"]
            else:
                metrics["step"] = checkpoint["step"]
        metrics["median"] = med
        print(dict_to_str(metrics))
        if test_configs.get("file_path", "") != "":
            with open(os.path.join(workspace, test_configs["file_path"]), "w") as f:
                yaml.dump(metrics, f, default_flow_style=False)

    def init_weights(self):
        """Initialize model weight at the beggining of training"""

    def info(self, logger, message):
        if self.local_rank <= 0:
            logger.info(message)

    def __init__(self, configs):
        project = configs.get("project", "src")
        self.configs = configs

        self.confgs = configs
        self.model_configs = configs["model_configs"]
        self.dataset_configs = configs["dataset_configs"]
        self.loss_configs = configs["loss_configs"]

        model = import_module(".models.{}".format(self.model_configs["name"]), project)
        dataset = import_module(
            ".datasets.{}".format(self.dataset_configs["name"]), project
        )
        loss = import_module(".losses.{}".format(self.loss_configs["name"]), project)

        self.model = getattr(model, _name_to_class(self.model_configs["name"]))(configs)
        # class
        self.dataset = getattr(dataset, _name_to_class(self.dataset_configs["name"]))
        self.loss_term = getattr(loss, _name_to_class(self.loss_configs["name"]))(
            configs
        )

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.distributed = self.local_rank >= 0

    def train(self):
        optimizer_configs = self.configs["optimizer_configs"]
        train_configs = self.configs["train_configs"]

        max_grad_norm = train_configs.get("max_grad_norm")
        batch_repeat = train_configs.get("batch_repeat", -1)

        accum_num_batches = train_configs.get("accum_num_batches", 1)

        # device
        workspace = train_configs["workspace"]
        self.logger = get_logger(workspace)
        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(train_configs["device"])

        self.info(
            self.logger,
            "Parameter count: {}".format(
                sum(p.numel() for p in self.model.parameters())
            ),
        )

        #######################
        # setup data parallel #
        #######################
        self.model.to(self.device)
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=train_configs.get(
                    "find_unused_parameters", False
                ),
            )
            world_size = dist.get_world_size()

        ####################
        # set up optimizer #
        ####################
        optim = import_module("torch.optim")
        name = optimizer_configs["name"]
        params = optimizer_configs.copy()
        params.pop("name")
        scheduler_configs = params.pop("lr_scheduler", None)
        self.optimizer = getattr(optim, name)(self.model.parameters(), **params)

        ####################
        # set up scheduler #
        ####################
        if scheduler_configs is None:
            # seudo scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda epoch: 1
            )
        else:
            lib_schedulers = import_module("torch.optim.lr_scheduler")
            name_scheduler = scheduler_configs["name"]
            self.params = scheduler_configs.copy()
            self.params.pop("name")
            self.params = {
                k: (v if k != "lr_lambda" else eval(v)) for k, v in self.params.items()
            }
            self.lr_scheduler = getattr(lib_schedulers, name_scheduler)(
                self.optimizer, **self.params
            )

        ################################
        # continue from the checkpoint #
        ################################
        self.epoch = 0
        self.loss_val_min = np.inf
        self.init_weights()
        resume = train_configs.get("resume", False)
        if resume:
            checkpoint = train_configs.get("checkpoint", None)
            if checkpoint is None:
                files = glob.glob(os.path.join(workspace, "ckpt_*.pth"))
                files = [os.path.basename(f) for f in files]
                files.sort()
                if len(files) > 0:
                    checkpoint = files[-1]
                else:
                    self.info(self.logger, "No checkpoint found!")

            if checkpoint is not None:
                if isinstance(checkpoint, int):
                    checkpoint = "ckpt_{:0>4}.pth".format(checkpoint)
                checkpoint = os.path.join(workspace, checkpoint)
                self.info(self.logger, "Loading checkpoint from {}.".format(checkpoint))
                self.load(checkpoint)
        load_from = train_configs.get("load_from", None)
        if load_from:
            self.info(self.logger, f"Loading model checkpoint from {load_from}")
            self.load(load_from, model_only=True)

        ####################
        # setup dataloader #
        ####################
        batch_size = self.dataset_configs["batch_size"]
        num_workers = self.dataset_configs["num_workers"]
        train_dataset = self.dataset(self.dataset_configs, "train")
        val_dataset = self.dataset(self.dataset_configs, "val")
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, sampler=val_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=num_workers,
            )

        ##########################
        # initialize tensorboard #
        ##########################
        if train_configs.get("enable_tensorboard", True) and self.local_rank <= 0:
            writer = SummaryWriter(workspace)
        else:
            writer = None

        ckpt_best = "best.pth"
        use_amp = train_configs.get("use_amp", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        while self.epoch < train_configs["num_epochs"]:
            # Train
            avg_meter = DictAverageMeter()
            if self.distributed:
                train_loader.sampler.set_epoch(self.epoch)
            self.model.train()
            self.optimizer.zero_grad()
            accum_count = 0
            for i, data in enumerate(train_loader):
                t1 = time.time()
                data = to_device(data, self.device)
                if batch_repeat > 0:
                    for k, v in data.items():
                        n_ = v.ndim - 1
                        data[k] = v.repeat(batch_repeat, *(1,) * n_)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    model_outputs = self.model.forward(data)
                    try:
                        loss = self.loss_term(model_outputs, data, mode="train")
                    except NoGradientError:
                        self.info(
                            self.logger,
                            "[Train] [Epoch {}/{}] [Iteration {}/{}] {}".format(
                                self.epoch + 1,
                                train_configs["num_epochs"],
                                i + 1,
                                len(train_loader),
                                "No Gradient!",
                            ),
                        )
                        continue
                if isinstance(loss, tuple):
                    loss, items = loss
                    if self.distributed:
                        _ = [dist.all_reduce(x) for x in items.values()]
                        items = {k: v / world_size for k, v in items.items()}
                else:
                    items = None

                # 梯度累积: loss除以accum_num_batches来取平均
                loss = loss / accum_num_batches
                self.scaler.scale(loss).backward()
                accum_count += 1

                # 只有在累积达到指定数量后才更新参数
                if accum_count == accum_num_batches:
                    if max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=max_grad_norm
                        )
                        if writer is not None and self.local_rank <= 0:
                            save_scalars(
                                writer,
                                "train_avg",
                                {"grad_norm": total_norm},
                                self.epoch + 1,
                            )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    accum_count = 0
                t2 = time.time()

                global_step = len(train_loader) * self.epoch + i

                # 使用平均后的 loss 用于日志记录
                loss_display = loss.detach() * accum_num_batches
                if self.distributed:
                    dist.all_reduce(loss_display)
                    loss_display /= world_size
                self.info(
                    self.logger,
                    "[Train] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s".format(
                        self.epoch + 1,
                        train_configs["num_epochs"],
                        i + 1,
                        len(train_loader),
                        float(loss_display),
                        t2 - t1,
                    ),
                )

                if items is not None:
                    items.update({"loss": float(loss_display)})
                else:
                    items = {"loss": float(loss_display)}

                metrics = self._metrics(model_outputs, data, mode="train")
                if metrics is not None:
                    if self.distributed:
                        _ = [dist.all_reduce(x) for x in metrics.values()]
                        metrics = {k: v / world_size for k, v in metrics.items()}
                    items.update(metrics)
                # save in average meter
                avg_meter.update(tensor2float(items))

                if (global_step + 1) % train_configs["summary_freq"] == 0:
                    if self.local_rank <= 0:
                        images = self._get_images(model_outputs, data)
                        save_scalars(writer, "train", items, global_step)
                        if images is not None:
                            save_images(writer, "train", images, global_step)

            if writer is not None and avg_meter.count != 0 and self.local_rank <= 0:
                save_scalars(writer, "train_avg", avg_meter.mean(), self.epoch + 1)

                self.info(
                    self.logger,
                    "[Train] [Epoch {}/{}] {}".format(
                        self.epoch + 1,
                        train_configs["num_epochs"],
                        dict_to_str(avg_meter.mean()),
                    ),
                )
            self.lr_scheduler.step()

            for g in self.optimizer.param_groups:
                self.info(
                    self.logger,
                    "Adjusting learning rate of group 0 to {}.".format(g["lr"]),
                )

            if (self.epoch + 1) % train_configs[
                "checkpoint_interval"
            ] == 0 and self.local_rank <= 0:
                self.save(workspace)

            train_loader.dataset.process_epoch()

            # Validation
            if (self.epoch + 1) % train_configs.get("val_interval", 1) == 0:
                avg_meter = DictAverageMeter()
                if self.distributed:
                    val_loader.sampler.set_epoch(self.epoch)
                with torch.no_grad():
                    self.model.eval()
                    for i, data in enumerate(val_loader):
                        data = to_device(data, self.device)
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            model_outputs = self.model(data, mode="val")

                            t1 = time.time()
                            try:
                                loss = self.loss_term(model_outputs, data, mode="val")
                            except NoGradientError:
                                self.info(
                                    self.logger,
                                    "[Val] [Epoch {}/{}] [Iteration {}/{}] {}".format(
                                        self.epoch + 1,
                                        train_configs["num_epochs"],
                                        i + 1,
                                        len(val_loader),
                                        "No Gradient!",
                                    ),
                                )
                                continue
                        t2 = time.time()

                        if isinstance(loss, tuple):
                            loss, items = loss
                            if self.distributed:
                                _ = [dist.all_reduce(x) for x in items.values()]
                                items = {k: v / world_size for k, v in items.items()}
                                dist.all_reduce(loss)
                                loss /= world_size
                        else:
                            items = None

                        self.info(
                            self.logger,
                            "[Val] [Epoch {}/{}] [Iteration {}/{}] Loss: {:.3f} | Time cost: {:.3f} s".format(
                                self.epoch + 1,
                                train_configs["num_epochs"],
                                i + 1,
                                len(val_loader),
                                float(loss),
                                t2 - t1,
                            ),
                        )
                        global_step = len(val_loader) * self.epoch + i + 1

                        if items is not None:
                            items.update({"loss": float(loss)})
                        else:
                            items = {"loss": float(loss)}

                        metrics = self._metrics(model_outputs, data, mode="val")
                        if metrics is not None:
                            items.update(metrics)

                        avg_meter.update(tensor2float(items))

                    if avg_meter.count != 0:
                        meter_mean = avg_meter.mean()
                        self.info(
                            self.logger,
                            "[Val] [Epoch {}/{}] {}".format(
                                self.epoch + 1,
                                train_configs["num_epochs"],
                                dict_to_str(meter_mean),
                            ),
                        )
                        if writer is not None:
                            save_scalars(writer, "val", meter_mean, self.epoch + 1)

                        loss_current = meter_mean["loss"]
                        if loss_current < self.loss_val_min:
                            self.loss_val_min = loss_current
                            self.info(
                                self.logger,
                                "Update best ckeckpoint, saved as {}".format(ckpt_best),
                            )
                            self.save(workspace, ckpt_best)

            self.epoch += 1

    def save(self, out_dir, ckpt_name=None):
        ckpt_name = (
            "ckpt_{:0>4}.pth".format(self.epoch + 1) if ckpt_name is None else ckpt_name
        )
        save_path = os.path.join(out_dir, ckpt_name)
        torch.save(
            {
                "epoch": self.epoch + 1,
                "model_state_dict": self.model.module.state_dict()
                if self.distributed
                else self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict()
                if hasattr(self, "scaler")
                else None,
                "loss_val_min": self.loss_val_min,
            },
            save_path,
        )

    def load(self, checkpoint_path, model_only=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        if not model_only:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            if (
                hasattr(self, "scaler")
                and "scaler_state_dict" in checkpoint
                and checkpoint["scaler_state_dict"] is not None
            ):
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.loss_val_min = checkpoint["loss_val_min"]

            for g in self.optimizer.param_groups:
                self.info(
                    self.logger,
                    "Adjusting learning rate of group 0 to {}.".format(g["lr"]),
                )

