import os
import time

import torch
import yaml
from scipy.io import savemat
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.rotation_utils import align_rotation, compute_angles
from .epoch_runner import EpochRunner
from .utils import to_device


class AbsRotRunner(EpochRunner):
    def __init__(self, configs):
        super().__init__(configs)

    @torch.no_grad()
    def export(self):
        """
        Export view graph.

        Returns:
            None
        """
        export_configs = self.configs["export_configs"]

        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(export_configs["device"])

        workspace = export_configs["workspace"]
        checkpoint_path = export_configs.get("checkpoint", "")
        if checkpoint_path != "" and checkpoint_path != "None":
            if isinstance(checkpoint_path, int):
                checkpoint_path = "ckpt_{:0>4}.pth".format(checkpoint_path)
            checkpoint_path = os.path.join(workspace, checkpoint_path)
        elif checkpoint_path != "None":
            checkpoint_path = os.path.join(workspace, "best.pth")

        if checkpoint_path != "None":
            print(f"Load checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=export_configs["device"]
            )
            print("Epoch: {}".format(checkpoint["epoch"]))
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

        out_dir = os.path.join(workspace, export_configs["outdir"])
        os.makedirs(out_dir, exist_ok=True)
        meta_path = os.path.join(out_dir, "meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(export_configs, f, default_flow_style=False)

        max_images = export_configs.get("max_images", 100000)
        min_images = export_configs.get("min_images", 0)
        print("k: {}".format(self.model.k))
        print("outdir: {}".format(out_dir))
        print("min_images: {}".format(min_images))
        print("max_images: {}".format(max_images))
        self.model.eval()
        bar = tqdm(test_loader)
        for data in bar:
            name = data["name"][0]
            bar.set_description("[{}]".format(name))
            data.pop("name")
            data = to_device(data, self.device)
            out_file = os.path.join(out_dir, "{}.mat".format(name))
            R_gt = data["rotations"].cpu()

            t1 = time.time()
            model_outputs = self.model(data, mode="test")
            t2 = time.time()
            dt = t2 - t1
            bar.set_description("[{}] time: {:.1f}".format(name, dt))

            rel_rots = model_outputs["rel_rotations"]
            weights = model_outputs["weights"]
            inds = model_outputs["inds"]

            mdict = {
                "rel_rots": rel_rots.cpu().numpy(),
                "weights": weights.cpu().numpy(),
                "inds": inds.cpu().numpy() + 1,
                "R_gt": R_gt.cpu().numpy(),
                "time": dt,
            }

            savemat(out_file, mdict)

    def _metrics(self, outputs_model, inputs_data, mode="train") -> dict:
        with torch.no_grad():
            R_pred: torch.Tensor = outputs_model["rotations"]
            weights: torch.Tensor = outputs_model.get("weights", None)
            R_gt: torch.Tensor = inputs_data["rotations"]

            R_pred = align_rotation(R_pred, R_gt)
            angles = compute_angles(R_pred, R_gt).abs()
            angles1deg = (angles <= 1).float().mean()
            angles3deg = (angles <= 3).float().mean()
            angles5deg = (angles <= 5).float().mean()
            angles10deg = (angles <= 10).float().mean()
            angles20deg = (angles <= 20).float().mean()
            out = {
                "mean": angles.mean(),
                "acc_3": angles3deg,
                "acc_5": angles5deg,
                "acc_10": angles10deg,
            }
            if weights is not None:
                std, m = torch.std_mean(weights)
                out.update(
                    {
                        "weight_std": std,
                        "weight_mean": m,
                    }
                )
        return out
