"""
FloodDetector: pretrained SAR flood UNet (ResNet34), frozen, inference only.

The HF repo ships only a checkpoint (`model.pt`, key `model_state`), so the
architecture comes from segmentation-models-pytorch, built exactly as in the
model card. Output is the softmax water probability rather than argmax, so
overlapping patches can be averaged before thresholding.
For two classes, `prob > 0.5` is the same decision as argmax.
"""

import hashlib
import logging
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from huggingface_hub import hf_hub_download
from torch import nn

log = logging.getLogger(__name__)


class FloodDetector(nn.Module):
    """
    (B, 2, H, W) standardized VV/VH -> (B, H, W) water probability in [0, 1]
    """

    def __init__(self,
                 repo: str,
                 cache_dir: Path,
                 device: str = "cpu",
                 filename: str = "model.pt"):
        super().__init__()
        path = hf_hub_download(repo, filename, cache_dir=str(cache_dir))
        # weights_only=False: the checkpoint also stores training metadata.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self.net = smp.create_model("unet", encoder_name="resnet34", encoder_weights=None,
                                    in_channels=2, classes=2)
        self.net.load_state_dict(ckpt["model_state"]) 
        self.net.requires_grad_(False)                 # frozen
        self.eval()                                    
        self.device = torch.device(device)
        self.to(self.device)

        self.info = {"hf_repo": repo, "file": filename,
                     "architecture": "smp Unet(resnet34, in_channels=2, classes=2)",
                     "output": "softmax probability of class 1 (water)",
                     "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_val_iou": ckpt.get("val_iou")}
        log.info("Model loaded: %s (epoch %s, val IoU %s) on %s",
                 repo, self.info["checkpoint_epoch"], self.info["checkpoint_val_iou"], self.device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x.to(self.device, torch.float32))
        return logits.softmax(dim=1)[:, 1]