"""
Written by TuyenNQ, email: s1262008@u-aizu.ac.jp

This is Trainer script for AdaptiveLP:
    add load_backbone_pretrained method for DefaultTrainer
"""

from torch import nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)
    @classmethod
    def load_backbone_pretrained(self, model: nn.Module):
        DetectionCheckpointer(model.backbone).resume_or_load(self.cfg.MODEL.ADAPTIVE_LP.BACKBONE_WEIGHTS,resume=False)


