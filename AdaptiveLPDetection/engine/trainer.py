"""
Written by TuyenNQ, email: s1262008@u-aizu.ac.jp

This is Trainer script for AdaptiveLP:
    add load_backbone_pretrained method for DefaultTrainer
"""

from torch import nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer

from AdaptiveLPDetection.data.build import AdaptiveDatasetMapper
from detectron2.data.build import build_detection_train_loader

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AdaptiveDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)


