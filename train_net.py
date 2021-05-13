"""
Written by TuyenNQ, email:s1262008@u-aizu.ac.jp

AdaptiveLP Training Transcript
"""
from datetime import timedelta
import os

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

from AdaptiveLPDetection.config.add_adaptivelp_config import add_adaptivelp_config
from AdaptiveLPDetection.engine.trainer import Trainer

def setup(args):
    cfg = get_cfg()
    add_adaptivelp_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "adaptivelp" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="adaptivelp")
    return cfg


def main(args):
    cfg = setup(args)
    # disable strict kwargs checking: allow one to specify path handle
    # hints through kwargs, like timeout in DP evaluation
    # PathManager.set_strict_kwargs_checking(False)

    # add dataset train and test here
    cfg.DATASETS.TRAIN = ("kittidata_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
    # cfg.MODEL.ADAPTIVE_LP.BACKBONE_WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    # change output saving checkpoint dir here
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    # if not os.listdir(cfg.OUTPUT_DIR):
    #     #if the OUTPUT_DIR is empty will initialize the pretraining for backbone model
    #     trainer.load_backbone_pretrained(trainer.model)
    trainer.resume_or_load(resume=args.resume)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    # timeout = (
    #     DEFAULT_TIMEOUT if cfg.DENSEPOSE_EVALUATION.DISTRIBUTED_INFERENCE else timedelta(hours=4)
    # )
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        # timeout=timeout,
    )