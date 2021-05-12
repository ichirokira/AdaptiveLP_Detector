from detectron2.config import CfgNode as CN

def add_adaptivelp_config(cfg):
    cfg.MODEL.ADAPTIVE_LP = CN()
    cfg.MODEL.ADAPTIVE_LP.POOL_SIZE = [1,16,32,64]
    #cfg.MODEL.ADAPTIVE_LP.NUM_CLASSES = 9
    cfg.MODEL.ADAPTIVE_LP.USING_MASK = True
    cfg.MODEL.ADAPTIVE_LP.BACKBONE_WEIGHTS = ""

