import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from trainers.calibration.base_model.coop import CustomCLIP as CoOpModel
from trainers.calibration.base_model.cocoop import CustomCLIP as CoCoOpModel
from trainers.calibration.base_model.kgcoop import CustomCLIP as KgCoOpModel
from trainers.calibration.base_model.maple import CustomCLIP as MaPLeModel
from trainers.calibration.base_model.proda import CustomCLIP as ProDAModel
from trainers.calibration.base_model.prograd import CustomCLIP as ProgradModel
from trainers.calibration.base_model.clip_adapter import CustomCLIP as CLIPAdapterModel
from trainers.calibration.base_model.zsclip import CustomCLIP as ZeroShotModel
from trainers.calibration.base_model.promptsrc import CustomCLIP as PromptSRCModel


def get_middle_model(cfg, classnames):
    model_name = cfg.TRAINER.NAME

    models = {
        'coop': CoOpModel,
        'cocoop': CoCoOpModel,
        'kgcoop': KgCoOpModel,
        'maple': MaPLeModel,
        'proda': ProDAModel,
        'prograd': ProgradModel,
        'promptsrc': PromptSRCModel,
        'clip_adapter': CLIPAdapterModel,
        'zeroshotclip': ZeroShotModel
    }

    model = models.get(model_name.lower())
    
    if model:
        return model(cfg, classnames)
    else:
        raise ValueError(f'Unknown model: {model_name}')

