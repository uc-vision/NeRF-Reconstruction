import torch
import torchmetrics
import math as m

from torch.nn import functional as F

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerf.losses import psnr


class MetricWrapper(object):
    pass


class SSIMWrapper(MetricWrapper):
    def __init__(self):
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def __call__(self, pred, target):
        pred_bchw = torch.clamp(torch.Tensor(pred.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)

        target_bchw = torch.clamp(target.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)

        return self.lpips(pred_bchw, target_bchw)


class LPIPSWrapper(MetricWrapper):
    def __init__(self):
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to('cuda')

    def __call__(self, pred, target):
        pred_bchw = torch.clamp(torch.Tensor(pred.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        pred_scale = torch.clamp(pred_bchw * 2 - 1, min=-1, max=1)

        target_bchw = torch.clamp(target.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        target_scale = torch.clamp(target_bchw * 2 - 1, min=-1, max=1)

        return self.scale(pred_scale, target_scale)
        

class PSNRWrapper(MetricWrapper):
    def __call__(self, pred, target):
        pred_bchw = torch.clamp(torch.Tensor(pred.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        target_bchw = torch.clamp(target.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        return psnr(pred, target)
