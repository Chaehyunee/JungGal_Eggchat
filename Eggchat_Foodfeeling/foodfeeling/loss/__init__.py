from foodfeeling.loss.coco_loss import COCOLoss
from foodfeeling.loss.cosface import CosFace
from foodfeeling.loss.center_loss import CenterLoss
from foodfeeling.loss.crf_loss import CRFLoss
from foodfeeling.loss.cross_entropy_loss import CrossEntropyLoss
from foodfeeling.loss.gaussian_mixture import GaussianMixture

__ALL__ = [CenterLoss, COCOLoss, CosFace, CRFLoss, CrossEntropyLoss, GaussianMixture]
