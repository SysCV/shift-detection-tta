import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from shift_tta.registry import MODELS


@MODELS.register_module()
class YOLOXConsistencyLoss(nn.Module):
    """YOLOXConsistencyLoss
    Args:
        weight (float, optional): Weight of the loss. Default to 1.0.
        obj_weight (float, optional): Weight of the objectness consistency loss.
            Default to 1.0.
        reg_weight (float, optional): Weight of the regression consistency loss.
            Default to 1.0.
        cls_weight (float, optional): Weight of the classification consistency loss.
            Default to 1.0.
    """

    def __init__(self,
                 weight=1.0,
                 obj_weight=1.0,
                 reg_weight=1.0,
                 cls_weight=1.0,
        ):
        super(YOLOXConsistencyLoss, self).__init__()
        self.weight = weight
        self.obj_weight = obj_weight
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
    
    def forward(self, inputs, targets, **kwargs):

        """Forward pass.
        Args:
            inputs: Dictionary of classification scores and bounding box
            refinements for the sampled proposals. For cls scores, the shape is
            (b*n) * (cats + 1), where n is sampled proposal in each image, cats
            is the total number of categories without the background. For bbox
            preds, the shape is (b*n) * (4*cats)
            targets: Same output by bbox_head from the teacher output.
        Returns:
            The YOLOX consistency loss.
        """
        teacher_obj = targets["objectness"]
        teacher_reg = targets["bbox_pred"]
        teacher_cls = targets["cls_score"]

        student_obj = inputs["objectness"]
        student_reg = inputs["bbox_pred"]
        student_cls = inputs["cls_score"]

        obj_elements = 0
        reg_elements = 0
        cls_elements = 0
        obj_loss = 0.
        reg_loss = 0.
        cls_loss = 0.
        for (t_obj, t_reg, t_cls, s_obj, s_reg, s_cls) in zip(
            teacher_obj, teacher_reg, teacher_cls,
            student_obj, student_reg, student_cls,
        ):
            assert s_obj.shape == t_obj.shape
            assert s_reg.shape == t_reg.shape
            assert s_cls.shape == t_cls.shape

            _obj_loss = mse_loss(t_obj, s_obj, reduction="none")
            _cls_loss = mse_loss(t_cls, s_cls, reduction="none")
            _reg_loss = mse_loss(t_reg, s_reg, reduction="none")

            obj_loss += torch.sum(_obj_loss)
            reg_loss += torch.sum(_reg_loss)
            cls_loss += torch.sum(_cls_loss)

            obj_elements += _obj_loss.numel()
            reg_elements += _reg_loss.numel()
            cls_elements += _cls_loss.numel()

        obj_loss = obj_loss / obj_elements
        reg_loss = reg_loss / reg_elements
        cls_loss = cls_loss / cls_elements

        loss = self.obj_weight * obj_loss
        loss += self.reg_weight * reg_loss 
        loss += self.cls_weight * cls_loss 

        return self.weight * loss