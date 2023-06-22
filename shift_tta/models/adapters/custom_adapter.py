# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmengine.structures import InstanceData
from mmtrack.structures import TrackDataSample

from shift_tta.registry import MODELS
from .base_adapter import BaseAdapter


@MODELS.register_module()
class CustomAdapter(BaseAdapter):
    """Custom adapter model.

    This a template class for implementing your own adapter module.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _adapt(self, *args, **kwargs):
        """Adapt the model."""
        pass

    def adapt(self, model: torch.nn.Module, img: torch.Tensor,
              feats: List[torch.Tensor], data_sample: TrackDataSample,
              **kwargs) -> InstanceData:
        """Adapt the model.
        
        
        Args:
            model (nn.Module): detection model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                ByteTrack method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.

        Returns:
            :obj:`InstanceData`: Detection results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_det_instances.bboxes
        labels = data_sample.pred_det_instances.labels
        scores = data_sample.pred_det_instances.scores

        frame_id = metainfo.get('frame_id', -1)
        if self.with_episodic and frame_id == 0:
            self.reset(model)
        
        # adapt model
        self._adapt()  # TODO: implement your own adapt method here

        # update pred_det_instances
        pred_det_instances = InstanceData()
        pred_det_instances.bboxes = bboxes
        pred_det_instances.labels = labels
        pred_det_instances.scores = scores

        return pred_det_instances