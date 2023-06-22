# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List

import torch
from copy import deepcopy

from mmengine.structures import InstanceData
from mmtrack.structures import TrackDataSample


class BaseAdapter(metaclass=ABCMeta):
    """Base adapter model.

    Args:
        episodic (bool, optional). If episodic is True, the model will be reset
            to its initial state at the end of every evaluated sequence.
            Defaults to True.
    """

    def __init__(self,
                 episodic: bool = True) -> None:
        super().__init__()
        self.episodic = episodic
        self.fp16_enabled = False

        self.source_model_state = None

    def _init_source_model_state(self, model) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        self.source_model_state = deepcopy(model.state_dict())

    def _restore_source_model_state(self, model) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """

        if self.source_model_state is None:
            raise Exception("cannot reset without saved model state")
        model.load_state_dict(self.source_model_state, strict=True)

    def reset(self, model) -> None:
        """Reset the model state to self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        if self.source_model_state is None:
            self._init_source_model_state(model)
        else:
            self._restore_source_model_state(model)

    @property
    def with_episodic(self) -> bool:
        """Whether the model has to be reset at the end of every sequence."""
        return True if self.episodic else False

    @abstractmethod
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