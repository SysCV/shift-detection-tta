# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
from torch import Tensor

from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList

from shift_tta.registry import MODELS, TASK_UTILS
from .base import BaseAdaptiveDetector


@MODELS.register_module()
class AdaptiveYOLOX(BaseAdaptiveDetector):
    """AdaptiveYOLOX: baseline test-time adaptation method for object detection.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        adapter (dict): Configuration of adapter. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 adapter: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if adapter is not None:
            self.adapter = MODELS.build(adapter)

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        # modify the inputs shape to fit mmdet
        img = inputs['img']
        assert img.size(1) == 1
        # convert 'inputs' shape to (N, C, H, W)
        img = torch.squeeze(img, dim=1)
        return self.detector.loss(img, data_samples, **kwargs)

    def predict(self, inputs: Dict[str, Tensor], data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_det_instances``
            or ``pred_track_instances``.
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'AdaptiveYOLOX inference only support 1 batch size per gpu for now.'
        img = img[0]

        assert len(data_samples) == 1, \
            'AdaptiveYOLOX inference only support 1 batch size per gpu for now.'

        data_sample = data_samples[0]

        det_results = self.detector.predict(img, data_samples)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        data_sample.pred_det_instances = \
            det_results[0].pred_instances.clone()

        if self.adapter is not None:
            pred_track_instances = self.adapter.adapt(
                model=self,
                img=img,
                feats=None,
                data_sample=data_sample,
                **kwargs)
        else:
            pass
            # TODO: build pred_track_instances as detections only
        # our framework builds on mmtrack since it implements video processing
        data_sample.pred_track_instances = pred_track_instances

        return [data_sample]