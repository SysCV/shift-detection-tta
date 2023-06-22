# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch

from copy import deepcopy

from mmengine.dataset import Compose
from mmengine.optim import build_optim_wrapper
from mmengine.structures import InstanceData
from mmtrack.structures import TrackDataSample

from shift_tta.registry import MODELS
from .base_adapter import BaseAdapter


@MODELS.register_module()
class MeanTeacherYOLOXAdapter(BaseAdapter):
    """Mean-teacher YOLOX adapter model.

    Args:
        teacher (dict): Configuration of teacher. Defaults to None.
        optim_wrapper (dict): Configuration of optimizer wrapper. 
            Defaults to None.
        loss (dict): Configuration of loss. Defaults to None.
        pipeline (list(dict)): Configuration of image transforms.
            Defaults to None.
    """

    def __init__(self,
                 teacher: Optional[dict] = None,
                 optim_wrapper: Optional[dict] = None,
                 optim_steps: int = 0,
                 loss: Optional[dict] = dict(
                     type='ROIConsistencyLoss',
                     weight=0.01,
                 ),
                 pipeline: Optional[list[dict]] = None,
                 teacher_pipeline: Optional[list[dict]] = None,
                 student_pipeline: Optional[list[dict]] = None,
                 views: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.teacher = None
        if teacher is not None:
            self.teacher_cfg = teacher
        
        # build optimizer
        self.optim_wrapper = None
        if optim_wrapper is not None:
            self.optim_wrapper_cfg = optim_wrapper
        self.optim_steps = optim_steps

        # build loss
        self.loss = MODELS.build(loss)

        # build image transforms
        self.pipeline = Compose(pipeline)
        self.teacher_pipeline = Compose(teacher_pipeline)
        self.student_pipeline = Compose(student_pipeline)
        self.views = views

        # TODO: implement param_scheduler for optimizer (e.g. lr decay)

    def _init_source_model_state(self, model) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        super()._init_source_model_state(model)

        self.optim_wrapper = build_optim_wrapper(model, self.optim_wrapper_cfg)
        self.optim_wrapper_state = deepcopy(self.optim_wrapper.state_dict())

        if self.teacher_cfg is not None:
            self.teacher_cfg['model'] = model
            self.teacher = MODELS.build(self.teacher_cfg)
            self.teacher_model_state = deepcopy(self.teacher.state_dict())

    def _reset_optimizer(self) -> None:
        """Reset optimizer state.
        
        Args:
            model (nn.Module): detection model."""
        if self.optim_wrapper is not None:
            self.optim_wrapper.load_state_dict(self.optim_wrapper_state)

    def _restore_source_model_state(self, model) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        super()._restore_source_model_state(model)
            
        if self.teacher is not None:
            self.teacher.load_state_dict(
                self.teacher_model_state, strict=True)

    def _detect_forward(self, detector: torch.nn.Module, img: torch.Tensor, 
                 batch_data_samples: TrackDataSample, rescale: bool = True):
        """Detector forward pass."""
        feats = detector.extract_feat(img)
        outs = detector.bbox_head.forward(feats)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        predictions = detector.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        det_results = detector.add_pred_to_datasample(
            batch_data_samples, predictions)

        return det_results, outs

    def _expand_view(self, outs: Tuple[torch.Tensor], views: int = 1):
        """Expand batch size of each element in outs to views."""
        outs = tuple(list(o.repeat_interleave(views, dim=0) for o in out) 
                     for out in outs)
        return outs

    def _adapt(self, model: torch.nn.Module, 
               teacher_img: torch.Tensor, 
               student_imgs: torch.Tensor, 
               teacher_data_samples: List[TrackDataSample],
               student_data_samples: List[TrackDataSample],
               *args, **kwargs) -> InstanceData:
        """Adapt the model."""

        # teacher forward
        teacher_det_results, teacher_outs = self._detect_forward(
            self.teacher.module.detector, teacher_img, teacher_data_samples)
        teacher_outs = self._expand_view(teacher_outs, views=self.views)
        teacher_outs = dict(
            cls_score=teacher_outs[0],
            bbox_pred=teacher_outs[1],
            objectness=teacher_outs[2])

        # student forward
        _, outs = self._detect_forward(
            model.detector, student_imgs, student_data_samples)
        outs = dict(
            cls_score=outs[0],
            bbox_pred=outs[1],
            objectness=outs[2])

        # adapt
        loss = self.loss(outs, teacher_outs)
        loss.backward()
        self.optim_wrapper.step()
        self.optim_wrapper.zero_grad()

        return teacher_det_results

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
        frame_id = metainfo.get('frame_id', -1)
        if self.with_episodic and frame_id == 0:
            self.reset(model)
        
        # adapt model
        # TODO: apply multiple image transforms
        # data_sample = self.transforms(deepcopy(data_sample))
        # TODO: create a batch
        # TODO: compute teacher prediction on clean target image
        # TODO: compute distill loss into augmented batch 
        #   (concat targets to batch size)

        # make teacher and student views
        results = dict(img_path=data_sample.img_path,
                       instances=data_sample.instances,
        )
        results = self.pipeline(results)
        teacher_results = self.teacher_pipeline(results)
        teacher_img = teacher_results['inputs']['img'].to(img)
        teacher_data_samples = [teacher_results['data_samples']]

        student_imgs = []
        student_data_samples = []
        for _ in range(self.views):
            student_results = self.student_pipeline(results)
            student_imgs.append(student_results['inputs']['img'])
            student_data_samples.append(student_results['data_samples'])
        student_imgs = torch.cat(student_imgs).to(img)

        with torch.enable_grad():
            model.requires_grad_(True)
            model.train(True)
            for _ in range(self.optim_steps):
                outs = self._adapt(
                    model, teacher_img, student_imgs,
                    teacher_data_samples, student_data_samples)
                self.teacher.update_parameters(model)

        self._reset_optimizer()

        # update pred_det_instances
        pred_det_instances = outs[0].pred_instances.clone()

        return pred_det_instances