
import warnings
from typing import Optional, Sequence

from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation import CocoMetric
from mmdet.structures.mask import encode_mask_results
from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.fileio import FileClient, dump, load
from mmtrack.evaluation.metrics import CocoVideoMetric

from scalabel.label.typing import Dataset, Frame, Label, Box3D, RLE
from scalabel.label.transforms import bbox_to_box2d, coco_rle_to_rle, polygon_to_poly2ds

from shift_tta.registry import METRICS


@METRICS.register_module()
class SHIFTVideoMetric(CocoVideoMetric):
    """SHIFT evaluation metric.

    Wraps CocoVideoMetric to implement dumping to coco json format so that 
    it is compatible for conversion to the scalabel format.

    Args:
        to_scalabel (bool): Whether to dump results to a Scalabel-style json 
            file. Defaults to True.
    """
    def __init__(self, to_scalabel: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.to_scalabel = to_scalabel

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Note that we only modify ``pred['pred_instances']`` in ``CocoMetric``
        to ``pred['pred_det_instances']`` here.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_det_instances']
            result['img_id'] = data_sample['img_id']
            result['img_name'] = data_sample['img_path'].split('/')[-1]
            result['video_name'] = data_sample['img_path'].split('/')[-2]
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy())
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def results2scalabel(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a Scalabel style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_frames = []
        segm_frames = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            labels = []
            for i, label in enumerate(result['labels']):
                label = Label(
                    id=i,
                    # category=self.cat_ids[label],  # check if this is text cat
                    category=self.dataset_meta['classes'][label],
                    box2d=bbox_to_box2d(self.xyxy2xywh(bboxes[i])),
                    score=float(scores[i]),
                )
                labels.append(label)

            bbox_frame = Frame(
                name=result["img_name"],
                videoName=result["video_name"],
                frameIndex=result["img_name"].split('.')[0].split('_')[0],
                labels=labels,
            )
            bbox_frames.append(bbox_frame)

            if segm_frames is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            labels = []
            for i, label in enumerate(result['labels']):
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()

                label = Label(
                    id=label["id"],
                    category=self.dataset_meta['classes'][label],
                    box2d=bbox_to_box2d(self.xyxy2xywh(bboxes[i])),
                    score=float(mask_scores[i])
                )
                if isinstance(masks[i], list):
                    label.poly2d = polygon_to_poly2ds(masks[i])
                else:
                    label.rle = coco_rle_to_rle(masks[i])
                labels.append(label)
            segm_frame = Frame(
                name=result["img_name"],
                videoName=result["video_name"],
                frameIndex=result["img_name"].split('.')[0].split('_')[0],
                labels=labels,
            )
            segm_frames.append(segm_frame)

        bbox_ds = Dataset(frames=bbox_frames, groups=None, config=None)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        with open(result_files['bbox'], "w") as f:
            f.write(bbox_ds.json(exclude_unset=True))

        if segm_frames is not None:
            segm_ds = Dataset(frames=segm_frames, groups=None, config=None)
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            with open(result_files['segm'], "w") as f:
                f.write(segm_ds.json(exclude_unset=True))

        return result_files

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO / Scalabel style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """

        if self.to_scalabel:
            _ = self.results2scalabel(results, outfile_prefix)

        return super().results2json(results, outfile_prefix)
