from shift_tta.registry import DATASETS
from mmtrack.datasets import BaseVideoDataset


@DATASETS.register_module()
class SHIFTDataset(BaseVideoDataset):
    """Dataset class for SHIFT.
    Args:
        visibility_thr (float, optional): The minimum visibility
            for the objects during training. Default to -1.
        detection_file (str, optional): The path of the public
            detection file. Default to None.
    """

    METAINFO = {
        'CLASSES':
        ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
    }