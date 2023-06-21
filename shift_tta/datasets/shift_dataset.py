from shift_tta.registry import DATASETS
from mmtrack.datasets import BaseVideoDataset


@DATASETS.register_module()
class SHIFTDataset(BaseVideoDataset):
    """Dataset class for SHIFT.
    Args:
        attributes (Optional[Dict[str, ...]]): a dictionary containing the
            allowed attributes. Dataset samples will be filtered based on
            the allowed attributes. If None, load all samples. Default: None.
    """

    METAINFO = {
        'CLASSES':
        ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
    }

    def __init__(self,
                 attributes=None,
                 *args,
                 **kwargs):
        self.attributes = attributes
        super().__init__(*args, **kwargs)
        