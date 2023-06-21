from typing import List

from mmengine.fileio import FileClient
from mmtrack.datasets import BaseVideoDataset
from mmtrack.datasets.api_wrappers import CocoVID

from shift_tta.registry import DATASETS

from .utils import check_attributes


@DATASETS.register_module()
class SHIFTDataset(BaseVideoDataset):
    """Dataset class for SHIFT.
    Args:
        attributes (Optional[Dict[str, ...]]): a dictionary containing the
            allowed attributes. Dataset samples will be filtered based on
            the allowed attributes. If None, load all samples. Default: None.
    """

    METAINFO = dict(
        classes = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
    )

    def __init__(self,
                 attributes=None,
                 *args,
                 **kwargs):
        self.attributes = attributes
        super().__init__(*args, **kwargs)


    def filter_by_attributes(self):
        """Filter annotations according to filter_cfg.attributes.

        Returns:
            list[int]: Filtered results.
        """
        if self.load_as_video:
            valid_data_indices = self._filter_video_by_attributes()
        else:
            valid_data_indices = self._filter_image_by_attributes()

        return valid_data_indices
    
    def _filter_video_by_attributes(self):
        """Filter video annotations according to filter_cfg.attributes.

        Annotations are filtered based on the attributes of the first
        frame in the video.

        Returns:
            list[int]: Filtered results.
        """
        file_client = FileClient.infer_client(uri=self.ann_file)
        with file_client.get_local_path(self.ann_file) as local_path:
            coco = CocoVID(local_path)

        valid_data_indices = []
        data_id = 0
        vid_ids = coco.get_vid_ids()
        for vid_id in vid_ids:
            img_ids = coco.get_img_ids_from_vid(vid_id)
            if not len(img_ids) > 0:
                continue
            raw_img_info = coco.load_imgs([img_ids[0]])[0]
            if check_attributes(
                raw_img_info['attributes'], self.filter_cfg['attributes']):
                valid_data_indices.extend(range(data_id, len(img_ids)))
            data_id += len(img_ids)

        set_valid_data_indices = set(self.valid_data_indices)
        valid_data_indices = [
            id for id in valid_data_indices if id in set_valid_data_indices
        ]
        return valid_data_indices


    def _filter_image_by_attributes(self):
        """Filter image annotations according to filter_cfg.attributes.

        Returns:
            list[int]: Filtered results.
        """        
        valid_data_indices = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            if check_attributes(
                data_info['attributes'], self.filter_cfg['attributes']
            ):
                valid_data_indices.append(i)

        set_valid_data_indices = set(self.valid_data_indices)
        valid_data_indices = [
            id for id in valid_data_indices if id in set_valid_data_indices
        ]
        return valid_data_indices


    def filter_data(self) -> List[int]:
        """Filter annotations according to filter_cfg.

        Returns:
            list[int]: Filtered results.
        """
        # filter data by attributes (useful for domain filtering)
        if self.filter_cfg is not None:
            self.valid_data_indices = self.filter_by_attributes()

        return super().filter_data()