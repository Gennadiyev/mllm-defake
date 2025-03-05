from abc import abstractmethod
from pathlib import Path

from mllm_defake.datasets.images_only import RealFakeDataset


class BboxReasonDataset(RealFakeDataset):
    @abstractmethod
    def get_bbox_reason(self, image_path: Path) -> list[list[tuple[int, int, int, int]], str]:
        """
        This method returns the bounding boxes and reasons for the fake images.

        @param image_path: The path of the image file.
        @return: A tuple containing a list of bounding boxes and a string of reasons.
        """
        return []
