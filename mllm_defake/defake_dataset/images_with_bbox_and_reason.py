from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import Dataset


class BboxReasonDataset(ABC, Dataset):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def list_images(self) -> tuple[list[Path], list[Path]]:
        """
        This method returns a list of image file paths in the dataset.

        @return: Two lists of image file paths. The first list contains the paths of real images, and the second list contains the paths of fake images.
        """
        return [], []

    @abstractmethod
    def get_bbox_reason(self, image_path: Path) -> list[list[tuple[int, int, int, int]], str]:
        """
        This method returns the bounding boxes and reasons for the fake images.

        @param image_path: The path of the image file.
        @return: A tuple containing a list of bounding boxes and a string of reasons.
        """
        return []
