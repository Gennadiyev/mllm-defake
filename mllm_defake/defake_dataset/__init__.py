from .images_only import ImageFolders, RealFakeDataset, WildFakeResampled
from .images_with_bbox_and_reason import BboxReasonDataset

__all__ = ["RealFakeDataset", "ImageFolders", "WildFakeResampled", "BboxReasonDataset"]

SUPPORTED_DATASETS = ["WildFakeResampled", "ImageFolders", "WildFakeResampled20K", ""]
