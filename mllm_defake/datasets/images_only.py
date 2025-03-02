from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import Dataset


class RealFakeDataset(ABC, Dataset):
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

    def __len__(self) -> int:
        """
        By default, it returns the total number of real and fake images.
        """
        # This default implementation is not optimized and can be overridden by the subclasses.
        real_images, fake_images = self.list_images()
        return len(real_images) + len(fake_images)

    def __getitem__(self, idx):
        """
        By default, it returns the image path and the label (0 for real, 1 for fake) at the given index.

        No image data is loaded in this method. Depending on the implementation, Path objects may be returned in the first element of the tuple.
        """
        # This default implementation is not optimized and can be overridden by the subclasses.
        real_images, fake_images = self.list_images()
        if idx < len(real_images):
            return real_images[idx], 0
        return fake_images[idx - len(real_images)], 1


class ImageFolders(RealFakeDataset):
    def __init__(self, real_folder: str | Path, fake_folder: str | Path):
        """
        @param real_folder: The path to the folder containing real images.
        @param fake_folder: The path to the folder containing fake images.

        Note that every file under the folders will be treated as an image by default.
        """
        super().__init__()
        self.real_folder = Path(real_folder)
        self.fake_folder = Path(fake_folder)

    def list_images(self) -> tuple[list[Path], list[Path]]:
        real_images = [x for x in self.real_folder.iterdir()]
        fake_images = [x for x in self.fake_folder.iterdir()]
        return real_images, fake_images


class WildFakeResampled(RealFakeDataset):
    """
    This dataset class is designed for the WildFakeResampled dataset used by the developers, which contains real and fake images in separate folders:

    ./WildFakeResampled
    ├── fake
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    └── real
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
    """

    def __init__(self, data_root: str | Path = "./WildFakeResampled"):
        super().__init__()
        self.data_root = Path(data_root)
        self.real_images = self.data_root / "real"
        self.fake_images = self.data_root / "fake"

    def list_images(self) -> tuple[list[Path], list[Path]]:
        return [x for x in self.real_images.iterdir()], [x for x in self.fake_images.iterdir()]
