from abc import ABC, abstractmethod
from pathlib import Path


class RealFakeDataset(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def list_images(self) -> tuple[list[Path], list[Path]]:
        """
        This method returns a list of image file paths in the dataset.

        @return: Two lists of image file paths. The first list contains the paths of real images, and the second list contains the paths of fake images.
        """
        return [], []


class ImageFolders(RealFakeDataset):
    def __init__(self, real_folder: str | Path, fake_folder: str | Path):
        super().__init__()
        self.real_folder = Path(real_folder)
        self.fake_folder = Path(fake_folder)

    def list_images(self) -> tuple[list[Path], list[Path]]:
        real_images = [x for x in self.real_folder.iterdir()]
        fake_images = [x for x in self.fake_folder.iterdir()]
        return real_images, fake_images


class WildFakeResampled(RealFakeDataset):
    def __init__(self, data_root: str | Path = "./WildFakeResampled"):
        super().__init__()
        self.data_root = Path(data_root)
        self.real_images = self.data_root / "real"
        self.fake_images = self.data_root / "fake"

    def list_images(self) -> tuple[list[Path], list[Path]]:
        return [x for x in self.real_images.iterdir()], [
            x for x in self.fake_images.iterdir()
        ]
