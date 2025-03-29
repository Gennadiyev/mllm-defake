import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class BasicClassifier(ABC):
    def __init__(self, real_samples: list[Path], fake_samples: list[Path]) -> None:
        if not all(isinstance(x, Path) for x in real_samples):
            raise ValueError(f"Invalid real_samples. Expected a list of Path objects, but got: {real_samples}")
        if not all(isinstance(x, Path) for x in fake_samples):
            raise ValueError(f"Invalid fake_samples. Expected a list of Path objects, but got: {fake_samples}")
        self.real_samples = real_samples
        self.fake_samples = fake_samples

    @abstractmethod
    def classify(self, sample: Path, label: int | bool) -> int:
        raise NotImplementedError("Subclasses must implement the classify method.")

    def _update_metrics(self, y_true, y_pred, pbar):
        """Helper method to update progress bar with current metrics"""
        if y_pred:
            acc = accuracy_score(y_true, y_pred) * 100
            real_acc = (
                len([1 for i, y in enumerate(y_true) if y == 1 and y_pred[i] == 1]) / max(1, y_true.count(1)) * 100
            )
            fake_acc = (
                len([1 for i, y in enumerate(y_true) if y == 0 and y_pred[i] == 0]) / max(1, y_true.count(0)) * 100
            )
            pbar.set_postfix(
                all=f"{acc:.2f}%",
                reals=f"{real_acc:.2f}%",
                fakes=f"{fake_acc:.2f}%",
            )

    def evaluate(self, output_path: Path, continue_from: pd.DataFrame = None) -> tuple[float, float, float]:
        """
        Evaluate the classifier on the provided samples

        Args:
            output_path (Path): Path to save evaluation results
            continue_from (pd.DataFrame, optional): Previous evaluation results to continue from

        Returns:
            tuple[float, float, float]: Accuracy, precision, and recall scores
        """
        # Combine real and fake samples with their labels
        self.samples = [(s, 1) for s in self.real_samples] + [(s, 0) for s in self.fake_samples]

        if continue_from is not None:
            df = continue_from
            processed_samples = set(df["path"].apply(Path))
            self.samples = [(sample, label) for sample, label in self.samples if sample not in processed_samples]
            write_mode = "a"
        else:
            df = pd.DataFrame(columns=["path", "label", "pred"])
            write_mode = "w"

        y_true = []
        y_pred = []

        pbar = tqdm(enumerate(self.samples), total=len(self.samples), desc="Evaluating...")
        for i, (sample, label) in pbar:
            pbar.set_description(f"Eval {sample.name[:19]}")

            try:
                # Use the classify method to get prediction
                pred = self.classify(sample, label)
                y_true.append(label)
                y_pred.append(pred)

                new_row = pd.DataFrame(
                    {
                        "path": [sample],
                        "label": [label],
                        "pred": [pred],
                    }
                )

                if i == 0 and write_mode == "w":
                    new_row.to_csv(output_path, index=False, mode="w")
                else:
                    new_row.to_csv(output_path, index=False, mode="a", header=False)

                # Calculate and update metrics
                self._update_metrics(y_true, y_pred, pbar)

            except Exception as e:
                print(f"Error processing sample {sample.name}: {e}")
                continue

        if not y_pred:
            print("No valid predictions were made during evaluation")
            return 0.0, 0.0, 0.0

        # Calculate final metrics
        accuracy = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p) / len(y_true)
        precision = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1) / (
            sum(1 for p in y_pred if p == 1) + 1e-10
        )
        recall = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1) / (
            sum(1 for t in y_true if t == 1) + 1e-10
        )

        return accuracy, precision, recall


class CannyClassifier(BasicClassifier):
    def canny(self, sample: Path) -> float:
        img = cv2.imread(str(sample), 0)
        edges = cv2.Canny(img, threshold1=100, threshold2=200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return edge_density

    def classify(self, sample: Path, label: int | bool) -> int:
        edge_density = self.canny(sample)
        return 1 if edge_density > 0.0672 else 0


class ComForClassifier(BasicClassifier):
    def __init__(
        self,
        community_forensics_checkpoint_path: Path,
        real_samples: list[Path],
        fake_samples: list[Path],
        input_size: Literal["224", "384"] = "384",
        device: str | torch.device = "cuda:0",
    ) -> None:
        """
        This classifier is a wrapper for the Community-Forensics project (https://jespark.net/projects/2024/community_forensics/), which trains a relatively lightweight ViT model to classify images as real or fake with high accuracy and good OOD performance.

        To use this classifier, follow these steps:

        1. Follow the instructions in the Community-Forensics repository (https://github.com/JeongsooP/Community-Forensics) to download the checkpoint, and specify the corresponding path in `community_forensics_checkpoint_path` parameter.

        2. Specify the `input_size` parameter to either "224" or "384" depending on the checkpoint you downloaded. Note that the community_forensics_checkpoint_path and timm_vit_checkpoint_path should be in the same resolution.
        """
        self.check_dependencies()
        import torch.nn as nn
        import timm
        from PIL import Image
        from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, ConvertImageDtype, Compose

        super().__init__(real_samples, fake_samples)

        class _ComForModel(nn.Module):
            def __init__(self, input_size: int, device: str | torch.device = "cuda:0"):
                super().__init__()
                self.input_size = input_size
                self.device = device  # Specify device to move input tensor to the correct device before forward pass
                if input_size == 224:
                    self.vit = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
                elif input_size == 384:
                    self.vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
                self.vit.head = nn.Linear(
                    in_features=384, out_features=1, bias=True, device=device, dtype=torch.float32
                )

            def preprocess_input(self, x: Image.Image):
                norm_mean = [0.48145466, 0.4578275, 0.40821073]
                norm_std = [0.26862954, 0.26130258, 0.27577711]
                augment_list = []
                resize_size = 440
                crop_size = 384
                if self.input_size == 224:
                    resize_size = 256
                    crop_size = 224
                augment_list.extend(
                    [
                        Resize(resize_size),
                        CenterCrop(crop_size),
                        ToTensor(),
                        Normalize(mean=norm_mean, std=norm_std),
                        ConvertImageDtype(torch.float32),
                    ]
                )
                preprocess = Compose(augment_list)
                x = preprocess(x)
                x = x.unsqueeze(0)
                return x

            def forward(self, x):
                x = self.preprocess_input(x).to(self.device)
                x = self.vit(x)
                x = torch.nn.functional.sigmoid(x)
                return x

        if input_size == "224":
            self.model = _ComForModel(224, device=device)
        elif input_size == "384":
            self.model = _ComForModel(384, device=device)
        else:
            raise ValueError(f"Unsupported input size: {input_size}")

        comfor_checkpoint = torch.load(community_forensics_checkpoint_path, map_location=device)
        self.model.load_state_dict(comfor_checkpoint["model"], strict=True)
        self.model.to(device)
        self.model.eval()

    def check_dependencies(self) -> None:
        if any(importlib.util.find_spec(module) is None for module in ["torch", "torchvision", "timm"]):
            raise ImportError("Missing dependencies, please run 'pip install -e .[comfor]'")

    def com_for(self, sample: Path) -> float:
        img = Image.open(sample).convert("RGB")
        fake_prob = self.model(img).item()
        return 1 - fake_prob

    def classify(self, sample: Path, label: int | bool) -> int:
        high_freq_content = self.high_freq_content(sample)
        return 1 if high_freq_content > 8.5 else 0


class ComForClassifier(BasicClassifier):
    def __init__(
        self,
        real_samples: list[Path],
        fake_samples: list[Path],
        community_forensics_checkpoint_path: Path,
        input_size: Literal["224", "384"] = "384",
        device: str | torch.device = "cuda:0",
    ) -> None:
        """
        This classifier is a wrapper for the Community-Forensics project (https://jespark.net/projects/2024/community_forensics/), which trains a relatively lightweight ViT model to classify images as real or fake with high accuracy and good OOD performance.

        To use this classifier, follow these steps:

        1. Follow the instructions in the Community-Forensics repository (https://github.com/JeongsooP/Community-Forensics) to download the checkpoint, and specify the corresponding path in `community_forensics_checkpoint_path` parameter.

        2. Specify the `input_size` parameter to either "224" or "384" depending on the checkpoint you downloaded. Note that the community_forensics_checkpoint_path and timm_vit_checkpoint_path should be in the same resolution.
        """
        self.check_dependencies()
        import torch.nn as nn
        import timm
        from PIL import Image
        from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, ConvertImageDtype, Compose

        super().__init__(real_samples, fake_samples)

        class _ComForModel(nn.Module):
            def __init__(self, input_size: int, device: str | torch.device = "cuda:0"):
                super().__init__()
                self.input_size = input_size
                self.device = device  # Specify device to move input tensor to the correct device before forward pass
                if input_size == 224:
                    self.vit = timm.create_model(
                        "vit_small_patch16_224.augreg_in21k_ft_in1k",
                        pretrained=False
                    )
                elif input_size == 384:
                    self.vit = timm.create_model(
                        "vit_small_patch16_384.augreg_in21k_ft_in1k",
                        pretrained=False
                    )
                self.vit.head = nn.Linear(
                    in_features=384, out_features=1, bias=True, device=device, dtype=torch.float32
                )

            def preprocess_input(self, x: Image.Image):
                norm_mean = [0.48145466, 0.4578275, 0.40821073]
                norm_std = [0.26862954, 0.26130258, 0.27577711]
                augment_list = []
                resize_size = 440
                crop_size = 384
                if self.input_size == 224:
                    resize_size = 256
                    crop_size = 224
                augment_list.extend(
                    [
                        Resize(resize_size),
                        CenterCrop(crop_size),
                        ToTensor(),
                        Normalize(mean=norm_mean, std=norm_std),
                        ConvertImageDtype(torch.float32),
                    ]
                )
                preprocess = Compose(augment_list)
                x = preprocess(x)
                x = x.unsqueeze(0)
                return x

            def forward(self, x):
                x = self.preprocess_input(x).to(self.device)
                x = self.vit(x)
                x = torch.nn.functional.sigmoid(x)
                return x

        if input_size == "224":
            self.model = _ComForModel(224, device=device)
        elif input_size == "384":
            self.model = _ComForModel(384, device=device)
        else:
            raise ValueError(f"Unsupported input size: {input_size}")

        comfor_checkpoint = torch.load(community_forensics_checkpoint_path, map_location=device)
        self.model.load_state_dict(comfor_checkpoint["model"], strict=True)
        self.model.to(device)
        self.model.eval()

    def check_dependencies(self) -> None:
        if any(importlib.util.find_spec(module) is None for module in ["torch", "torchvision", "timm"]):
            raise ImportError("Missing dependencies, please run 'pip install -e .[comfor]'")

    def com_for(self, sample: Path) -> float:
        img = Image.open(sample).convert("RGB")
        fake_prob = self.model(img).item()
        return 1 - fake_prob

    def classify(self, sample: Path, label: int | bool) -> int:
        com_for_value = self.com_for(sample)
        return 1 if com_for_value > 0.5 else 0
