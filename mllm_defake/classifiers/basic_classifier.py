from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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
        return 0  # For fake

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


class FreqClassifier(BasicClassifier):
    def high_freq_content(self, sample: Path) -> float:
        img = cv2.imread(str(sample), 0)
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        high_freq_content = np.mean(magnitude_spectrum[magnitude_spectrum > np.median(magnitude_spectrum)])
        return high_freq_content

    def classify(self, sample: Path, label: int | bool) -> int:
        high_freq_content = self.high_freq_content(sample)
        return 1 if high_freq_content > 8.5 else 0
