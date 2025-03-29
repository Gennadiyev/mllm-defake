"""
Tests Community-Forensics classifier and its integration with the mllm_defake library.
"""

import os
import pytest
from pathlib import Path


@pytest.mark.skipif(os.getenv("CI") == "true", reason="CI environment does not support checkpoint loading")
def test_on_demo_images():
    from mllm_defake.classifiers.basic_classifier import ComForClassifier

    device = "cuda:0"
    real_samples = Path("demo/real").rglob("*")
    fake_samples = Path("demo/fake").rglob("*")
    community_forensics_checkpoint_path = Path("local/comfor/model_v11_ViT_224_base_ckpt.pt")
    input_size = "224"
    classifier = ComForClassifier(
        community_forensics_checkpoint_path,
        real_samples,
        fake_samples,
        input_size=input_size,
        device=device,
    )
    classifier.evaluate("local/comfor.csv")
