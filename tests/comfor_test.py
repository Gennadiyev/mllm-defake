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
    timm_vit_checkpoint_path = Path("local/vit224")
    model_type = "224"
    classifier = ComForClassifier(
        real_samples,
        fake_samples,
        community_forensics_checkpoint_path=community_forensics_checkpoint_path,
        timm_vit_checkpoint_path=timm_vit_checkpoint_path,
        model_type=model_type,
        device=device,
    )
    classifier.evaluate("local/comfor.csv")
