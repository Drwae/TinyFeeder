"""Paired frame / log-Mel data: the AV-FFIA `.npz` layout, a synthetic generator with the same structure, the
leakage-safe split by recording unit, and the PyTorch dataset with augmentation and test-time corruptions."""
from .dataset import AVFeedingDataset, load_npz, save_split, unit_key, video_level_split
from .synthetic import SynthConfig, generate

__all__ = ["AVFeedingDataset", "SynthConfig", "generate", "load_npz", "save_split", "unit_key", "video_level_split"]
