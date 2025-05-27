from datasets.video import (
    MinecraftVideoDataset
)

from algorithms.worldmem import WorldMemMinecraft
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video_worldmemminecraft=WorldMemMinecraft,
    )

    compatible_datasets = dict(
        # video datasets
        video_minecraft=MinecraftVideoDataset,
    )
