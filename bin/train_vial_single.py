"""Simplified training script for Vial dataset with single class."""
import contextlib
import logging
import os
import sys

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

# Add dataset import
from patchcore.datasets import vial

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--data_path", type=str, required=True, help="Path to vial dataset")
@click.option("--results_path", type=str, default="results", help="Path to save results")
@click.option("--gpu", type=int, default=None, help="GPU ID (None for CPU)")
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option("--backbone", type=str, default="wideresnet50", help="Backbone network")
@click.option("--layers", type=str, multiple=True, default=["layer2", "layer3"], help="Layers to extract features from")
@click.option("--imagesize", type=int, default=224, help="Image size for training")
@click.option("--resize", type=int, default=256, help="Resize size before crop")
@click.option("--sampler_percentage", type=float, default=0.1, help="Coreset sampler percentage")
@click.option("--save_model", is_flag=True, help="Save trained model")
@click.option("--use_tiling", is_flag=True, help="Use tiling for large images")
@click.option("--tile_size", type=int, default=224, help="Tile size for tiling")
@click.option("--tile_stride", type=int, default=None, help="Tile stride (default: tile_size//2)")
def main(
    data_path,
    results_path,
    gpu,
    seed,
    backbone,
    layers,
    imagesize,
    resize,
    sampler_percentage,
    save_model,
    use_tiling,
    tile_size,
    tile_stride,
):
    """Train PatchCore on Vial dataset."""
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))

    # Create results directory
    os.makedirs(results_path, exist_ok=True)

    # Set device - check if CUDA is available
    if torch.cuda.is_available() and gpu is not None:
        device = patchcore.utils.set_torch_device([gpu])
        device_context = (
            torch.cuda.device("cuda:{}".format(device.index))
            if "cuda" in device.type.lower()
            else contextlib.suppress()
        )
        LOGGER.info("Using GPU: {}".format(device))
    else:
        device = torch.device("cpu")
        device_context = contextlib.suppress()
        LOGGER.info("CUDA not available, using CPU")

    # Fix seeds
    patchcore.utils.fix_seeds(seed, device)

    # Load dataset
    LOGGER.info("Loading training dataset...")
    train_dataset = vial.VialDataset(
        source=data_path,
        resize=resize,
        imagesize=imagesize,
        split=vial.DatasetSplit.TRAIN,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize PatchCore
    LOGGER.info("Initializing PatchCore...")
    backbone_model = patchcore.backbones.load(backbone)
    backbone_model.name = backbone

    nn_method = patchcore.common.FaissNN(False, 4)
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
        sampler_percentage, device
    )

    patchcore_model = patchcore.patchcore.PatchCore(device)
    patchcore_model.load(
        backbone=backbone_model,
        layers_to_extract_from=list(layers),
        device=device,
        input_shape=(3, imagesize, imagesize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        anomaly_score_num_nn=1,
        featuresampler=sampler,
        nn_method=nn_method,
        use_tiling=use_tiling,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )

    # Train
    LOGGER.info("Training PatchCore...")
    with device_context:
        patchcore_model.fit(train_dataloader)

    # Save model
    if save_model:
        model_save_path = os.path.join(results_path, "models")
        os.makedirs(model_save_path, exist_ok=True)
        LOGGER.info("Saving model to {}".format(model_save_path))
        patchcore_model.save_to_path(model_save_path)

    LOGGER.info("Training completed!")


if __name__ == "__main__":
    main()

