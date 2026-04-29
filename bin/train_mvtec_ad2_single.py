"""Simplified training script for MVTec AD2 dataset with single class support."""
import contextlib
import json
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
from patchcore.datasets import mvtec_ad2

LOGGER = logging.getLogger(__name__)

# Available classes in MVTec AD2
AVAILABLE_CLASSES = [
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", 
    "vial", "wallplugs", "walnuts"
]


@click.command()
@click.option("--data_path", type=str, required=True, help="Path to MVTec AD2 dataset folder (e.g., ./mvtec_ad_2)")
@click.option("--classname", type=str, required=True, help=f"Class name: {', '.join(AVAILABLE_CLASSES)}")
@click.option("--results_path", type=str, default="results", help="Path to save results")
@click.option("--gpu", type=int, default=None, help="GPU ID (None for CPU)")
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option("--backbone", type=str, default="wideresnet50", help="Backbone network")
@click.option("--layers", type=str, multiple=True, default=["layer2", "layer3"], help="Layers to extract features from")
@click.option("--imagesize", type=int, default=224, help="Image size for training")
@click.option("--resize", type=int, default=256, help="Resize size before crop")
@click.option("--batch_size", type=int, default=2, show_default=True)
@click.option("--num_workers", type=int, default=4, show_default=True)
@click.option("--sampler_percentage", type=float, default=0.1, help="Coreset sampler percentage")
@click.option(
    "--sampler_type",
    type=click.Choice(["identity", "greedy_coreset", "approx_greedy_coreset"]),
    default="approx_greedy_coreset",
    show_default=True,
)
@click.option("--pretrain_embed_dimension", type=int, default=1024, show_default=True)
@click.option("--target_embed_dimension", type=int, default=1024, show_default=True)
@click.option("--patchsize", type=int, default=3, show_default=True)
@click.option("--patchstride", type=int, default=1, show_default=True)
@click.option("--anomaly_score_num_nn", type=int, default=1, show_default=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=4, show_default=True)
@click.option("--save_model", is_flag=True, help="Save trained model")
@click.option("--use_tiling", is_flag=True, help="Use tiling for large images")
@click.option("--tile_size", type=int, default=224, help="Tile size for tiling")
@click.option("--tile_stride", type=int, default=None, help="Tile stride (default: tile_size//2)")
def main(
    data_path,
    classname,
    results_path,
    gpu,
    seed,
    backbone,
    layers,
    imagesize,
    resize,
    batch_size,
    num_workers,
    sampler_percentage,
    sampler_type,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    patchstride,
    anomaly_score_num_nn,
    faiss_on_gpu,
    faiss_num_workers,
    save_model,
    use_tiling,
    tile_size,
    tile_stride,
):
    """Train PatchCore on MVTec AD2 dataset for a single class."""
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))

    # Validate classname
    if classname not in AVAILABLE_CLASSES:
        LOGGER.error(f"Invalid classname: {classname}. Available classes: {', '.join(AVAILABLE_CLASSES)}")
        sys.exit(1)

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
    LOGGER.info("Loading training dataset for class: {}".format(classname))
    if use_tiling and resize > imagesize:
        LOGGER.info(
            "Tiling enabled: forcing resize={} to avoid crop mismatch with imagesize={}".format(
                imagesize, imagesize
            )
        )
        resize = imagesize

    train_dataset = mvtec_ad2.MVTecAD2Dataset(
        source=data_path,
        classname=classname,
        resize=resize,
        imagesize=imagesize,
        split=mvtec_ad2.DatasetSplit.TRAIN,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    LOGGER.info("Training dataset size: {}".format(len(train_dataset)))

    # Initialize PatchCore
    LOGGER.info("Initializing PatchCore...")
    backbone_model = patchcore.backbones.load(backbone)
    backbone_model.name = backbone

    if sampler_type == "identity":
        sampler = patchcore.sampler.IdentitySampler()
    elif sampler_type == "greedy_coreset":
        sampler = patchcore.sampler.GreedyCoresetSampler(sampler_percentage, device)
    else:
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            sampler_percentage, device
        )

    nn_method = patchcore.common.FaissNN(
        on_gpu=(faiss_on_gpu and device.type == "cuda"),
        num_workers=faiss_num_workers,
    )

    patchcore_model = patchcore.patchcore.PatchCore(device)
    patchcore_model.load(
        backbone=backbone_model,
        layers_to_extract_from=list(layers),
        device=device,
        input_shape=(3, imagesize, imagesize),
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        patchstride=patchstride,
        anomaly_score_num_nn=anomaly_score_num_nn,
        featuresampler=sampler,
        nn_method=nn_method,
        use_tiling=use_tiling,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )

    # Train
    LOGGER.info("Training PatchCore for class: {}...".format(classname))
    with device_context:
        patchcore_model.fit(train_dataloader)

    run_config = {
        "data_path": data_path,
        "classname": classname,
        "backbone": backbone,
        "layers": list(layers),
        "resize": resize,
        "imagesize": imagesize,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "sampler_type": sampler_type,
        "sampler_percentage": sampler_percentage,
        "pretrain_embed_dimension": pretrain_embed_dimension,
        "target_embed_dimension": target_embed_dimension,
        "patchsize": patchsize,
        "patchstride": patchstride,
        "anomaly_score_num_nn": anomaly_score_num_nn,
        "faiss_on_gpu": faiss_on_gpu,
        "faiss_num_workers": faiss_num_workers,
        "use_tiling": use_tiling,
        "tile_size": tile_size,
        "tile_stride": tile_stride if tile_stride is not None else tile_size // 2,
        "seed": seed,
    }
    with open(os.path.join(results_path, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)

    # Save model
    if save_model:
        model_save_path = os.path.join(results_path, "models", classname)
        os.makedirs(model_save_path, exist_ok=True)
        LOGGER.info("Saving model to {}".format(model_save_path))
        patchcore_model.save_to_path(model_save_path)

    LOGGER.info("Training completed for class: {}!".format(classname))


if __name__ == "__main__":
    main()

