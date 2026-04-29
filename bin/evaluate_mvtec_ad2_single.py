"""Simplified evaluation script for MVTec AD2 dataset with tiling support."""
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
import torch.nn.functional as F

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
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
@click.option("--model_path", type=str, required=True, help="Path to trained model")
@click.option("--results_path", type=str, default="evaluation_results", help="Path to save results")
@click.option("--gpu", type=int, default=None, help="GPU ID (None for CPU)")
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option("--imagesize", type=int, default=224, help="Image size (original, before tiling)")
@click.option("--resize", type=int, default=256, help="Resize size before crop")
@click.option("--use_tiling", is_flag=True, help="Use tiling for large images")
@click.option("--tile_size", type=int, default=224, help="Tile size for tiling")
@click.option("--tile_stride", type=int, default=None, help="Tile stride (default: tile_size//2)")
@click.option("--save_images", is_flag=True, help="Save segmentation images")
def main(
    data_path,
    classname,
    model_path,
    results_path,
    gpu,
    seed,
    imagesize,
    resize,
    use_tiling,
    tile_size,
    tile_stride,
    save_images,
):
    """Evaluate PatchCore on MVTec AD2 dataset for a single class."""
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

    # Load test dataset
    # For tiling, we need to ensure consistency with training preprocessing
    if use_tiling:
        # For tiling, we want to load larger images (not cropped to tile_size)
        # The imagesize parameter should be the target size for tiling (e.g., 512)
        # We'll resize but NOT crop, so images remain large enough for tiling
        LOGGER.info("Loading test dataset with tiling support...")
        # When using tiling, imagesize should be larger than tile_size
        # We resize to imagesize but don't crop, preserving full size for tiling
        if imagesize < tile_size:
            LOGGER.warning(f"imagesize ({imagesize}) is smaller than tile_size ({tile_size}). Setting imagesize to {tile_size * 2}")
            test_imagesize = tile_size * 2
        else:
            test_imagesize = imagesize
        # Resize to test_imagesize but don't crop (set resize = imagesize to skip cropping)
        test_resize = test_imagesize
        LOGGER.info(f"Tiling mode: Using resize={test_resize}, imagesize={test_imagesize} (no crop, full size for tiling)")
    else:
        test_imagesize = imagesize
        test_resize = resize
    
    LOGGER.info("Loading test dataset for class: {}".format(classname))
    test_dataset = mvtec_ad2.MVTecAD2Dataset(
        source=data_path,
        classname=classname,
        resize=test_resize,
        imagesize=test_imagesize,
        split=mvtec_ad2.DatasetSplit.TEST,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for tiling
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    LOGGER.info("Test dataset size: {}".format(len(test_dataset)))

    # Load model
    LOGGER.info("Loading model from {}".format(model_path))
    with device_context:
        nn_method = patchcore.common.FaissNN(False, 4)
        patchcore_model = patchcore.patchcore.PatchCore(device)
        patchcore_model.load_from_path(
            load_path=model_path,
            device=device,
            nn_method=nn_method,
        )
        
        # Enable tiling if specified
        if use_tiling:
            patchcore_model.use_tiling = True
            patchcore_model.tile_size = tile_size
            patchcore_model.tile_stride = tile_stride if tile_stride is not None else tile_size // 2
            LOGGER.info("Tiling enabled: tile_size={}, tile_stride={}".format(
                patchcore_model.tile_size, patchcore_model.tile_stride
            ))

    # Evaluate
    LOGGER.info("Evaluating class: {}...".format(classname))
    with device_context:
        # Add debugging information before prediction
        LOGGER.info(f"Dataset image size: {test_dataset.imagesize}")
        LOGGER.info(f"Model tiling enabled: {patchcore_model.use_tiling}")
        if patchcore_model.use_tiling:
            LOGGER.info(f"Model tile size: {patchcore_model.tile_size}")
            LOGGER.info(f"Model tile stride: {patchcore_model.tile_stride}")
        
        scores, segmentations, labels_gt, masks_gt = patchcore_model.predict(test_dataloader)
    
    # Add debugging information
    LOGGER.info("Prediction completed. Number of predictions: {}".format(len(scores)))
    if use_tiling:
        LOGGER.info("Tiling was enabled. Tile size: {}, Tile stride: {}".format(
            patchcore_model.tile_size, patchcore_model.tile_stride))
    else:
        LOGGER.info("Tiling was disabled.")
    
    # Log some statistics about predictions
    if len(scores) > 0:
        scores_array = np.array(scores)
        LOGGER.info("Prediction score statistics - Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
            scores_array.min(), scores_array.max(), scores_array.mean(), scores_array.std()))

    # Convert masks_gt to numpy and ensure correct shape
    processed_masks_gt = []
    for mask in masks_gt:
        # Handle different mask types
        if isinstance(mask, torch.Tensor):
            # Convert torch tensor to numpy
            mask = mask.squeeze().cpu().numpy()
        elif isinstance(mask, list):
            # Convert list to numpy array
            mask = np.array(mask)
        elif isinstance(mask, np.ndarray):
            # Already numpy array, just ensure it's squeezed
            mask = mask.squeeze()
        else:
            # Try to convert to numpy
            mask = np.array(mask)
        
        # Ensure mask is 2D (H, W)
        if mask.ndim > 2:
            # If 3D or more, take first channel or first slice
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[2] == 1:  # (H, W, 1)
                mask = mask[:, :, 0]
            else:
                # Take first channel
                mask = mask[0] if mask.shape[0] < mask.shape[2] else mask[:, :, 0]
        
        # Ensure 2D
        if mask.ndim != 2:
            LOGGER.warning(f"Mask shape is {mask.shape}, expected 2D. Squeezing...")
            mask = mask.squeeze()
            if mask.ndim != 2:
                raise ValueError(f"Cannot convert mask to 2D, current shape: {mask.shape}")
        
        # Binarize mask (0 or 1)
        mask = (mask > 0.5).astype(np.uint8)
        processed_masks_gt.append(mask)
    masks_gt = processed_masks_gt
    
    # Ensure segmentations and masks have the same size
    processed_segmentations = []
    for i, seg in enumerate(segmentations):
        if seg.shape != masks_gt[i].shape:
            # Resize segmentation to match mask size
            seg_tensor = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()
            target_size = masks_gt[i].shape
            seg_tensor = F.interpolate(
                seg_tensor, size=target_size, mode='bilinear', align_corners=False
            )
            seg = seg_tensor.squeeze().cpu().numpy()
        processed_segmentations.append(seg)
    segmentations = np.array(processed_segmentations)
    
    # Debug: Check score and label distribution
    scores_array = np.array(scores)
    labels_array = np.array(labels_gt)
    
    LOGGER.info("Score statistics: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
        scores_array.min(), scores_array.max(), scores_array.mean(), scores_array.std()
    ))
    LOGGER.info("Label distribution: {} normal, {} anomaly".format(
        np.sum(labels_array == 0), np.sum(labels_array == 1)
    ))
    normal_scores = scores_array[labels_array == 0]
    anomaly_scores = scores_array[labels_array == 1]
    LOGGER.info("Score for normal images: min={:.4f}, max={:.4f}, mean={:.4f}, median={:.4f}".format(
        normal_scores.min() if len(normal_scores) > 0 else 0,
        normal_scores.max() if len(normal_scores) > 0 else 0,
        normal_scores.mean() if len(normal_scores) > 0 else 0,
        np.median(normal_scores) if len(normal_scores) > 0 else 0,
    ))
    LOGGER.info("Score for anomaly images: min={:.4f}, max={:.4f}, mean={:.4f}, median={:.4f}".format(
        anomaly_scores.min() if len(anomaly_scores) > 0 else 0,
        anomaly_scores.max() if len(anomaly_scores) > 0 else 0,
        anomaly_scores.mean() if len(anomaly_scores) > 0 else 0,
        np.median(anomaly_scores) if len(anomaly_scores) > 0 else 0,
    ))
    
    # Check if scores are reversed (normal > anomaly)
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        if normal_scores.mean() > anomaly_scores.mean():
            LOGGER.warning("WARNING: Score distribution is REVERSED! Normal images have higher scores than anomaly images!")
            LOGGER.warning("This suggests a problem with the tiling implementation.")
    
    # Normalize scores (don't normalize - use raw scores for AUROC)
    # AUROC is scale-invariant, so normalization is not necessary
    scores = scores_array
    
    # Normalize segmentations
    if segmentations.size > 0:
        seg_min = segmentations.min()
        seg_max = segmentations.max()
        if seg_max > seg_min:
            segmentations = (segmentations - seg_min) / (seg_max - seg_min + 1e-8)

    # Compute metrics
    LOGGER.info("Computing metrics...")
    
    # Image-level AUROC
    auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
        scores, labels_gt
    )["auroc"]
    
    # Pixel-level AUROC (all images)
    pixel_scores_all = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segmentations, masks_gt
    )
    full_pixel_auroc = pixel_scores_all["auroc"]
    full_seg_f1 = pixel_scores_all.get("optimal_f1", 0.0)  # Segmentation F1 score
    
    # Pixel-level AUROC (anomaly images only)
    sel_idxs = []
    for i in range(len(masks_gt)):
        if np.sum(masks_gt[i]) > 0:
            sel_idxs.append(i)
    
    if len(sel_idxs) > 0:
        pixel_scores_anomaly = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            [segmentations[i] for i in sel_idxs],
            [masks_gt[i] for i in sel_idxs],
        )
        anomaly_pixel_auroc = pixel_scores_anomaly["auroc"]
        anomaly_seg_f1 = pixel_scores_anomaly.get("optimal_f1", 0.0)  # Segmentation F1 score
    else:
        anomaly_pixel_auroc = 0.0
        anomaly_seg_f1 = 0.0
    
    # Compute AUPRO (all images)
    pro_scores_all = patchcore.metrics.compute_pro_score(
        segmentations, masks_gt
    )
    full_aupro = pro_scores_all["pro"]
    
    # Compute AUPRO (anomaly images only)
    if len(sel_idxs) > 0:
        pro_scores_anomaly = patchcore.metrics.compute_pro_score(
            [segmentations[i] for i in sel_idxs],
            [masks_gt[i] for i in sel_idxs],
        )
        anomaly_aupro = pro_scores_anomaly["pro"]
    else:
        anomaly_aupro = 0.0

    # Print results
    LOGGER.info("=" * 50)
    LOGGER.info("Evaluation Results for class: {}".format(classname))
    LOGGER.info("=" * 50)
    LOGGER.info("Instance AUROC: {:.4f}".format(auroc))
    LOGGER.info("Full Pixel AUROC: {:.4f}".format(full_pixel_auroc))
    LOGGER.info("Anomaly Pixel AUROC: {:.4f}".format(anomaly_pixel_auroc))
    LOGGER.info("Full Segmentation F1: {:.4f}".format(full_seg_f1))
    LOGGER.info("Anomaly Segmentation F1: {:.4f}".format(anomaly_seg_f1))
    LOGGER.info("Full AUPRO: {:.4f}".format(full_aupro))
    LOGGER.info("Anomaly AUPRO: {:.4f}".format(anomaly_aupro))
    LOGGER.info("=" * 50)

    # Save results to CSV
    import csv
    results_file = os.path.join(results_path, "results_{}.csv".format(classname))
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Class", classname])
        writer.writerow(["Instance AUROC", "{:.4f}".format(auroc)])
        writer.writerow(["Full Pixel AUROC", "{:.4f}".format(full_pixel_auroc)])
        writer.writerow(["Anomaly Pixel AUROC", "{:.4f}".format(anomaly_pixel_auroc)])
        writer.writerow(["Full Segmentation F1", "{:.4f}".format(full_seg_f1)])
        writer.writerow(["Anomaly Segmentation F1", "{:.4f}".format(anomaly_seg_f1)])
        writer.writerow(["Full AUPRO", "{:.4f}".format(full_aupro)])
        writer.writerow(["Anomaly AUPRO", "{:.4f}".format(anomaly_aupro)])
    
    LOGGER.info("Results saved to {}".format(results_file))

    # Save segmentation images
    if save_images:
        LOGGER.info("Saving segmentation images...")
        image_paths = [
            x[2] for x in test_dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in test_dataset.data_to_iterate
        ]
        
        def image_transform(image):
            in_std = np.array(test_dataset.transform_std).reshape(-1, 1, 1)
            in_mean = np.array(test_dataset.transform_mean).reshape(-1, 1, 1)
            image = test_dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)
        
        def mask_transform(mask):
            return test_dataset.transform_mask(mask).numpy()
        
        image_save_path = os.path.join(results_path, "segmentation_images", classname)
        os.makedirs(image_save_path, exist_ok=True)
        
        # Clean image paths to avoid path issues on Windows
        cleaned_image_paths = []
        for path in image_paths:
            # Normalize path and extract just the filename
            cleaned_path = os.path.normpath(path)
            cleaned_image_paths.append(cleaned_path)
        
        # Fix: Use a simpler save depth to avoid path issues
        try:
            patchcore.utils.plot_segmentation_images(
                image_save_path,
                cleaned_image_paths,
                segmentations,
                scores,
                mask_paths,
                image_transform=image_transform,
                mask_transform=mask_transform,
                save_depth=1,  # Only use filename, not full path
            )
            LOGGER.info("Segmentation images saved to {}".format(image_save_path))
        except Exception as e:
            LOGGER.warning("Failed to save segmentation images: {}. Continuing...".format(e))


if __name__ == "__main__":
    main()

