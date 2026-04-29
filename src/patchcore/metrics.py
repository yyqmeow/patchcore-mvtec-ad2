"""Anomaly metrics."""
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    optimal_f1 = np.max(F1_scores)
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "optimal_f1": optimal_f1,  # Segmentation F1 score
    }


def compute_pro_score(anomaly_segmentations, ground_truth_masks):
    """
    Computes Per-Region Overlap (PRO) score for anomaly segmentations.
    PRO score measures the overlap between predicted and ground truth regions
    across different thresholds.
    
    Standard implementation: Use fixed number of thresholds from 0 to 1,
    compute PRO at each threshold, then integrate.
    
    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks (normalized to [0, 1]).
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    
    Returns:
        dict: Contains 'pro' (AUPRO score), 'pro_curve' (PRO values at different thresholds)
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)
    
    # Normalize segmentations to [0, 1] if not already
    seg_min = anomaly_segmentations.min()
    seg_max = anomaly_segmentations.max()
    if seg_max > seg_min:
        anomaly_segmentations = (anomaly_segmentations - seg_min) / (seg_max - seg_min + 1e-8)
    
    # Convert to binary masks (0 or 1)
    if ground_truth_masks.dtype != np.uint8:
        ground_truth_masks = (ground_truth_masks > 0.5).astype(np.uint8)
    
    # Use fixed number of thresholds from 0 to 1 (standard approach)
    num_thresholds = 200
    thresholds = np.linspace(0, 1, num_thresholds)
    
    pro_values = []
    
    from scipy import ndimage
    
    for threshold in thresholds:
        # Create binary predictions at this threshold
        predictions = (anomaly_segmentations >= threshold).astype(np.uint8)
        
        # Compute per-region overlap
        per_region_overlaps = []
        
        for i in range(len(predictions)):
            pred_mask = predictions[i]
            gt_mask = ground_truth_masks[i]
            
            # Get connected components in ground truth
            labeled_gt, num_regions = ndimage.label(gt_mask)
            
            if num_regions == 0:
                # No ground truth regions, skip this image
                continue
            
            # For each connected component in ground truth
            for region_id in range(1, num_regions + 1):
                gt_region = (labeled_gt == region_id).astype(np.uint8)
                
                # Compute overlap: intersection over union for this region
                intersection = np.logical_and(pred_mask, gt_region).sum()
                union = np.logical_or(pred_mask, gt_region).sum()
                
                if union > 0:
                    overlap = intersection / union
                    per_region_overlaps.append(overlap)
        
        if len(per_region_overlaps) > 0:
            pro_value = np.mean(per_region_overlaps)
            pro_values.append(pro_value)
        else:
            pro_values.append(0.0)
    
    # Compute AUPRO (Area Under PRO curve)
    # Since thresholds are uniformly spaced from 0 to 1, we can simply integrate
    if len(pro_values) > 0:
        # Use trapezoidal rule: area = sum of (y[i] + y[i+1]) / 2 * dx
        # Since dx = 1/(num_thresholds-1) and thresholds go from 0 to 1
        aupro = np.trapz(pro_values, thresholds)
    else:
        aupro = 0.0
    
    return {
        "pro": aupro,
        "pro_curve": pro_values,
        "thresholds": thresholds,
    }
