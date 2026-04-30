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


def compute_pro_score(anomaly_segmentations, ground_truth_masks, fpr_limit=0.05,
                      num_thresholds=200):
    """
    Standard AU-PRO@k% (Bergmann et al., MVTec convention).

    PRO = average per-region recall across all connected anomalous regions
    (NOT IoU). FPR = fraction of non-anomalous pixels predicted positive.
    AU-PRO@k% = trapezoidal integral of PRO over FPR in [0, k%], normalized
    by k%.

    Args:
        anomaly_segmentations: list/array, shape [N, H, W], anomaly score maps
                               (any range; thresholds are picked from the
                               actual score distribution).
        ground_truth_masks:    list/array, shape [N, H, W], binary GT masks.
        fpr_limit:             FPR cutoff for the integral (default 0.05 = 5%).
        num_thresholds:        number of thresholds to sample.

    Returns:
        dict with:
            'pro':         AU-PRO@k% in [0, 1]
            'pro_curve':   PRO values at each threshold (sorted by FPR)
            'fpr_curve':   matching FPR values
            'fpr_limit':   the limit used
    """
    from scipy import ndimage

    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    if ground_truth_masks.dtype != np.uint8:
        ground_truth_masks = (ground_truth_masks > 0.5).astype(np.uint8)

    # Sample thresholds from the actual score distribution (more efficient
    # than uniform 0..1 when scores are concentrated in a small range).
    flat = anomaly_segmentations.reshape(-1)
    if flat.size > 100_000:
        flat = np.random.default_rng(0).choice(flat, 100_000, replace=False)
    qs = np.linspace(0.0, 1.0, num_thresholds)
    thresholds = np.quantile(flat, qs)
    # Make strictly decreasing for high-to-low sweep, deduplicated.
    thresholds = np.unique(thresholds)[::-1]

    # Pre-compute connected-component labeling for each GT mask.
    gt_components = []
    for i in range(len(ground_truth_masks)):
        labeled, num = ndimage.label(ground_truth_masks[i])
        gt_components.append((labeled, num))

    total_negatives = int((ground_truth_masks == 0).sum())
    if total_negatives == 0:
        return {"pro": 0.0, "pro_curve": [], "fpr_curve": [],
                "fpr_limit": fpr_limit}

    pros, fprs = [], []
    for threshold in thresholds:
        predictions = (anomaly_segmentations >= threshold)

        # PRO: per-region recall, averaged
        recalls = []
        for i in range(len(predictions)):
            labeled_gt, num_regions = gt_components[i]
            if num_regions == 0:
                continue
            pred_i = predictions[i]
            for region_id in range(1, num_regions + 1):
                region = (labeled_gt == region_id)
                region_size = int(region.sum())
                if region_size == 0:
                    continue
                tp = int(np.logical_and(pred_i, region).sum())
                recalls.append(tp / region_size)
        pro = float(np.mean(recalls)) if recalls else 0.0

        # FPR over non-anomalous pixels
        fp = int(np.logical_and(predictions, ground_truth_masks == 0).sum())
        fpr = fp / total_negatives

        pros.append(pro)
        fprs.append(fpr)

    pros = np.asarray(pros)
    fprs = np.asarray(fprs)

    # Sort by FPR ascending; deduplicate for monotonic integration.
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    # Keep points up to fpr_limit, plus a synthesized boundary point at
    # exactly fpr_limit by linear interpolation if we cross it.
    in_range = fprs <= fpr_limit
    fprs_clip = fprs[in_range]
    pros_clip = pros[in_range]

    if fprs_clip.size == 0:
        return {"pro": 0.0, "pro_curve": pros.tolist(),
                "fpr_curve": fprs.tolist(), "fpr_limit": fpr_limit}

    # Add an interpolated end-point at fpr_limit if the curve continues past it.
    above_idx = np.where(fprs > fpr_limit)[0]
    if above_idx.size > 0:
        i_above = above_idx[0]
        f_lo, p_lo = fprs[i_above - 1], pros[i_above - 1]
        f_hi, p_hi = fprs[i_above], pros[i_above]
        if f_hi > f_lo:
            t = (fpr_limit - f_lo) / (f_hi - f_lo)
            p_at_limit = p_lo + t * (p_hi - p_lo)
            fprs_clip = np.concatenate([fprs_clip, [fpr_limit]])
            pros_clip = np.concatenate([pros_clip, [p_at_limit]])

    if fprs_clip.size < 2:
        # Cannot integrate a single point.
        aupro = float(pros_clip[0]) if pros_clip.size else 0.0
    else:
        # numpy 2.x renamed trapz -> trapezoid
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        aupro = float(_trapz(pros_clip, fprs_clip) / fpr_limit)

    return {
        "pro": aupro,
        "pro_curve": pros.tolist(),
        "fpr_curve": fprs.tolist(),
        "fpr_limit": fpr_limit,
    }
