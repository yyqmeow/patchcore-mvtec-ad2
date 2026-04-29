"""PatchCore and PatchCore detection methods."""
import json
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        use_tiling=False,
        tile_size=224,
        tile_stride=None,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        
        # Tiling parameters
        self.use_tiling = use_tiling
        self.tile_size = tile_size
        self.tile_stride = tile_stride if tile_stride is not None else tile_size // 2

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    # Store labels and masks properly
                    batch_is_anomaly = image["is_anomaly"]
                    batch_mask = image["mask"]
                    
                    # Convert to numpy if tensor
                    if isinstance(batch_is_anomaly, torch.Tensor):
                        labels_gt.extend(batch_is_anomaly.cpu().numpy().tolist())
                    else:
                        labels_gt.extend(batch_is_anomaly.tolist() if hasattr(batch_is_anomaly, 'tolist') else batch_is_anomaly)
                    
                    # Store masks as numpy arrays (not lists)
                    if isinstance(batch_mask, torch.Tensor):
                        # Keep as tensor or convert to numpy, but preserve shape
                        for i in range(batch_mask.shape[0]):
                            mask_item = batch_mask[i].cpu().numpy()
                            masks_gt.append(mask_item)
                    else:
                        # Already numpy or list
                        if isinstance(batch_mask, np.ndarray):
                            for i in range(batch_mask.shape[0]):
                                masks_gt.append(batch_mask[i])
                        else:
                            # List of masks
                            masks_gt.extend(batch_mask)
                    
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        # Add debugging information
        LOGGER.debug(f"_predict called with images shape: {images.shape}")
        LOGGER.debug(f"Tiling enabled: {self.use_tiling}")
        if self.use_tiling:
            LOGGER.debug(f"Tile size: {self.tile_size}, Image size: {images.shape[2:]}")
            LOGGER.debug(f"Condition check: {images.shape[2] > self.tile_size and images.shape[3] > self.tile_size}")
        
        if self.use_tiling and images.shape[2] > self.tile_size and images.shape[3] > self.tile_size:
            LOGGER.info("Using tiling prediction")
            return self._predict_with_tiling(images)
        else:
            if self.use_tiling:
                LOGGER.info("Tiling enabled but image size too small, using regular prediction")
            else:
                LOGGER.debug("Using regular prediction")
        
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]
    
    def _predict_with_tiling(self, images):
        """Predict with tiling strategy for large images.
        
        Key design principles:
        1. Process each tile exactly like a full image in non-tiling mode
        2. Merge patch scores (not image scores) to maintain consistency
        3. Compute image-level score from merged patch scores (same as non-tiling)
        4. Merge masks using weighted average for smooth boundaries
        """
        LOGGER.info(f"_predict_with_tiling called with images shape: {images.shape}")
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()
        
        batchsize = images.shape[0]
        original_height, original_width = images.shape[2], images.shape[3]
        LOGGER.info(f"Processing {batchsize} images of size {original_height}x{original_width}")
        LOGGER.info(f"Tile size: {self.tile_size}, Tile stride: {self.tile_stride}")
        
        all_scores = []
        all_masks = []
        
        for batch_idx in range(batchsize):
            image = images[batch_idx:batch_idx+1]
            LOGGER.debug(f"Processing image {batch_idx} with shape: {image.shape}")
            
            # Create tiles
            tiles, tile_positions = self._create_tiles(
                image, self.tile_size, self.tile_stride
            )
            LOGGER.info(f"Created {len(tiles)} tiles for image {batch_idx}")
            
            # Collect all patch scores from all tiles (1D arrays)
            # Then merge and compute score exactly like non-tiling version
            all_patch_scores_1d = []  # List of 1D patch score arrays from all tiles
            all_tile_masks = []  # List of (mask, position) for merging
            
            # Process each tile exactly like non-tiling version
            for tile_idx, tile in enumerate(tiles):
                LOGGER.debug(f"Processing tile {tile_idx} with shape: {tile.shape}")
                with torch.no_grad():
                    features, patch_shapes = self._embed(tile, provide_patch_shapes=True)
                    features = np.asarray(features)
                    
                    # Get patch-level scores (1D array) - EXACTLY like non-tiling
                    patch_scores_1d = self.anomaly_scorer.predict([features])[0]
                    all_patch_scores_1d.append(patch_scores_1d)
                    
                    # Get tile mask for segmentation (same as non-tiling)
                    patch_scores_unpatched = self.patch_maker.unpatch_scores(
                        patch_scores_1d, batchsize=1
                    )
                    scales = patch_shapes[0]
                    patch_scores_2d = patch_scores_unpatched.reshape(1, scales[0], scales[1])
                    tile_mask = self.anomaly_segmentor.convert_to_segmentation(patch_scores_2d)[0]
                    all_tile_masks.append((tile_mask, tile_positions[tile_idx]))
            
            # Concatenate all patch scores from all tiles into one 1D array
            # This is the simplest and most reliable way to ensure consistency
            if len(all_patch_scores_1d) > 0:
                all_patches_combined = np.concatenate(all_patch_scores_1d, axis=0)
            else:
                all_patches_combined = np.array([0.0], dtype=np.float32)
            
            # Compute image-level score EXACTLY like non-tiling version
            # Non-tiling: unpatch -> reshape(*shape[:2], -1) -> score
            batchsize = 1
            image_scores = self.patch_maker.unpatch_scores(all_patches_combined, batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            image_score = float(image_scores[0])
            
            # Merge masks using weighted average for smooth boundaries
            merged_mask = self._merge_tiles_weighted(
                all_tile_masks, original_height, original_width
            )
            
            LOGGER.info(f"Image {batch_idx} final score: {image_score}")
            all_scores.append(image_score)
            all_masks.append(merged_mask)
        
        return all_scores, all_masks
    
    def _create_tiles(self, image, tile_size, tile_stride):
        """Create tiles from an image."""
        _, _, h, w = image.shape
        tiles = []
        positions = []
        
        # Calculate number of tiles
        num_tiles_h = (h - tile_size) // tile_stride + 1
        num_tiles_w = (w - tile_size) // tile_stride + 1
        
        # Adjust for cases where the last tile doesn't align perfectly
        if (h - tile_size) % tile_stride != 0:
            num_tiles_h += 1
        if (w - tile_size) % tile_stride != 0:
            num_tiles_w += 1
        
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                y_start = i * tile_stride
                x_start = j * tile_stride
                
                # Ensure we don't go out of bounds
                y_end = min(y_start + tile_size, h)
                x_end = min(x_start + tile_size, w)
                
                # Adjust start positions if we're at the edge
                if y_end - y_start < tile_size:
                    y_start = max(0, h - tile_size)
                if x_end - x_start < tile_size:
                    x_start = max(0, w - tile_size)
                
                y_end = y_start + tile_size
                x_end = x_start + tile_size
                
                tile = image[:, :, y_start:y_end, x_start:x_end]
                
                # No need to resize since we ensure correct size above
                tiles.append(tile)
                positions.append((y_start, x_start, y_end, x_end))
        
        return tiles, positions
    
    def _merge_patch_scores_at_patch_resolution(self, tile_patch_scores_list, target_h, target_w):
        """Merge patch scores from all tiles at patch resolution (NOT upsampled).
        
        This is critical: we compute image-level score at patch resolution,
        exactly like non-tiling version does.
        """
        if len(tile_patch_scores_list) == 0:
            return np.zeros((1, 1), dtype=np.float32)
        
        first_patch_scores, _ = tile_patch_scores_list[0]
        tile_patch_h, tile_patch_w = first_patch_scores.shape
        
        # Calculate full image patch resolution
        # If a 224x224 tile has patch_h x patch_w patches,
        # then target_h x target_w image should have:
        scale_h = target_h / self.tile_size
        scale_w = target_w / self.tile_size
        full_patch_h = max(1, int(tile_patch_h * scale_h))
        full_patch_w = max(1, int(tile_patch_w * scale_w))
        
        # Initialize merged patch score map at patch resolution
        merged_patch_scores = np.zeros((full_patch_h, full_patch_w), dtype=np.float32)
        
        # Merge all tile patch scores
        for patch_scores, (y_start, x_start, y_end, x_end) in tile_patch_scores_list:
            # Calculate patch coordinates in full image patch map
            patch_y_start = int((y_start / target_h) * full_patch_h)
            patch_x_start = int((x_start / target_w) * full_patch_w)
            patch_y_end = min(patch_y_start + patch_scores.shape[0], full_patch_h)
            patch_x_end = min(patch_x_start + patch_scores.shape[1], full_patch_w)
            
            if patch_y_end > patch_y_start and patch_x_end > patch_x_start:
                # Extract the region we need
                patch_h_actual = patch_y_end - patch_y_start
                patch_w_actual = patch_x_end - patch_x_start
                tile_patch_region = patch_scores[:patch_h_actual, :patch_w_actual]
                
                # Use maximum for overlapping regions (preserve high anomaly scores)
                existing = merged_patch_scores[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
                merged_patch_scores[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = np.maximum(
                    existing, tile_patch_region
                )
        
        return merged_patch_scores
    
    def _merge_tiles_weighted(self, tile_masks_list, target_h, target_w):
        """Merge tile masks using weighted average for smooth boundaries."""
        merged_mask = np.zeros((target_h, target_w), dtype=np.float32)
        weight_sum = np.zeros((target_h, target_w), dtype=np.float32)
        
        for tile_mask, (y_start, x_start, y_end, x_end) in tile_masks_list:
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            
            # Resize if needed
            if tile_mask.shape != (tile_h, tile_w):
                tile_mask_tensor = torch.from_numpy(tile_mask).unsqueeze(0).unsqueeze(0).float()
                tile_mask_tensor = F.interpolate(
                    tile_mask_tensor, size=(tile_h, tile_w),
                    mode='bilinear', align_corners=False
                )
                tile_mask = tile_mask_tensor.squeeze().cpu().numpy()
            
            # Create Gaussian weight (center has higher weight)
            center_y, center_x = tile_h // 2, tile_w // 2
            y_coords, x_coords = np.ogrid[:tile_h, :tile_w]
            sigma = min(tile_h, tile_w) / 4.0
            weights = np.exp(-((y_coords - center_y)**2 + (x_coords - center_x)**2) / (2 * sigma**2))
            
            # Weighted combination
            merged_mask[y_start:y_end, x_start:x_end] += tile_mask * weights
            weight_sum[y_start:y_end, x_start:x_end] += weights
        
        # Normalize
        valid = weight_sum > 0
        merged_mask = np.where(valid, merged_mask / np.maximum(weight_sum, 1e-8), 0.0)
        
        return merged_mask
    
    def _merge_tiles(self, tile_masks, tile_positions, target_h, target_w):
        """Merge tile masks back to original image size.
        
        Improved merging strategy for anomaly detection:
        1. Use maximum pooling for overlapping regions to preserve high anomaly scores
        2. This is critical for anomaly detection where we want to preserve any detection
        3. For non-overlapping regions, use the tile value directly
        4. This preserves edge sharpness and high-confidence detections better than averaging
        """
        merged_mask = np.zeros((target_h, target_w), dtype=np.float32)
        overlap_count = np.zeros((target_h, target_w), dtype=np.int32)
        
        for tile_mask, (y_start, x_start, y_end, x_end) in zip(tile_masks, tile_positions):
            # Ensure tile_mask has the correct dimensions
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            
            # Resize tile mask if needed
            if tile_mask.shape != (tile_h, tile_w):
                tile_mask_tensor = torch.from_numpy(tile_mask).unsqueeze(0).unsqueeze(0).float()
                tile_mask_tensor = F.interpolate(
                    tile_mask_tensor, size=(tile_h, tile_w),
                    mode='bilinear', align_corners=False
                )
                tile_mask = tile_mask_tensor.squeeze().cpu().numpy()
            
            # For overlapping regions, use maximum to preserve high anomaly scores
            # This is better than averaging because:
            # 1. Anomaly detection benefits from preserving any high-confidence detection
            # 2. Maximum preserves edge sharpness better than averaging
            # 3. Reduces false negatives by keeping the highest score across overlapping tiles
            existing_region = merged_mask[y_start:y_end, x_start:x_end]
            merged_mask[y_start:y_end, x_start:x_end] = np.maximum(existing_region, tile_mask)
            overlap_count[y_start:y_end, x_start:x_end] += 1
        
        # Optional: Apply slight smoothing only to heavily overlapping regions (3+ overlaps)
        # This helps reduce artifacts while preserving edge sharpness
        heavily_overlapping = overlap_count >= 3
        if np.any(heavily_overlapping):
            from scipy import ndimage
            # Apply very light Gaussian smoothing only to heavily overlapping regions
            smoothed = ndimage.gaussian_filter(merged_mask, sigma=0.5)
            merged_mask = np.where(heavily_overlapping, 
                                  0.7 * merged_mask + 0.3 * smoothed,  # Light smoothing
                                  merged_mask)
        
        return merged_mask
    
    def _merge_tile_patch_scores(self, tile_patch_scores_list, target_h, target_w):
        """Merge tile patch scores back to original image size.
        
        This merges raw patch scores (before segmentation conversion) to compute
        image-level scores consistently with the non-tiling version.
        
        Patch scores are at patch resolution (e.g., 28x28), so we need to:
        1. Upsample each tile's patch scores to tile size (e.g., 224x224)
        2. Merge all tiles using maximum
        3. Then compute image-level score from the merged scores
        """
        merged_scores = np.zeros((target_h, target_w), dtype=np.float32)
        
        for patch_scores, (y_start, x_start, y_end, x_end) in tile_patch_scores_list:
            # Patch scores are at patch resolution (e.g., 28x28 for a 224x224 tile)
            # We need to upsample them to tile size first
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            
            # Upsample patch scores to tile size (same as convert_to_segmentation does)
            # but without Gaussian smoothing, to preserve raw scores
            patch_scores_tensor = torch.from_numpy(patch_scores).unsqueeze(0).unsqueeze(0).float()
            patch_scores_tensor = F.interpolate(
                patch_scores_tensor, size=(tile_h, tile_w),
                mode='bilinear', align_corners=False
            )
            upsampled_scores = patch_scores_tensor.squeeze().cpu().numpy()
            
            # Use maximum for overlapping regions (preserve high anomaly scores)
            existing_scores = merged_scores[y_start:y_end, x_start:x_end]
            merged_scores[y_start:y_end, x_start:x_end] = np.maximum(existing_scores, upsampled_scores)
        
        return merged_scores

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            "use_tiling": getattr(self, "use_tiling", False),
            "tile_size": getattr(self, "tile_size", 224),
            "tile_stride": getattr(self, "tile_stride", 112),
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)
        with open(
            os.path.join(save_path, prepend + "patchcore_params.json"), "w"
        ) as save_file:
            json.dump(patchcore_params, save_file, indent=2, sort_keys=True)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        
        # Extract tiling parameters if they exist
        use_tiling = patchcore_params.pop("use_tiling", False)
        tile_size = patchcore_params.pop("tile_size", 224)
        tile_stride = patchcore_params.pop("tile_stride", None)
        
        self.load(**patchcore_params, device=device, nn_method=nn_method)
        
        # Set tiling parameters after load
        self.use_tiling = use_tiling
        self.tile_size = tile_size
        self.tile_stride = tile_stride if tile_stride is not None else tile_size // 2

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
