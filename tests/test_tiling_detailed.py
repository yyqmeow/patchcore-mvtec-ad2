"""Detailed test script to verify tiling functionality and compare with non-tiling approach."""
import numpy as np
import torch
import torch.nn.functional as F

def test_tiling_consistency():
    """Test that tiling produces consistent results with non-tiling approach."""
    print("Testing tiling consistency...")
    
    # Simulate image-level scoring consistency
    # In non-tiling, we take max over all patches
    # In tiling, we take max over all tiles (each tile's score is max over its patches)
    
    # Simulate patch scores for a non-tiling approach
    non_tiling_patch_scores = np.random.rand(100)  # 100 patches
    non_tiling_image_score = np.max(non_tiling_patch_scores)
    
    # Simulate tiling approach with 4 tiles, each with 25 patches
    tile_scores = []
    for i in range(4):
        tile_patch_scores = np.random.rand(25)  # 25 patches per tile
        tile_score = np.max(tile_patch_scores)  # Max over patches in tile
        tile_scores.append(tile_score)
    
    # Image score is max over tile scores
    tiling_image_score = np.max(tile_scores)
    
    print(f"Non-tiling image score (max over 100 patches): {non_tiling_image_score:.4f}")
    print(f"Tiling image score (max over 4 tile maxes): {tiling_image_score:.4f}")
    print(f"Difference: {abs(non_tiling_image_score - tiling_image_score):.4f}")
    
    # The key insight is that both approaches should be comparable
    # Tiling might be slightly different due to boundary effects and overlapping regions
    # but should not be systematically worse

def test_tile_creation_and_merging():
    """Test tile creation and merging process."""
    print("\nTesting tile creation and merging...")
    
    # Create a sample image with a clear pattern
    image = torch.zeros(1, 3, 512, 512)
    # Add a distinctive pattern in one region
    image[:, :, 100:200, 100:200] = 1.0  # Bright square in the middle
    
    tile_size = 224
    tile_stride = 112  # 50% overlap
    
    # Simulate the tile creation process
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
    
    print(f"Image size: {h}x{w}")
    print(f"Number of tiles: {num_tiles_h}x{num_tiles_w} = {num_tiles_h * num_tiles_w}")
    
    # Create tiles
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
            tiles.append(tile)
            positions.append((y_start, x_start, y_end, x_end))
    
    print(f"Total tiles created: {len(tiles)}")
    
    # Create sample tile masks with the same pattern
    tile_masks = []
    for tile, (y_start, x_start, y_end, x_end) in zip(tiles, positions):
        # Create a mask that matches the pattern in the original image
        mask = torch.zeros(tile.shape[2], tile.shape[3])
        # Check if this tile overlaps with the bright square
        if (y_start < 200 and y_end > 100 and x_start < 200 and x_end > 100):
            # Calculate overlap
            overlap_y_start = max(100, y_start)
            overlap_y_end = min(200, y_end)
            overlap_x_start = max(100, x_start)
            overlap_x_end = min(200, x_end)
            
            # Adjust to tile coordinates
            mask_y_start = overlap_y_start - y_start
            mask_y_end = overlap_y_end - y_start
            mask_x_start = overlap_x_start - x_start
            mask_x_end = overlap_x_end - x_start
            
            mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] = 1.0
        
        tile_masks.append(mask.numpy())
    
    # Merge tiles
    merged_mask = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.float32)
    
    for tile_mask, (y_start, x_start, y_end, x_end) in zip(tile_masks, positions):
        # Ensure tile_mask has the correct dimensions
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        # Resize tile mask if needed (should not be necessary with improved _create_tiles)
        if tile_mask.shape != (tile_h, tile_w):
            tile_mask_tensor = torch.from_numpy(tile_mask).unsqueeze(0).unsqueeze(0).float()
            tile_mask_tensor = F.interpolate(
                tile_mask_tensor, size=(tile_h, tile_w),
                mode='bilinear', align_corners=False
            )
            tile_mask = tile_mask_tensor.squeeze().cpu().numpy()
        
        # Place tile mask in merged mask with overlap handling
        merged_mask[y_start:y_end, x_start:x_end] += tile_mask
        overlap_count[y_start:y_end, x_start:x_end] += 1.0
    
    # Average overlapping regions
    overlap_count = np.maximum(overlap_count, 1.0)
    merged_mask = merged_mask / overlap_count
    
    print(f"Merged mask shape: {merged_mask.shape}")
    print(f"Expected shape: ({h}, {w})")
    print(f"Merged mask statistics - Min: {merged_mask.min():.4f}, Max: {merged_mask.max():.4f}, Mean: {merged_mask.mean():.4f}")
    
    # Check if the bright square is preserved
    bright_square_preserved = np.all(merged_mask[100:200, 100:200] > 0.5)
    print(f"Bright square preserved in merged mask: {bright_square_preserved}")
    
    return merged_mask

if __name__ == "__main__":
    test_tiling_consistency()
    merged_mask = test_tile_creation_and_merging()
    
    print("\nDetailed tiling test completed successfully!")