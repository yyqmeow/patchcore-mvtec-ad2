"""Test script to verify tiling functionality."""
import numpy as np
import torch
import torch.nn.functional as F

def test_create_tiles():
    """Test the tile creation function."""
    # Create a sample image (1, 3, 512, 512)
    image = torch.randn(1, 3, 512, 512)
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
            
            print(f"Tile ({i},{j}): position ({y_start},{x_start}) to ({y_end},{x_end}), shape: {tile.shape}")
    
    print(f"Total tiles created: {len(tiles)}")
    return tiles, positions

def test_merge_tiles(tiles, positions, original_h, original_w):
    """Test the tile merging function."""
    # Create sample tile masks (random values for testing)
    tile_masks = []
    for i, (y_start, x_start, y_end, x_end) in enumerate(positions):
        tile_h, tile_w = y_end - y_start, x_end - x_start
        # Create a sample mask with some pattern
        mask = np.random.rand(tile_h, tile_w).astype(np.float32)
        tile_masks.append(mask)
    
    # Merge tiles
    merged_mask = np.zeros((original_h, original_w), dtype=np.float32)
    overlap_count = np.zeros((original_h, original_w), dtype=np.float32)
    
    for tile_mask, (y_start, x_start, y_end, x_end) in zip(tile_masks, positions):
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
        
        # Place tile mask in merged mask with overlap handling
        merged_mask[y_start:y_end, x_start:x_end] += tile_mask
        overlap_count[y_start:y_end, x_start:x_end] += 1.0
    
    # Average overlapping regions
    overlap_count = np.maximum(overlap_count, 1.0)
    merged_mask = merged_mask / overlap_count
    
    print(f"Merged mask shape: {merged_mask.shape}")
    print(f"Expected shape: ({original_h}, {original_w})")
    print(f"Merged mask statistics - Min: {merged_mask.min():.4f}, Max: {merged_mask.max():.4f}, Mean: {merged_mask.mean():.4f}")
    
    return merged_mask

if __name__ == "__main__":
    print("Testing tile creation...")
    tiles, positions = test_create_tiles()
    
    print("\nTesting tile merging...")
    merged_mask = test_merge_tiles(tiles, positions, 512, 512)
    
    print("\nTest completed successfully!")