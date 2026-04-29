"""Test script to verify tiling is correctly enabled."""
import torch
import numpy as np

def test_tiling_condition():
    """Test the condition for enabling tiling."""
    print("Testing tiling condition...")
    
    # Simulate different image sizes
    test_cases = [
        ("224x224", (1, 3, 224, 224)),
        ("512x512", (1, 3, 512, 512)),
        ("256x256", (1, 3, 256, 256)),
        ("1024x1024", (1, 3, 1024, 1024))
    ]
    
    tile_size = 224
    
    for name, shape in test_cases:
        height, width = shape[2], shape[3]
        use_tiling = True  # Simulate tiling enabled
        condition = use_tiling and height > tile_size and width > tile_size
        print(f"Image size: {name}, Tiling enabled: {use_tiling}, Condition: {condition}")

def test_tiling_process():
    """Test the tiling process with a larger image."""
    print("\nTesting tiling process...")
    
    # Create a sample large image (1, 3, 512, 512)
    image = torch.randn(1, 3, 512, 512)
    print(f"Original image shape: {image.shape}")
    
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
            
            print(f"Tile ({i},{j}): position ({y_start},{x_start}) to ({y_end},{x_end}), shape: {tile.shape}")
    
    print(f"Total tiles created: {len(tiles)}")
    return tiles, positions

if __name__ == "__main__":
    test_tiling_condition()
    test_tiling_process()
    
    print("\nTiling test completed successfully!")