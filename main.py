import os
import cv2
import numpy as np
from parser.osz_parser import parse_osz_file
import utils.curves as curve  # Import the curve module for Slider
import sys

# Fix the curve import in Slider class
sys.modules['curve'] = curve

def generate_masks_from_beatmap(osz_path, output_dir="beatmap_masks", radius=10):
    """
    Generate segmentation masks for all objects in a beatmap
    
    Args:
        osz_path (str): Path to the .osz file
        output_dir (str): Directory to save masks
        radius (int): Radius to use for hit objects
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set canvas dimensions - using standard osu playfield dimensions
    width, height = 192, 144
    
    # Parse the beatmap file
    print(f"Parsing beatmap: {osz_path}")
    beatmap = parse_osz_file(osz_path)
    print(beatmap)
    if not beatmap or not beatmap.difficulties:
        print("No difficulties found in beatmap")
        return
    
    # Process each difficulty
    for diff_name, difficulty in beatmap.difficulties.items():
        print(f"Processing difficulty: {diff_name}")
        
        # Create a subdirectory for this difficulty
        diff_dir = os.path.join(output_dir, diff_name)
        os.makedirs(diff_dir, exist_ok=True)
        
        # Calculate circle size (CS) radius
        cs = difficulty.difficulty.get("cs", 4)
        object_radius = (width/16) * (1-(0.7*(cs-5)/5))
        print(object_radius)
        # Process each hit object
        for i, obj in enumerate(difficulty.hit_objects):
            # Generate mask
            mask = obj.get_segmentation_mask(width, height, object_radius)
            
            # Save mask as image
            object_type = "circle" if isinstance(obj, HitCircle) else "slider"
            mask_filename = os.path.join(diff_dir, f"{object_type}_{i:04d}.png")
            cv2.imwrite(mask_filename, mask * 255)
            
            # Create a visualization
            vis = np.zeros((height, width, 3), dtype=np.uint8)
            vis[mask == 1] = [0, 255, 0]  # Green for the mask
            
            # Additional visualization based on object type
            if object_type == "circle":
                cv2.circle(vis, (int(obj.x), int(obj.y)), int(object_radius), (0, 0, 255), 1)
            else:
                # Draw curve points for slider
                cv2.circle(vis, (int(obj.x), int(obj.y)), 3, (255, 0, 0), -1)
                
                # Draw the polygon outline from segmentation
                polygon = obj.get_segmentation_polygon(object_radius)
                polygon_int = [(int(x), int(y)) for x, y in polygon]
                cv2.polylines(vis, [np.array(polygon_int)], True, (0, 0, 255), 1)
            
            # Save visualization
            vis_filename = os.path.join(diff_dir, f"{object_type}_{i:04d}_vis.png")
            cv2.imwrite(vis_filename, vis)
            
            # Progress update every 20 objects
            if i % 20 == 0:
                print(f"  Processed {i} objects...")
        
        print(f"  Completed {len(difficulty.hit_objects)} objects for difficulty {diff_name}")
    
    print(f"All masks saved to {output_dir} directory")

if __name__ == "__main__":
    from emulator.objects import HitCircle, Slider
    
    # Test on the example map
    generate_masks_from_beatmap("maps/396221 Kurokotei - Galaxy Collapse.osz")