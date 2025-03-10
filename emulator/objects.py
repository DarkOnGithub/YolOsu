import numpy as np
import math
import cv2
import utils.curves as curve
class HitObject:
    def __init__(self, x, y, time):
        self.x, self.y = x, y 
        self.time = int(time)
    
    def get_segmentation_polygon(self, r):
        """Must be implemented by subclasses"""
        raise NotImplementedError
        
    def get_segmentation_mask(self, width, height, r):
        """Must be implemented by subclasses"""
        raise NotImplementedError


class HitCircle(HitObject):
    def __init__(self, x, y, time):
        super().__init__(x, y, time)
        
    def get_bounding_box(self, r):
        top_left = (self.x - r, self.y - r)
        top_right = (self.x + r, self.y - r)
        bottom_right = (self.x + r, self.y + r)
        bottom_left = (self.x - r, self.y + r)
        
        return (top_left, top_right, bottom_right, bottom_left)
    
    def get_segmentation_polygon(self, r, num_points=32):
        """
        Returns polygon points for a circle to use with YOLO v8 segmentation
        
        Args:
            r (float): Radius of the circle
            num_points (int): Number of points to use for the polygon approximation
            
        Returns:
            list: List of (x,y) tuples representing polygon vertices
        """
        polygon = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = self.x + r * math.cos(angle)
            y = self.y + r * math.sin(angle)
            polygon.append((x, y))
        
        return polygon
    
    def get_segmentation_mask(self, width, height, r):
        """
        Creates a binary mask for the circle
        
        Args:
            width (int): Width of the mask
            height (int): Height of the mask
            r (float): Radius of the circle
            
        Returns:
            np.ndarray: Binary mask where 1 indicates the hitcircle
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Calculate distance from center
        dist = np.sqrt((x_coords - self.x) ** 2 + (y_coords - self.y) ** 2)
        
        # Set mask to 1 inside the circle
        mask[dist <= r] = 1
        
        return mask
          
class Slider(HitObject):
    def __init__(self, x, y, time, curve_type, curve_points, length):
        super().__init__(x, y, time)
        curve_type = curve_type.lower()
        if curve_type == "b":
            self.curve = curve.Bezier(curve_points)
        elif curve_type == "c":
            self.curve = curve.Catmull(curve_points)
        self.curve_points = curve_points
        self.length = length
    
    def get_segmentation_polygon(self, r, num_samples=50):
        """
        Returns polygon points for a slider to use with YOLO v8 segmentation
        
        Args:
            r (float): Radius of the slider path
            num_samples (int): Number of points to sample along the slider
            
        Returns:
            list: List of (x,y) tuples representing polygon vertices
        """
        # Sample points along the actual curve
        path_points = []
        start_point = (self.x, self.y)
        path_points.append(start_point)
        
        # Sample points from the curve
        if hasattr(self, 'curve'):
            for i in range(1, num_samples + 1):
                t = i / num_samples
                # For Bezier curves
                if hasattr(self.curve, 'at'):
                    point = self.curve.at(t)
                    path_points.append((point[0], point[1]))
                # For Catmull curves or other types
                elif hasattr(self.curve, 'point_at_distance'):
                    distance = t * self.length
                    point = self.curve.point_at_distance(distance)
                    if point:
                        path_points.append((point[0], point[1]))
        else:
            # Fallback to control points if curve object is missing
            for point in self.curve_points:
                path_points.append(point)
        
        # Generate left and right boundaries along the path
        left_boundary = []
        right_boundary = []
        
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            
            # Find the normal vector to the path
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                nx = -dy / length
                ny = dx / length
                
                # Add points on both sides of the path
                left_boundary.append((p1[0] + r*nx, p1[1] + r*ny))
                right_boundary.append((p1[0] - r*nx, p1[1] - r*ny))
        
        # Add the last point
        if len(path_points) >= 2:
            p1 = path_points[-2]
            p2 = path_points[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                nx = -dy / length
                ny = dx / length
                left_boundary.append((p2[0] + r*nx, p2[1] + r*ny))
                right_boundary.append((p2[0] - r*nx, p2[1] - r*ny))
        
        # Combine boundaries to form a closed polygon
        polygon = left_boundary + list(reversed(right_boundary))
        
        return polygon
    
    def get_segmentation_mask(self, width, height, r):
        """
        Creates a binary mask for the slider using Bezier curve sampling
        
        Args:
            width (int): Width of the mask
            height (int): Height of the mask
            r (float): Radius of the slider path
            
        Returns:
            np.ndarray: Binary mask where 1 indicates the slider
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Sample points along the Bezier curve
        curve_points = []
        if hasattr(self, 'curve') and hasattr(self.curve, 'at'):
            for t in np.arange(0, 1.01, 0.01):  # Sample 100 points
                point = self.curve.at(t)
                curve_points.append((point[0], point[1]))
        else:
            # Fallback if curve isn't available
            curve_points = [(self.x, self.y)] + self.curve_points
        
        # Define tight bounds to avoid checking every pixel
        min_x = max(0, int(min(p[0] for p in curve_points) - r))
        max_x = min(width, int(max(p[0] for p in curve_points) + r + 1))
        min_y = max(0, int(min(p[1] for p in curve_points) - r))
        max_y = min(height, int(max(p[1] for p in curve_points) + r + 1))
        
        # For each pixel in bounds, check if it's within radius of any curve point
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # Check if this pixel is within radius of any curve point
                for cx, cy in curve_points:
                    if (x - cx)**2 + (y - cy)**2 <= r**2:
                        mask[y, x] = 1
                        break
        
        return mask