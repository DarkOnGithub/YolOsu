import numpy as np
import math
import cv2
import utils.curves as curve
from utils.utils import osu_pixels_to_normal_coords

class HitObject:
    def __init__(self, x, y, time, resolution_width=192, resolution_height=144):
        
        self.osu_x, self.osu_y = x, y
        self.time = int(time)
        
        
        self.x, self.y = osu_pixels_to_normal_coords(x, y, resolution_width, resolution_height)
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
    
    def convert_point(self, x, y):
        return osu_pixels_to_normal_coords(x, y, self.resolution_width, self.resolution_height)
    
    def get_segmentation_polygon(self, r):
        raise NotImplementedError
        
    def get_segmentation_mask(self, width, height, r):
        raise NotImplementedError


class HitCircle(HitObject):
    def __init__(self, x, y, time, resolution_width=192, resolution_height=144):
        super().__init__(x, y, time, resolution_width, resolution_height)
        
    def get_bounding_box(self, r):
        top_left = (self.x - r, self.y - r)
        top_right = (self.x + r, self.y - r)
        bottom_right = (self.x + r, self.y + r)
        bottom_left = (self.x - r, self.y + r)
        
        return (top_left, top_right, bottom_right, bottom_left)
    
    def get_segmentation_polygon(self, r, num_points=32):
        polygon = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = self.x + r * math.cos(angle)
            y = self.y + r * math.sin(angle)
            polygon.append((x, y))
        
        return polygon
    
    def get_segmentation_mask(self, width, height, r):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        y_coords, x_coords = np.ogrid[:height, :width]
        
        dist = np.sqrt((x_coords - self.x) ** 2 + (y_coords - self.y) ** 2)
        
        mask[dist <= r] = 1
        
        return mask
          
    
class Slider(HitObject):
    def __init__(self, x, y, time, curve_type, curve_points, length, resolution_width=192, resolution_height=144):
        super().__init__(x, y, time, resolution_width, resolution_height)
        self.curve_type = curve_type.lower()
        
        
        self.curve_points = []
        for point_x, point_y in curve_points:
            screen_x, screen_y = self.convert_point(point_x, point_y)
            self.curve_points.append((screen_x, screen_y))
        
        
        scale_factor = self.resolution_height * 0.8 / 384  
        self.length = int(float(length) * scale_factor)
        
        
        curve_points_with_start = [(self.x, self.y)] + self.curve_points
        
        if self.curve_type == "b":
            self.curve = curve.Bezier(curve_points_with_start)
        elif self.curve_type == "c":
            self.curve = curve.Catmull(curve_points_with_start) 
        elif self.curve_type == "l":
            self.curve = curve.Linear(curve_points_with_start)
        elif self.curve_type == "p":
            self.curve = curve.PassThrough(curve_points_with_start)
        else:
            self.curve = curve.Bezier(curve_points_with_start)
            
    def get_segmentation_polygon(self, r, num_samples=50):

        
        curve_points = []
        step = 1.0 / num_samples
        
        for i in range(num_samples + 1):
            t = i * step
            distance = t * self.curve.pxlength
            point = self.curve.point_at_distance(distance)
            if point:
                curve_points.append(point)
        
        if len(curve_points) < 2:
            return []
        
        
        polygon = []
        
        
        first_circle = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            x = curve_points[0][0] + r * math.cos(angle)
            y = curve_points[0][1] + r * math.sin(angle)
            first_circle.append((x, y))
        
        
        right_side = []
        for i in range(len(curve_points) - 1):
            p1 = curve_points[i]
            p2 = curve_points[i + 1]
            
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                
                perpx = -dy / length
                perpy = dx / length
                
                
                right_side.append((p1[0] + r * perpx, p1[1] + r * perpy))
        
        
        if len(curve_points) >= 2:
            p1 = curve_points[-2]
            p2 = curve_points[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                perpx = -dy / length
                perpy = dx / length
                right_side.append((p2[0] + r * perpx, p2[1] + r * perpy))
        
        
        last_circle = []
        for i in range(16):
            angle = 2 * math.pi * i / 16 + math.pi
            x = curve_points[-1][0] + r * math.cos(angle)
            y = curve_points[-1][1] + r * math.sin(angle)
            last_circle.append((x, y))
        
        
        left_side = []
        for i in range(len(curve_points) - 1, 0, -1):
            p1 = curve_points[i]
            p2 = curve_points[i - 1]
            
            
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                
                perpx = -dy / length
                perpy = dx / length
                
                
                left_side.append((p1[0] + r * perpx, p1[1] + r * perpy))
        
        
        if len(curve_points) >= 2:
            p1 = curve_points[1]
            p2 = curve_points[0]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                perpx = -dy / length
                perpy = dx / length
                left_side.append((p2[0] + r * perpx, p2[1] + r * perpy))
        
        
        
        polygon = right_side + last_circle + left_side + first_circle
        
        return polygon

    def get_segmentation_mask(self, width, height, r):

        mask = np.zeros((height, width), dtype=np.uint8)
        
        
        curve_points = []
        num_samples = max(int(self.curve.pxlength / r) * 2, 50)  
        step = 1.0 / num_samples
        
        for i in range(num_samples + 1):
            t = i * step
            distance = t * self.curve.pxlength
            point = self.curve.point_at_distance(distance)
            if point:
                curve_points.append(point)
        
        
        for point in curve_points:
            center_x, center_y = round(point[0]), round(point[1])
            
            
            x_min = max(0, round(center_x - r))
            x_max = min(width, round(center_x + r + 1))
            y_min = max(0, round(center_y - r))
            y_max = min(height, round(center_y + r + 1))
            
            
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2:
                        mask[y, x] = 1
        
        return mask