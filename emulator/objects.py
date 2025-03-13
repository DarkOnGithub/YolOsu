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
    
    def get_segmentation_mask(self, width, height, r):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        center_x, center_y = int(round(self.x)), int(round(self.y))
        radius = int(r)
        
        cv2.circle(mask, (center_x, center_y), radius, 1, -1)
        
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
            
    def get_segmentation_mask(self, width, height, r):

        mask = np.zeros((height, width), dtype=np.uint8)
        
        curve_points = []
        num_samples = max(int(self.curve.pxlength / r) * 2, 10)
        step = 1.0 / num_samples
        
        for i in range(num_samples):
            t = i * step
            distance = t * self.curve.pxlength
            point = self.curve.point_at_distance(distance)
            if point:
                curve_points.append((int(round(point[0])), int(round(point[1]))))
        
        end_point = self.curve.point_at_distance(self.curve.pxlength)
        if end_point:
            curve_points.append((int(round(end_point[0])), int(round(end_point[1]))))
        
        radius = int(r)
        for center_x, center_y in curve_points:
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
        
        return mask


class ApproachCircle:
    def __init__(self, center, time, approach_time_ms, radius=10):
        self.center = center
        self.time = time
        self.radius = radius            
        self.appear = time - approach_time_ms
        
    def get_segmentation_mask(self, width, height, time, thickness=1):
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if time < self.appear or time >= self.time:
            return mask
            
        progress = (time - self.appear) / (self.time - self.appear)
        
        scale = 4.0 - 3.0 * progress
        
        current_radius = int(round(self.radius * scale))
        
        center_x = int((self.center[0]))
        center_y = int((self.center[1]))
        print(self.time, time)
        if abs(self.time - time) < 20:
            cv2.circle(mask, (center_x, center_y), current_radius, 1, 5)
        else:
            cv2.circle(mask, (center_x, center_y), current_radius, 1, thickness)
        
        return mask