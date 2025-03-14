import math
import bisect
from abc import ABC, abstractmethod


class Curve(ABC):
    def __init__(self, points):
        self.points = points
        self.length = 0.0
        self.distances = []
        self._calculate()

    @abstractmethod
    def _calculate(self):
        pass

    @abstractmethod
    def point_at(self, t):
        pass

    def get_length(self):
        return self.length

    def get_points(self):
        return self.points

    @abstractmethod
    def get_end_direction(self):
        pass


class Linear(Curve):
    def _calculate(self):
        if len(self.points) < 2:
            self.length = 0.0
            return

        self.distances = [0.0]
        self.length = 0.0

        for i in range(1, len(self.points)):
            dx = self.points[i][0] - self.points[i - 1][0]
            dy = self.points[i][1] - self.points[i - 1][1]
            segment_length = math.hypot(dx, dy)
            self.length += segment_length
            self.distances.append(self.length)

    def point_at(self, t):
        if len(self.points) < 2:
            return (0, 0) if not self.points else self.points[0]

        t = max(0.0, min(1.0, t))
        target = t * self.length

        idx = bisect.bisect_left(self.distances, target)
        if idx == 0:
            return self.points[0]
        if idx >= len(self.points):
            return self.points[-1]

        p1 = self.points[idx - 1]
        p2 = self.points[idx]
        segment_length = self.distances[idx] - self.distances[idx - 1]

        if segment_length == 0:
            return p1

        t_segment = (target - self.distances[idx - 1]) / segment_length
        return (
            p1[0] + (p2[0] - p1[0]) * t_segment,
            p1[1] + (p2[1] - p1[1]) * t_segment
        )

    def get_end_direction(self):
        if len(self.points) < 2:
            return (0, 0)
        p1 = self.points[-2]
        p2 = self.points[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        return (dx / length, dy / length) if length > 0 else (0, 0)


class Bezier(Curve):
    
    def _de_casteljau(self, t):
        points = [list(p) for p in self.points]
        n = len(points)
        for r in range(1, n):
            for i in range(n - r):
                points[i][0] = (1 - t) * points[i][0] + t * points[i + 1][0]
                points[i][1] = (1 - t) * points[i][1] + t * points[i + 1][1]
        return (points[0][0], points[0][1])

    def _calculate(self):
        if len(self.points) < 2:
            self.length = 0.0
            return

        
        self.samples = []
        self.tangents = []
        self.distances = [0.0]
        self.length = 0.0
        tolerance = 0.1  
        max_depth = 6

        def sample(t0, t1, p0, p1, depth=0):
            if depth > max_depth:
                self._add_sample(p1, t1)
                return

            mid_t = (t0 + t1) * 0.5
            p_mid = self._de_casteljau(mid_t)
            linear_p = (
                (p0[0] + p1[0]) * 0.5,
                (p0[1] + p1[1]) * 0.5
            )
            
            
            dx = p_mid[0] - linear_p[0]
            dy = p_mid[1] - linear_p[1]
            if math.hypot(dx, dy) > tolerance:
                sample(t0, mid_t, p0, p_mid, depth+1)
                sample(mid_t, t1, p_mid, p1, depth+1)
            else:
                self._add_sample(p1, t1)

        
        p_start = self._de_casteljau(0.0)
        p_end = self._de_casteljau(1.0)
        self._add_sample(p_start, 0.0)
        sample(0.0, 1.0, p_start, p_end)
        
        
        self._calculate_tangents()

    def _add_sample(self, point, t):
        if self.samples:
            prev = self.samples[-1]
            distance = math.hypot(point[0]-prev[0], point[1]-prev[1])
            self.length += distance
        self.distances.append(self.length)
        self.samples.append(point)
        self.tangents.append(self._calculate_tangent_at(t))

    def _calculate_tangent_at(self, t):
        n = len(self.points) - 1
        if n < 1:
            return (0, 0)
            
        
        points = [list(p) for p in self.points]
        for r in range(1, n):
            for i in range(n - r):
                points[i][0] = (1 - t) * points[i][0] + t * points[i+1][0]
                points[i][1] = (1 - t) * points[i][1] + t * points[i+1][1]

        dx = n * (points[1][0] - points[0][0])
        dy = n * (points[1][1] - points[0][1])
        length = math.hypot(dx, dy)
        return (dx/length, dy/length) if length > 0 else (0, 0)

    def _calculate_tangents(self):
        for i in range(1, len(self.tangents)-1):
            
            prev_t = self.tangents[i-1]
            next_t = self.tangents[i+1]
            self.tangents[i] = (
                (prev_t[0] + next_t[0]) * 0.5,
                (prev_t[1] + next_t[1]) * 0.5
            )

    def point_at(self, t):
        if len(self.points) < 2:
            return (0, 0) if not self.points else self.points[0]

        t = max(0.0, min(1.0, t))
        target = t * self.length

        
        idx = bisect.bisect_left(self.distances, target)
        if idx == 0:
            return self.samples[0]
        if idx >= len(self.samples):
            return self.samples[-1]

        
        t_prev = (idx-1)/(len(self.samples)-1)
        t_next = idx/(len(self.samples)-1)
        d_prev = self.distances[idx-1]
        d_next = self.distances[idx]
        
        if d_next - d_prev < 1e-6:
            return self.samples[idx-1]

        
        alpha = (target - d_prev)/(d_next - d_prev)
        alpha = max(0.0, min(1.0, alpha))
        
        p0 = self.samples[idx-1]
        p1 = self.samples[idx]
        m0 = self.tangents[idx-1]
        m1 = self.tangents[idx]
        
        
        h00 = 2*alpha**3 - 3*alpha**2 + 1
        h10 = alpha**3 - 2*alpha**2 + alpha
        h01 = -2*alpha**3 + 3*alpha**2
        h11 = alpha**3 - alpha**2
        
        
        x = h00*p0[0] + h10*(d_next - d_prev)*m0[0] + h01*p1[0] + h11*(d_next - d_prev)*m1[0]
        y = h00*p0[1] + h10*(d_next - d_prev)*m0[1] + h01*p1[1] + h11*(d_next - d_prev)*m1[1]
        
        return (x, y)

    def get_end_direction(self):
        if len(self.tangents) < 2:
            return (0, 0)
        return self.tangents[-1]
class Catmull(Curve):
    def __init__(self, points, alpha=0.5):
        self.alpha = alpha
        super().__init__(points)

    def _calculate(self):
        if len(self.points) < 2:
            self.length = 0.0
            return

        self.segments = []
        self.distances = [0.0]
        self.length = 0.0

        for i in range(len(self.points) - 1):
            p0 = self.points[max(i - 1, 0)]
            p1 = self.points[i]
            p2 = self.points[i + 1]
            p3 = self.points[min(i + 2, len(self.points) - 1)]

            t0 = 0.0
            t1 = self._get_t(t0, p0, p1)
            t2 = self._get_t(t1, p1, p2)
            t3 = self._get_t(t2, p2, p3)

            self.segments.append((p0, p1, p2, p3, t0, t1, t2, t3))

        prev = self.point_at(0.0)
        self.distances = [0.0]
        steps = 100
        for i in range(1, steps + 1):
            t = i / steps
            current = self.point_at(t)
            distance = math.hypot(current[0] - prev[0], current[1] - prev[1])
            self.length += distance
            self.distances.append(self.length)
            prev = current

    def _get_t(self, t, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return t + math.pow(math.hypot(dx, dy), self.alpha)

    def point_at(self, t):
        if len(self.points) < 2:
            return (0, 0) if not self.points else self.points[0]

        t_total = max(0.0, min(1.0, t)) * self.length
        idx = bisect.bisect_left(self.distances, t_total)
        if idx == 0:
            return self.points[0]
        if idx >= len(self.distances):
            return self.points[-1]

        t1 = (idx - 1) / len(self.distances)
        t2 = idx / len(self.distances)
        ratio = (t_total - self.distances[idx - 1]) / (self.distances[idx] - self.distances[idx - 1])
        t_segment = t1 + (t2 - t1) * ratio

        return self._catmull_rom(t_segment)

    def _catmull_rom(self, t):
        t = max(0.0, min(1.0, t))
        segment_idx = int(t * len(self.segments))
        if segment_idx >= len(self.segments):
            return self.points[-1]

        p0, p1, p2, p3, t0, t1, t2, t3 = self.segments[segment_idx]
        t_segment = (t - segment_idx / len(self.segments)) * len(self.segments)

        t01 = t1 - t0
        t12 = t2 - t1
        t23 = t3 - t2

        m1 = (
            (p1[0] - p0[0]) * t12 / t01,
            (p1[1] - p0[1]) * t12 / t01
        ) if t01 > 0 else (0, 0)

        m2 = (
            (p2[0] - p1[0]) * t12 / t23,
            (p2[1] - p1[1]) * t12 / t23
        ) if t23 > 0 else (0, 0)

        a = 2 * p1[0] - 2 * p2[0] + m1[0] + m2[0]
        b = -3 * p1[0] + 3 * p2[0] - 2 * m1[0] - m2[0]
        c = m1[0]
        d = p1[0]

        x = a * t_segment ** 3 + b * t_segment ** 2 + c * t_segment + d

        a = 2 * p1[1] - 2 * p2[1] + m1[1] + m2[1]
        b = -3 * p1[1] + 3 * p2[1] - 2 * m1[1] - m2[1]
        c = m1[1]
        d = p1[1]

        y = a * t_segment ** 3 + b * t_segment ** 2 + c * t_segment + d

        return (x, y)

    def get_end_direction(self):
        
        if len(self.samples) < 2:
            return (0, 0)
        p1 = self.samples[-2]
        p2 = self.samples[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        return (dx / length, dy / length) if length > 0 else (0, 0)


class Perfect(Curve):
    def _calculate(self):
        if len(self.points) != 3:
            self.linear = Linear(self.points)
            self.length = self.linear.get_length()
            return

        p1, p2, p3 = self.points
        self.center, self.radius = self._find_circle(p1, p2, p3)
        if self.center is None:
            self.linear = Linear(self.points)
            self.length = self.linear.get_length()
            return

        self.start_angle = math.atan2(p1[1] - self.center[1], p1[0] - self.center[0])
        self.end_angle = math.atan2(p3[1] - self.center[1], p3[0] - self.center[0])
        self.angle_diff = self.end_angle - self.start_angle

        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        
        if cross > 0:
            
            if self.angle_diff < 0:
                self.angle_diff += 2 * math.pi
        else:
            
            if self.angle_diff > 0:
                self.angle_diff -= 2 * math.pi

        self.length = abs(self.angle_diff) * self.radius

    def _find_circle(self, p1, p2, p3):
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if d == 0:
            return None, 0

        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d

        radius = math.hypot(ax - ux, ay - uy)
        return (ux, uy), radius

    def point_at(self, t):
        if len(self.points) != 3 or self.center is None:
            if hasattr(self, 'linear'):
                return self.linear.point_at(t)
            return self.points[0] if self.points else (0, 0)

        angle = self.start_angle + self.angle_diff * t
        return (
            self.center[0] + self.radius * math.cos(angle),
            self.center[1] + self.radius * math.sin(angle)
        )

    def get_end_direction(self):
        if self.center is None:
            return (0, 0)
        
        angle = self.end_angle
        return (-math.sin(angle), math.cos(angle))
