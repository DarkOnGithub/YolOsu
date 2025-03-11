import math

def is_point_in_circle(point, center, radius):
    return distance_points(point, center) <= radius


def distance_points(p1, p2):
    x = (p1[0] - p2[0])
    y = (p1[1] - p2[1])
    return math.sqrt(x * x + y * y)


def distance_from_points(array):
    distance = 0

    for i in range(1, len(array)):
        distance += distance_points(array[i], array[i - 1])

    return distance


def angle_from_points(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def cart_from_pol(r, teta):
    x2 = (r * math.cos(teta))
    y2 = (r * math.sin(teta))

    return [x2, y2]


def point_at_distance(array, distance):
    current_distance = 0

    if len(array) < 2:
        return [0, 0, 0, 0]

    if distance == 0:
        angle = angle_from_points(array[0], array[1])
        return [array[0][0], array[0][1], angle, 0]

    if distance_from_points(array) <= distance:
        angle = angle_from_points(array[len(array) - 2], array[len(array) - 1])
        return [array[len(array) - 1][0],
                array[len(array) - 1][1],
                angle,
                len(array) - 2]

    i = 0
    new_distance = 0
    for i in range(len(array) - 1):
        x = (array[i][0] - array[i + 1][0])
        y = (array[i][1] - array[i + 1][1])

        new_distance = math.sqrt(x * x + y * y)
        current_distance += new_distance

        if distance <= current_distance:
            break

    current_distance -= new_distance

    if distance == current_distance:
        coord = [array[i][0], array[i][1]]
        angle = angle_from_points(array[i], array[i + 1])
    else:
        angle = angle_from_points(array[i], array[i + 1])
        cart = cart_from_pol((distance - current_distance), angle)

        if array[i][0] > array[i + 1][0]:
            coord = [(array[i][0] - cart[0]), (array[i][1] - cart[1])]
        else:
            coord = [(array[i][0] + cart[0]), (array[i][1] + cart[1])]

    return [coord[0], coord[1], angle, i]


def cpn(p, n):
    if p < 0 or p > n:
        return 0
    p = min(p, n - p)
    out = 1
    for i in range(1, p + 1):
        out = out * (n - p + i) / i
    return out


def array_values(array):
    if isinstance(array, dict):
        return list(array.values())
    elif not isinstance(array, list):
        return []
    return array


def array_calc(op, array1, array2):
    minimum = min(len(array1), len(array2))
    retour = []

    for i in range(minimum):
        try:
            if op == "+":
                retour.append(array1[i] + array2[i])
            elif op == "-":
                retour.append(array1[i] - array2[i])
            elif op == "*":
                retour.append(array1[i] * array2[i])
            elif op == "/":
                retour.append(array1[i] / array2[i] if array2[i] != 0 else 0)
            else:  
                retour.append(array1[i] + array2[i])
        except Exception:
            retour.append(0)

    return retour


class Bezier:
    def __init__(self, points):
        self.points = points
        self.order = len(points)

        self.step = (0.0025 / self.order) if self.order > 0 else 1  
        self.pos = {}
        self.pxlength = 0
        self.calc_points()

    def at(self, t):
        if t in self.pos:
            return self.pos[t]

        x = 0
        y = 0
        n = self.order - 1

        for i in range(n + 1):
            x += cpn(i, n) * ((1 - t) ** (n - i)) * (t ** i) * self.points[i][0]
            y += cpn(i, n) * ((1 - t) ** (n - i)) * (t ** i) * self.points[i][1]

        self.pos[t] = [x, y]

        return [x, y]

    
    def calc_points(self):
        if self.pos and self.pxlength > 0:  
            return

        self.pxlength = 0
        prev = self.at(0)
        i = 0
        end = 1 + self.step

        while i < end:
            current = self.at(i)
            self.pxlength += distance_points(prev, current)
            prev = current
            i += self.step

    def point_at_distance(self, dist):
        if self.order == 0:
            return [0, 0]
        elif self.order == 1:
            return self.points[0]
        else:
            return self.rec(dist)

    def rec(self, dist):
        self.calc_points()
        values = array_values(self.pos)
        if not values:
            return [0, 0]
        result = point_at_distance(values, dist)
        return result[:2] if result else [0, 0]



class Catmull:
    def __init__(self, points):
        self.points = points
        self.order = len(points)

        self.step = 0.025
        self.pos = []
        self.calc_points()

    def at(self, x, t):
        v1 = self.points[x - 1] if x >= 1 else self.points[x]
        v2 = self.points[x]
        v3 = self.points[x + 1] if x + 1 < self.order else self._calc_point(v2, v1, "-")
        v4 = self.points[x + 2] if x + 2 < self.order else self._calc_point(v3, v2, "-")

        retour = [0, 0]  
        for i in range(2):
            retour[i] = 0.5 * (
                (-v1[i] + 3 * v2[i] - 3 * v3[i] + v4[i]) * t * t * t + (
                    2 * v1[i] - 5 * v2[i] + 4 * v3[i] - v4[i]) * t * t + (
                    -v1[i] + v3[i]) * t + 2 * v2[i])

        return retour

    def _calc_point(self, p1, p2, op):
        result = []
        for i in range(2):
            if op == "+":
                result.append(p1[i] + p2[i])
            elif op == "-":
                result.append(p1[i] - p2[i])
        return result

    def calc_points(self):
        if self.pos:  
            return
        for i in range(self.order - 1):
            t = 0
            while t <= 1:
                self.pos.append(self.at(i, t))
                t += self.step

    def point_at_distance(self, dist):
        if self.order == 0:
            return [0, 0]
        elif self.order == 1:
            return self.points[0]
        else:
            return self.rec(dist)

    def rec(self, dist):
        self.calc_points()
        if not self.pos:
            return [0, 0]
        result = point_at_distance(self.pos, dist)
        return result[:2] if result else [0, 0]


class Linear:
    def __init__(self, points):
        self.points = points
        self.order = len(points)
        self.pxlength = 0
        if self.order >= 2:
            self.calc_points()
    
    def calc_points(self):
        if self.order < 2:
            return
            
        self.pxlength = 0
        for i in range(1, self.order):
            p1 = self.points[i-1]
            p2 = self.points[i]
            self.pxlength += distance_points(p1, p2)
    
    def point_on_line(self, p1, p2, length):
        full_length = math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))
        if full_length == 0:
            return p1
            
        n = full_length - length
        
        x = (n * p1[0] + length * p2[0]) / full_length
        y = (n * p1[1] + length * p2[1]) / full_length
        return [x, y]
    
    def point_at_distance(self, dist):
        if self.order < 2:
            return self.points[0] if self.order > 0 else [0, 0]
            
        if dist <= 0:
            return self.points[0]
        if dist >= self.pxlength:
            return self.points[-1]
            
        current_dist = 0
        for i in range(1, self.order):
            p1 = self.points[i-1]
            p2 = self.points[i]
            segment_length = distance_points(p1, p2)
            
            if current_dist + segment_length >= dist:
                segment_dist = dist - current_dist
                return self.point_on_line(p1, p2, segment_dist)
                
            current_dist += segment_length
            
        return self.points[-1]


class PassThrough:
    def __init__(self, points):
        self.points = points
        self.order = len(points)
        self.pxlength = 0
        self.cx = None
        self.cy = None
        self.radius = None
        self.calc_points()
        
    def calc_points(self):
        if self.order < 2:
            return
            
        
        if self.order == 2:
            linear = Linear(self.points)
            self.pxlength = linear.pxlength
            return
            
        
        if self.order > 3:
            bezier = Bezier(self.points)
            self.pxlength = bezier.pxlength
            return
            
        
        p1 = self.points[0]
        p2 = self.points[1]
        p3 = self.points[2]
        
        circle_data = self.get_circum_circle(p1, p2, p3)
        if circle_data is None:
            
            linear = Linear(self.points)
            self.pxlength = linear.pxlength
            return
            
        cx, cy, radius = circle_data
        
        self.cx = cx
        self.cy = cy
        self.radius = radius
        
        
        angle = self.calculate_arc_angle(p1, p2, p3, cx, cy)
        self.pxlength = abs(angle * radius)
    
    def get_circum_circle(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-10:  
            return None
            
        ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
        uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
        
        px = ux - x1
        py = uy - y1
        r = math.sqrt(px * px + py * py)
        
        return ux, uy, r
    
    def is_left(self, a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 0
    
    def rotate(self, cx, cy, x, y, radians):
        cos = math.cos(radians)
        sin = math.sin(radians)
        
        return [
            (cos * (x - cx)) - (sin * (y - cy)) + cx,
            (sin * (x - cx)) + (cos * (y - cy)) + cy
        ]
    
    def calculate_arc_angle(self, p1, p2, p3, cx, cy):
        
        angle = math.atan2(p3[1] - cy, p3[0] - cx) - math.atan2(p1[1] - cy, p1[0] - cx)
        
        
        if self.is_left(p1, p2, p3):
            if angle > 0:
                angle = -(2 * math.pi - angle)
        else:
            if angle < 0:
                angle = 2 * math.pi + angle
                
        return angle
    
    def point_at_distance(self, dist):
        if self.order < 2:
            return self.points[0] if self.order > 0 else [0, 0]
            
        
        if self.order == 2:
            linear = Linear(self.points)
            return linear.point_at_distance(dist)
            
        
        if self.order > 3:
            bezier = Bezier(self.points)
            return bezier.point_at_distance(dist)
            
        
        if dist <= 0:
            return self.points[0]
        if dist >= self.pxlength:
            return self.points[-1]
            
        
        if self.radius is None or self.cx is None or self.cy is None:
            linear = Linear(self.points)
            return linear.point_at_distance(dist)
            
        
        radians = dist / self.radius
        
        
        if self.is_left(self.points[0], self.points[1], self.points[2]):
            radians *= -1
            
        
        return self.rotate(self.cx, self.cy, self.points[0][0], self.points[0][1], radians)