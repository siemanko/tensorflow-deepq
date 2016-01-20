import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2
from itertools import combinations

import tf_rl.utils.svg as svg


class GameObject(object):
    def __init__(self, position, speed, obj_type, radius=0.03, mass=1.0):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.position = position
        self.speed    = speed
        self.obj_type = obj_type
        self.mass     = mass
        self.radius   = radius

    def step(self, dt, viscosity=1.0):
        """Move as if dt seconds passed"""
        self.position += dt * self.speed
        self.position.x = max(self.radius, min(1.0 - self.radius, self.position.x))
        self.position.y = max(self.radius, min(1.0 - self.radius, self.position.y))
        self.speed *= viscosity

    def as_circle(self):
        return Circle(self.position, float(self.radius))


def speed_after_collision(a, b, cor=1.0):
    a_vdiff = cor * (a.speed - b.speed)
    a_xdiff = (a.position - b.position)
    a_rmass = 2 * b.mass / (a.mass + b.mass)
    a_factor = a_rmass * a_vdiff.dot(a_xdiff) / a_xdiff.magnitude_squared()
    return a.speed - a_factor * a_xdiff

def objects_colliding(a, b):
    distance_squared = (a.position - b.position).magnitude_squared()
    return distance_squared < (a.radius + b.radius)**2

def objects_will_collide(a, b, dt):
    distance_squared = ((a.position + a.speed*dt) - (b.position + b.speed * dt)).magnitude_squared()
    return distance_squared < (a.radius + b.radius)**2

def wall_collision_soon(a, dt):
    cur = a.position
    nex = a.position + a.speed * dt
    r = a.radius
    if cur.x < r or nex.x < r:
        return Vector2(-1, 0)
    if cur.x > 1 - r or nex.x > 1 - r:
        return Vector2(1, 0)
    if cur.y < r or nex.y < r:
        return Vector2(0, -1)
    if cur.y > 1 - r or nex.y > 1 - r:
        return Vector2(0, 1)
    return None

def resolve_wall_colision(obj, direction, cor=1.0):
    wall_obj =  GameObject(obj.position + direction * obj.radius, Vector2(0.0,0.0), "wall", 0.0, mass=1000000.0)
    obj.speed = speed_after_collision(obj, wall_obj, cor=cor)

def correct_penetration(a, b):
    sep    = a.position.distance(b.position)
    minsep = a.radius + b.radius
    correction = minsep - sep
    if correction > 0:
        dir_a = (a.position - b.position)
        dir_a.normalize()
        dir_a *= correction / 2.
        a.position += dir_a
        b.position -= dir_a

class Simulation(object):
    def __init__(self, settings):
        self.objects = []
        self.settings = settings
        self.size = self.settings["size"]
        self.restitution = self.settings["restitution"]
        self.viscosity   = self.settings["viscosity"]
        self.game_time_passed = 0.0

    def add(self, obj):
        self.objects.append(obj)

    def _repr_html_(self):
        return self.to_html()

    def step(self, dt):
        """Simulate all the objects for a given ammount of time.

        Also resolve collisions with the hero"""
        for obj in self.objects:
            obj.step(dt, self.viscosity)

        for obj in self.objects:
            wall_col_dir = wall_collision_soon(obj, dt)
            if wall_col_dir is not None:
                resolve_wall_colision(obj, wall_col_dir, self.restitution)

        for obj1, obj2 in combinations(self.objects, 2):
            if objects_colliding(obj1, obj2) or objects_will_collide(obj1, obj2, dt):
                obj1.speed, obj2.speed = (
                    speed_after_collision(obj1, obj2, self.restitution),
                    speed_after_collision(obj2, obj1, self.restitution),
                )
            if objects_colliding(obj1, obj2):
                correct_penetration(obj1, obj2)

        self.game_time_passed += dt

    def create_scene(self, stats):
        scene = svg.Scene((self.size + 20, self.size + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), (self.size, self.size)))
        return scene

    def draw_objects(self, scene):
        for obj in self.objects:
            color = self.settings["colors"][obj.obj_type]
            obj_drawing = svg.Circle(obj.position * self.size + Point2(10, 10), obj.radius * self.size, color=color)
            scene.add(obj_drawing)

    def draw_stats(self, scene, stats):
        offset = self.size + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        scene = self.create_scene(stats)
        self.draw_objects(scene)
        self.draw_stats(scene, stats)

        return scene


class HeroSimulation(Simulation):
    def __init__(self, settings):
        super(HeroSimulation, self).__init__(settings)

        self.walls = [
            LineSegment2(Point2(0,   0),    Point2(0,   1.0)),
            LineSegment2(Point2(0,   1.0),  Point2(1.0, 1.0)),
            LineSegment2(Point2(1.0, 1.0),  Point2(1.0, 0)),
            LineSegment2(Point2(1.0, 0),    Point2(0,   0))
        ]

        self.observation_lines = self.generate_observation_lines()
        self.line_end_where = {l: l.p2 for l in self.observation_lines}
        self.line_end_who   = {l: None for l in self.observation_lines}

        self.hero = GameObject(Point2(0.5,0.5), Vector2(0.0,0.0), "hero", radius=self.settings['obj_radius'])
        self.add(self.hero)

        self.update_observation_lines()
        self.observation_size = self.settings["num_observation_lines"] * (len(self.settings["observable_objects"]) + 4)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observable_distance"], self.settings["observable_distance"])
        for angle in np.linspace(0, 2 * np.pi, self.settings["num_observation_lines"], endpoint=False):
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    def add(self, obj):
        super(HeroSimulation, self).add(obj)
        self.update_observation_lines()

    def observe(self):
        observable_distance = self.settings["observable_distance"]
        nl = self.settings["num_observation_lines"]
        observ_obj = self.settings["observable_objects"]
        no = len(observ_obj)

        observation = np.empty((nl, no + 4))

        hero_speed_x, hero_speed_y = self.hero.speed
        for i, line in enumerate(self.observation_lines):
            obj = self.line_end_who[line]
            speed_x, speed_y = 0.0, 0.0
            if obj is not None and type(obj) == GameObject:
                speed_x, speed_y = obj.speed
                obj = obj.obj_type

            observation[i] = 1.0
            if obj in observ_obj:
                proximity = self.hero.position.distance(self.line_end_where[line]) / observable_distance
                observation[i, observ_obj.index(obj)] = proximity
            observation[i, no:] = (speed_x, speed_y, hero_speed_x, hero_speed_y)
        return observation.ravel()

    def update_observation_lines(self):
        observable_distance = self.settings["observable_distance"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance
                               and obj is not self.hero]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))


        for observation_line in self.observation_lines:
            # shift to hero position
            l = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                             self.hero.position + Vector2(*observation_line.p2))
            start_l, end_l = l.p1, l.p2

            observed_object = None
            # if end of observation line is outside of walls, we see the wall.
            if end_l.x < 0.0 or end_l.x > 1.0 or end_l.y < 0.0 or end_l.y > 1.0:
                observed_object = "wall"
            for obj in relevant_objects:
                if l.distance(obj.position) < obj.radius:
                    observed_object = obj
                    break

            intersection = end_l
            if observed_object == "wall": # wall seen
                # best candidate is intersection between
                # l and a wall, that's
                # closest to the hero
                for wall in self.walls:
                    candidate = l.intersect(wall)
                    if candidate is not None:
                        if (intersection is None or
                                intersection.distance(self.hero.position) >
                                candidate.distance(self.hero.position)):
                            intersection = candidate
            elif observed_object is not None: # agent seen
                intersection_segment = obj.as_circle().intersect(l)
                if intersection_segment is not None:
                    if (intersection_segment.p1.distance(self.hero.position) <
                            intersection_segment.p2.distance(self.hero.position)):
                        intersection = intersection_segment.pl
                    else:
                        intersection = intersection_segment.p2

            self.line_end_where[observation_line] = intersection
            self.line_end_who[observation_line]   = observed_object

    def step(self, dt):
        super(HeroSimulation, self).step(dt)
        self.update_observation_lines()

    def draw_observation(self, scene):
        for line in self.observation_lines:
            obj = self.line_end_who[line]
            color = self.settings["colors"][obj.obj_type] if type(obj) is GameObject else 'black'
            line_drawn = svg.Line(self.hero.position * self.size + Point2(10,10),
                                  self.line_end_where[line] * self.size + Point2(10,10),
                                  stroke=color)
            scene.add(line_drawn)

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        scene = self.create_scene(stats)
        self.draw_observation(scene)
        self.draw_objects(scene)
        self.draw_stats(scene, stats)

        return scene


class ConvSimulation(Simulation):
    def __init__(self, settings):
        super(ConvSimulation, self).__init__(settings)
        self.hero = GameObject(Point2(0.5,0.5), Vector2(0.0,0.0), "hero", radius=self.settings['obj_radius'])
        self.add(self.hero)
        self.observation_res  = self.settings["observation_resolution"]
        self.observation_size = [self.observation_res, self.observation_res, len(self.settings["observable_objects"]) + 2]

    def observe(self):
        observ_obj = self.settings["observable_objects"]
        no = len(observ_obj)
        observation = np.zeros((self.observation_res, self.observation_res, no + 2))
        def rescale(i):
            return min(self.observation_res - 1, max(0, int(round(self.observation_res * i))))

        for obj in self.objects:
            cx, cy = rescale(obj.position.x), rescale(obj.position.y)
            sx, ex = obj.position.x - obj.radius, obj.position.x + obj.radius
            sy, ey = obj.position.y - obj.radius, obj.position.y + obj.radius
            sx, ex, sy, ey = [ rescale(i) for i in [sx,ex,sy,ey]]
            layer = observ_obj.index(obj.obj_type)
            for x in range(sx, ex + 1):
                for y in range(sy, ey + 1):
                    if (x-cx)**2 + (y-cy)**2 <= (obj.radius*self.observation_res)**2:
                        observation[x,y,layer]  = 1.0
                        observation[x,y,no]     = obj.speed.x
                        observation[x,y,no + 1] = obj.speed.y
        return observation
