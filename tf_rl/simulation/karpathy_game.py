import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

import tf_rl.utils.svg as svg

class GameObject(object):
    def __init__(self, position, speed, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.settings = settings
        self.radius = self.settings["object_radius"]

        self.obj_type = obj_type
        self.position = position
        self.speed    = speed
        self.bounciness = 1.0

    def wall_collisions(self):
        """Update speed upon collision with the wall."""
        world_size = self.settings["world_size"]

        for dim in range(2):
            if self.position[dim] - self.radius       <= 0               and self.speed[dim] < 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness
            elif self.position[dim] + self.radius + 1 >= world_size[dim] and self.speed[dim] > 0:
                self.speed[dim] = - self.speed[dim] * self.bounciness

    def move(self, dt):
        """Move as if dt seconds passed"""
        self.position += dt * self.speed
        self.position = Point2(*self.position)

    def step(self, dt):
        """Move and bounce of walls."""
        self.wall_collisions()
        self.move(dt)

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        return svg.Circle(self.position + Point2(10, 10), self.radius, color=color)

class KarpathyGame(object):
    def __init__(self, settings):
        """Initiallize game simulator with settings"""
        self.settings = settings
        self.size  = self.settings["world_size"]
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]

        self.hero = GameObject(Point2(*self.settings["hero_initial_position"]),
                               Vector2(*self.settings["hero_initial_speed"]),
                               "hero",
                               self.settings)
        if not self.settings["hero_bounces_off_walls"]:
            self.hero.bounciness = 0.0

        self.objects = []
        for obj_type, number in settings["num_objects"].items():
            for _ in range(number):
                self.spawn_object(obj_type)

        self.observation_lines = self.generate_observation_lines()

        self.object_reward = 0
        self.collected_rewards = []

        # every observation_line sees one of objects or wall and
        # two numbers representing speed of the object (if applicable)
        self.eye_observation_size = len(self.settings["objects"]) + 3
        # additionally there are two numbers representing agents own speed and position.
        self.observation_size = self.eye_observation_size * len(self.observation_lines) + 2 + 2

        self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0],[0,-1],[0.0,0.0]]]
        self.num_actions      = len(self.directions)

        self.objects_eaten = defaultdict(lambda: 0)

    def perform_action(self, action_id):
        """Change speed to one of hero vectors"""
        assert 0 <= action_id < self.num_actions
        self.hero.speed *= 0.5
        self.hero.speed += self.directions[action_id] * self.settings["delta_v"]

    def spawn_object(self, obj_type):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]
        position = np.random.uniform([radius, radius], np.array(self.size) - radius)
        position = Point2(float(position[0]), float(position[1]))
        max_speed = np.array(self.settings["maximum_speed"])
        speed    = np.random.uniform(-max_speed, max_speed).astype(float)
        speed = Vector2(float(speed[0]), float(speed[1]))

        self.objects.append(GameObject(position, speed, obj_type, self.settings))

    def step(self, dt):
        """Simulate all the objects for a given ammount of time.

        Also resolve collisions with the hero"""
        for obj in self.objects + [self.hero] :
            obj.step(dt)
        self.resolve_collisions()

    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def resolve_collisions(self):
        """If hero touches, hero eats. Also reward gets updated."""
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        to_remove = []
        for obj in self.objects:
            if self.squared_distance(self.hero.position, obj.position) < collision_distance2:
                to_remove.append(obj)
        for obj in to_remove:
            self.objects.remove(obj)
            self.objects_eaten[obj.obj_type] += 1
            self.object_reward += self.settings["object_reward"][obj.obj_type]
            self.spawn_object(obj.obj_type)

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    def observe(self):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        num_obj_types = len(self.settings["objects"]) + 1 # and wall
        max_speed_x, max_speed_y = self.settings["maximum_speed"]

        observable_distance = self.settings["observation_line_length"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        observation        = np.zeros(self.observation_size)
        observation_offset = 0
        for i, observation_line in enumerate(self.observation_lines):
            # shift to hero position
            observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                                            self.hero.position + Vector2(*observation_line.p2))

            observed_object = None
            # if end of observation line is outside of walls, we see the wall.
            if not self.inside_walls(observation_line.p2):
                observed_object = "**wall**"
            for obj in relevant_objects:
                if observation_line.distance(obj.position) < self.settings["object_radius"]:
                    observed_object = obj
                    break
            object_type_id = None
            speed_x, speed_y = 0, 0
            proximity = 0
            if observed_object == "**wall**": # wall seen
                object_type_id = num_obj_types - 1
                # a wall has fairly low speed...
                speed_x, speed_y = 0, 0
                # best candidate is intersection between
                # observation_line and a wall, that's
                # closest to the hero
                best_candidate = None
                for wall in self.walls:
                    candidate = observation_line.intersect(wall)
                    if candidate is not None:
                        if (best_candidate is None or
                                best_candidate.distance(self.hero.position) >
                                candidate.distance(self.hero.position)):
                            best_candidate = candidate
                if best_candidate is None:
                    # assume it is due to rounding errors
                    # and wall is barely touching observation line
                    proximity = observable_distance
                else:
                    proximity = best_candidate.distance(self.hero.position)
            elif observed_object is not None: # agent seen
                object_type_id = self.settings["objects"].index(observed_object.obj_type)
                speed_x, speed_y = tuple(observed_object.speed)
                intersection_segment = obj.as_circle().intersect(observation_line)
                assert intersection_segment is not None
                try:
                    proximity = min(intersection_segment.p1.distance(self.hero.position),
                                    intersection_segment.p2.distance(self.hero.position))
                except AttributeError:
                    proximity = observable_distance
            for object_type_idx_loop in range(num_obj_types):
                observation[observation_offset + object_type_idx_loop] = 1.0
            if object_type_id is not None:
                observation[observation_offset + object_type_id] = proximity / observable_distance
            observation[observation_offset + num_obj_types] =     speed_x   / max_speed_x
            observation[observation_offset + num_obj_types + 1] = speed_y   / max_speed_y
            assert num_obj_types + 2 == self.eye_observation_size
            observation_offset += self.eye_observation_size

        observation[observation_offset]     = self.hero.speed[0] / max_speed_x
        observation[observation_offset + 1] = self.hero.speed[1] / max_speed_y
        observation_offset += 2
        
        # add normalized locaiton of the hero in environment        
        observation[observation_offset]     = self.hero.position[0] / 350.0 - 1.0
        observation[observation_offset + 1] = self.hero.position[1] / 250.0 - 1.0
        
        assert observation_offset + 2 == self.observation_size

        return observation

    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        return res - self.settings["object_radius"]

    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""
        wall_reward =  self.settings["wall_distance_penalty"] * \
                       np.exp(-self.distance_to_walls() / self.settings["tolerable_distance_to_wall"])
        assert wall_reward < 1e-3, "You are rewarding hero for being close to the wall!"
        total_reward = wall_reward + self.object_reward
        self.object_reward = 0
        self.collected_rewards.append(total_reward)
        return total_reward

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])
        for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        recent_reward = self.collected_rewards[-100:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            "nearest wall = %.1f" % (self.distance_to_walls(),),
            "reward       = %.1f" % (sum(recent_reward)/len(recent_reward),),
            "objects eaten => %s" % (objects_eaten_str,),
        ])

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), self.size))


        for line in self.observation_lines:
            scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
                               line.p2 + self.hero.position + Point2(10,10)))

        for obj in self.objects + [self.hero] :
            scene.add(obj.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene

