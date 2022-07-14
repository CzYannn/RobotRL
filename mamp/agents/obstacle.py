import numpy as np
import math

from mamp.envs import Config
from mamp.util import wrap


class Obstacle(object):
    def __init__(self, pos, shape_dict, id=0):
        self.shape = shape = shape_dict['shape']
        if shape == 'rect':
            self.width, self.heigh = shape_dict['feature']
            self.radius = math.sqrt(self.width ** 2 + self.heigh ** 2) / 2
        elif shape == 'circle':
            self.radius = shape_dict['feature']
        else:
            raise NotImplementedError

        self.pos_global_frame = np.array(pos, dtype='float64')
        self.id = id
        self.t = 0.0
        self.step_num = 0
        self.is_at_goal = True
        self.was_in_collision_already = False
        self.in_collision = False
        if pos is not None :
            self.x = pos[0]
            self.y = pos[1]
        self.r = self.radius

