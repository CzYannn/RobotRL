import numpy as np
import abc

class Robot(object):
    """
    机器人基类
    """

    def __init__(self, name, pref_speed, policy, id):
        
        # basic info
        self.name = name 
        self.pref_speed = pref_speed
        self.policy = policy 
        self.id = id
        self.group = None
        # state info
        self.pose_global_frame = np.zeros(2)
        self.vel_global_frame = np.zeros(2)
        self.heading_global_frame = 0
        # ros port
        self.rosport = None
        # neighbor robot info
        self.neighbor_info = {}

    def reset(self, pos=None, vel=None, heading=None):
        if pos is None:
            self.pose_global_frame = np.zeros(2)
        else: 
            self.pose_global_frame = pos
        if vel is None:
            self.vel_global_frame = np.zeros(2)
        else:
            self.vel_global_frame = vel 
        if heading is None:
            self.heading_global_frame = 0
        else:
            self.heading_global_frame = heading

    def set_pos(self, pos, heading=0):
        self.pose_global_frame = np.array(pos)
        self.heading_global_frame = heading
    
    @abc.abstractmethod
    def act(self, ob, warmup, eval, action_num):

        """
        select an action according to agent's policy
        """
    
    


        