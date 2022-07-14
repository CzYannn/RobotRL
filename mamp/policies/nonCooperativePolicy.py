import numpy as np

from mamp.util import wrap
from mamp.util import l2norm
from mamp.policies import Policy

class NonCooperativePolicy(Policy):
    """ Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents. """
    def __init__(self):
        Policy.__init__(self, str="NonCooperativePolicy")
        self.type = "internal"

    def find_next_action(self, obs, agent):
        """ Go at pref_speed, apply a change in heading equal to zero out current ego heading (heading to goal)
        Returns:
            np array of shape (2,)... [spd, delta_heading]
        """
        action = np.array([agent.pref_speed, -agent.heading_ego_frame])
        return action


class NonCooperativePlanner(Policy):
    """ Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents. """
    def __init__(self):
        Policy.__init__(self, str="NonCooperativePlanner")
        self.type = "internal"
        self.now_goal = None

    def find_next_action(self, obs, agent):
        """ Go at pref_speed, apply a change in heading equal to zero out current ego heading (heading to goal)
        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            i (int): this agent's index in that list
        Returns:
            np array of shape (2,)... [spd, delta_heading]

        """
        if self.now_goal is None:   # first
            if agent.path:
                self.now_goal = np.array(agent.path.pop(), dtype='float64')
            else:
                self.now_goal = agent.goal_global_frame

        dis = l2norm(agent.pos_global_frame, self.now_goal)

        if dis <= agent.near_goal_threshold:
            if agent.path:
                self.now_goal = np.array(agent.path.pop(), dtype='float64')
            else:
                self.now_goal = agent.goal_global_frame

        vec_to_next_pose = self.now_goal - agent.pos_global_frame
        heading_global_frame_exp = np.arctan2(vec_to_next_pose[1], vec_to_next_pose[0])
        delta_heading = wrap(heading_global_frame_exp - agent.heading_global_frame)
        action = np.array([agent.pref_speed, delta_heading])

        return action

