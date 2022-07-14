import numpy as np

from mamp.util import wrap
from mamp.util import l2norm
from mamp.policies import Policy

from mamp.envs import Config


class TimePositionPolicy(Policy):
    """ Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents. """

    def __init__(self):
        Policy.__init__(self, str="TimePositionPolicy")
        self.type = "internal"
        self.now_goal = None
        self.now_time = 0

    def find_next_action(self, obs, agent):
        if not agent.path:
            # print(1111111111111111111111111111111)
            self.now_goal = agent.goal_global_frame

        if self.now_goal is None or (l2norm(agent.pos_global_frame, self.now_goal) <= agent.near_goal_threshold and
                                     agent.t - self.now_time >= Config.DT):
            time_pos = agent.path.pop()
            self.now_goal = np.array(time_pos['pos'], dtype='float64')
            self.now_time = np.array(time_pos['t'], dtype='float64')

        vec_to_next_pose = self.now_goal - agent.pos_global_frame
        heading_global_frame_exp = np.arctan2(vec_to_next_pose[1], vec_to_next_pose[0])
        delta_heading = wrap(heading_global_frame_exp - agent.heading_global_frame)
        action = np.array([agent.pref_speed, delta_heading])

        return action

    # def find_next_action1(self, obs, agent):
    #     if not agent.path: self.now_goal = agent.goal_global_frame
    #
    #     if self.now_goal is None or (l2norm(agent.pos_global_frame, self.now_goal) <= agent.near_goal_threshold and
    #                                  agent.t - self.now_time >= Config.DT):
    #         time_pos = agent.path.pop()
    #         self.now_goal = np.array(time_pos['pos'], dtype='float64')
    #         self.now_time = np.array(time_pos['t'], dtype='float64')
    #
    #     dis = l2norm(agent.pos_global_frame, self.now_goal)
    #     if dis <= agent.near_goal_threshold and agent.t - self.now_time < - Config.DT:
    #         select_speed = dis / Config.DT
    #     else:
    #         select_speed = agent.pref_speed
    #
    #     vec_to_next_pose = self.now_goal - agent.pos_global_frame
    #     heading_global_frame_exp = np.arctan2(vec_to_next_pose[1], vec_to_next_pose[0])
    #     delta_heading = wrap(heading_global_frame_exp - agent.heading_global_frame)
    #
    #     action = np.array([select_speed, delta_heading])
    #
    #     return action
