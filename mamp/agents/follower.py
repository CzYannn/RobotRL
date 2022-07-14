import numpy as np
import math 
from mamp.util import wrap
from mamp.agents.robot import Robot 
import rospy

class Follower(Robot):
    """
    role of follower
    """
    def __init__(self, name, pref_speed, policy, id, RL):

        super().__init__(name, pref_speed, policy, id)
        self.group = 1
        self.v_x = None # linear of x axis
        self.w_z = None # angular of z axis
        # done flag
        self.failed_flag = None
        self.success_flag = None
        self.done = None 
        self.quit = None # 提前退出训练标志位
        # distance has made
        self.pose_list = []
        self.distance = 0
        self.last_distance = 0

        self.w_p =  1
        self.w_d = 0.1
        self.fix_action = [self.w_p, self.w_d]
        self.w_v =  1.0
        self.step = None

        self.RL = RL
        # RL action
        self.kp = 0
        self.kd = 0

        # formation error 
        self.error = np.zeros((5,2))

        # RL state
        self.state = []
        
        # 相邻机器人
        self.neighbor = []

        # 偏差信息
        self.error_list = []

    def reset(self, pos=None, vel=None, heading=None):
        """
        重置机器人状态
        """
        super().reset(pos, vel, heading)
        self.v_x = 0
        self.w_z = 0
        self.failed_flag = False  
        self.success_flag = False 
        self.done = False
        self.quit = False
        self.step = 0 

        self.pose_list = []
        self.error_list = []
        self.distance = 0
        self.last_distance = 0
        self.error = np.zeros((3,2))

    # def find_next_action(self, dict_comm):
    #     """
    #     UnicycleDynamics: formation control
    #     """  
    #     neighbor_num = len(self.neighbor_info)
    #     pos_diff_sum = np.zeros(2)
    #     vel_sum = np.zeros(2)
    #     for idx, pos_diff in self.neighbor_info.items():
    #         rel_pos_to_other_global_frame = self.pose_global_frame - dict_comm[idx]['pose_global_frame']
    #         pos_diff_sum -= rel_pos_to_other_global_frame - pos_diff
    #         vel_sum += dict_comm[idx]['vel_global_frame']
    #     pos_diff_sum = np.array(pos_diff_sum)
    #     vel_sum = np.array(vel_sum)
    #     self.error = np.insert(self.error, 0, pos_diff_sum, axis=0)
    #     self.error = np.delete(self.error, len(self.error)-1, axis=0)
    #     vel_consensus = self.w_p * self.error[0] + self.w_d *(self.error[0] - self.error[1])
    #     vel_consensus +=  1 / neighbor_num * vel_sum
    #     selected_heading = np.arctan2(vel_consensus[1], vel_consensus[0])
    #     selected_speed = math.sqrt(vel_consensus[0] ** 2 + vel_consensus[1] ** 2)
    #     delta_heading = wrap(selected_heading - self.heading_global_frame)

    #     action = np.array([selected_speed, delta_heading])

    #     return action

    # def act(self, action):
    #     self.step += 1
    #     self.rosport.pubTwist(action, dt=0.1)

    def stop(self):
        self.rosport.stop_moving()
    
    def act(self, ob, warmup=False, eval=False, action_num=6):
        """
        根据策略选择强化学习动作
        """
        if self.RL:
            action = self.policy.choose_action(ob, warmup, eval, action_num)
        else:
            action = self.fix_action
        return action

    def generate_next_command(self, action, dict_common):
        """
        根据强化学习动作产生控制指令
        """
        if self.RL:
            new_pid = dict(kp=action[0]*3, kd=action[1])
        else:
            new_pid = dict(kp=action[0], kd=action[1])
        self._update_pid(**new_pid)
        command = self._formation_control(dict_common)
        return command
    

    def _formation_control(self, dict_comm):
        """
        编队控制
        """
        neighbor_num = len(self.neighbor_info)
        pos_diff_sum = np.zeros(2)
        vel_sum = np.zeros(2)
        for idx, pos_diff in self.neighbor_info.items():
            rel_pos_to_other_global_frame = self.pose_global_frame - dict_comm[idx]['pose_global_frame']
            pos_diff_sum -= rel_pos_to_other_global_frame - pos_diff
            vel_sum += dict_comm[idx]['vel_global_frame']
        pos_diff_sum = np.array(pos_diff_sum)
        vel_sum = np.array(vel_sum)
        self.error_list.append(list(pos_diff_sum))
        self.error = np.insert(self.error, 0, pos_diff_sum, axis=0)
        self.error = np.delete(self.error, len(self.error)-1, axis=0)
        vel_consensus = self.kp * self.error[0] + self.kd *(self.error[0] - self.error[1])
        vel_consensus +=  1 / neighbor_num * vel_sum
        selected_heading = np.arctan2(vel_consensus[1], vel_consensus[0])
        selected_speed = math.sqrt(vel_consensus[0] ** 2 + vel_consensus[1] ** 2)
        delta_heading = wrap(selected_heading - self.heading_global_frame)

        command = np.array([selected_speed, delta_heading])

        return command

    
    def _update_pid(self, kp, kd):
        """
        更新PID系数
        """
        self.kp = kp 
        self.kd = kd

        


        
