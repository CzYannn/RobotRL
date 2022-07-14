import rospy
import numpy as np
import math 
from mamp.util import wrap
from mamp.agents.follower import Follower

class Leader(Follower):
    """
    role of leader 
    """
    def __init__(self, name, pref_speed, policy, id, RL):
        super().__init__(name, pref_speed, policy, id, RL)
        self.group = 0
        # infomation to control car
        self.error = list(np.zeros(5))
        self.error_k = list(np.zeros(5))
        self.errorx_list=[] 
        self.last_error = 0 # error made by last action
        # control command
        self.linearx = 0 
        self.angularz = 0
        # track line point
        self.cx = np.zeros(5)
        self.cy = np.zeros(5)
        # the action take
        self.kp = 0
        self.ki = 0
        self.kd = 0
        self.kp2 = 0
        self.ki2 = 0
        self.kd2 = 0
        # fix action 
        self.fix_action = [4, 0.2, 0.05, 2, 0.2, 0] # [kp, kd, ki, kp2, kd2, ki2]

    def reset(self, pos=None, vel=None, heading=None):
        super().reset(pos, vel, heading)
        self.kp = 0
        self.ki = 0
        self.kd = 0
        self.kp2 = 0
        self.ki2 = 0
        self.kd2 = 0

        self.linearx = 0
        self.angularz = 0
        self.cx = np.zeros(5)
        self.cy = np.zeros(5)
        self.error = list(np.zeros(5))
        self.error_k = list(np.zeros(5))
        self.last_error = 0


    def _pid1_incremental(self):
        """
        1. 计算主要增量式PID控制器输出角速度
        2. 计算输出线速度
        3. 主要增量式和辅助增量式输出做叠加得到最终角速度
        """
        er1 = self.error[0]
        er2 = self.error[1]
        er3 = self.error[2]

        self.linearx = self.pref_speed - abs(self.error[0])*0.5

        add_wz = -(self.kp * (er1 - er2) + self.ki * er1 + self.kd * (er1 - 2 * er2 + er3))
        angular_z = self.angularz
        angular_z += (add_wz+0.5*self._pid2_incremental())
        angular_z = max(-np.pi, min(np.pi, angular_z)) 
        self.angularz = self.angularz*0.1+0.9*angular_z

        return [self.linearx, self.angularz]

    def _pid2_incremental(self):

        """
        计算辅助式PID增量控制器输出角速度
        """
    
        er1_k = self.error_k[0]
        er2_k = self.error_k[1]
        er3_k = self.error_k[2]
        add_wz_2 = self.kp2 * (er1_k - er2_k) + self.ki2 * er1_k + \
                    self.kd2 * (er1_k - 2 * er2_k + er3_k)
        return add_wz_2
    
    def _updata_pid(self,kp,kd,kp2,kd2,ki=None,ki2=None):

        """
        更新PID系数
        """
        if ki and ki2:
            self.kp = kp
            self.kd = kd
            self.ki = ki
            self.kp2 = kp2
            self.ki2 = ki2
            self.kd2 = kd2
        else:
            self.kp = kp
            self.kd = kd
            self.kp2 = kp2
            self.kd2 = kd2
    
    # def find_next_action(self, ob, warmup=False, eval=False, action_num=6, RL=True):

    #     """
    #     1. 根据状态选择动作
    #     2. 根据动作更新线速度，角速度指令        
    #     """
    #     action = self.policy.choose_action(ob, warmup, eval, action_num)
    #     if not RL:
    #         fix_pid = dict(kp=4, kd=0.2, ki=0.05, kp2=2, kd2=0.2, ki2=0)
    #         self._updata_pid(**fix_pid)
    #         command = self._pid1_incremental()
    #         return action , command
    #     if len(action) == 6 :
    #         new_pid = dict(kp=action[0]*4, kd=action[1], kp2=action[2]*4, kd2=action[3],ki=action[4], ki2=action[5])
    #     elif len(action) == 4 :
    #         new_pid = dict(kp=action[0]*4, kd=action[1], kp2=action[2]*4, kd2=action[3])
    #     self._updata_pid(**new_pid)
    #     command = self._pid1_incremental()
    #     return action, command
    
    def generate_next_command(self, action, dict_common):
        """
        根据RL动作产生控制指令
        """
        if len(action) == 6:
            if self.RL:
                new_pid = dict(kp=action[0]*4, kd=action[1], kp2=action[2]*4, kd2=action[3], ki=action[4], ki2=action[5])
            else:
                new_pid = dict(kp=action[0], kd=action[1], kp2=action[2], kd2=action[3], ki=action[4], ki2=action[5])
        if len(action) == 4:
            if self.RL:
                new_pid = dict(kp=action[0]*4, kd=action[1], kp2=action[2]*4, kd2=action[3])
            else:
                new_pid = dict(kp=action[0], kd=action[1], kp2=action[2], kd2=action[3])
        self._updata_pid(**new_pid)
        command = self._pid1_incremental()
        return command

    def act(self, ob, warmup=False, eval=False, action_num=6):
        """
        根据策略选择强化学习动作
        """
        if self.RL:
            action = self.policy.choose_action(ob, warmup, eval, action_num)
        else:
            action = self.fix_action
        return action


        













