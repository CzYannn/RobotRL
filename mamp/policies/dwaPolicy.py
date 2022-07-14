import numpy as np
from math import *

from mamp.envs import Config
from mamp.policies import Policy


class DWAPolicy(Policy):
    """ Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents. """

    def __init__(self):
        Policy.__init__(self, str="DWAPolicy")
        self.V_Min = 0.0
        self.V_Max = 2.0
        self.W_Min = -30 * pi / 180.0
        self.W_Max = 30 * pi / 180.0
        self.Va = 4.0
        self.Wa = 60.0 * pi / 180.0
        self.Vreso = 0.1
        self.Wreso = 0.5 * pi / 180.0
        self.radius = 0.1
        self.Dt = 0.1
        self.Predict_Time = 4.0
        self.alpha = 1.0
        self.Belta = 1.0
        self.Gamma = 1.0

        self.inflation = 0.2

        # self.x =np.array([self.agents[i].start_pos[0], self.agents[i].start_pos[1],45*pi/180,0,0])
        # self.u =np.array([0,0])

        self.type = "internal"
        self.current = None

    def find_next_action(self, obs, agent):
        """X[0] += u[0] * dt * cos(X[2])  # x方向上位置
        X[1] += u[0] * dt * sin(X[2])  # y方向上位置
        X[2] += u[1] * dt  # 角度变换
        X[3] = u[0]  # 速度
        X[4] = u[1]  # 角速度"""
        i = agent.id
        if agent.goal_global_frame is not None:
            if self.flag[i] == False:
                self.x[i] = np.array([agent.pos_global_frame[0], agent.pos_global_frame[1], 45 * pi / 180, 0, 0])
                # self.x[i]=np.array([3.0,0.0,45*pi/180,0,0])
                self.u[i] = np.array([0, 0])
                # self.goal=np.array(agents[i].goal_pos)
                global_tarj = np.array(self.x[i])
                self.flag[i] = True
            else:
                # goal = np.array(agents[i].goal_pos)
                self.x[i][0] = agent.pos_global_frame[0]
                self.x[i][1] = agent.pos_global_frame[1]
                self.x[i][2] = agent.radianY

                self.u[i], self.current = self.dwa_Core(self.x[i], self.u[i], agent.goal_global_frame, agent.other_agent_list,)
                self.x[i] = self.Motion(self.x[i], self.u[i], self.Dt)
#                print("agent_pos:")
#                print(agent.pos_global_frame)
#                print("ag.other_agent_list")
#                print(agent.other_agent_list[0].pos_global_frame)
    #            print("policy:")
    #            print(self.x[i][3])  #xian su du
    #            print(self.u~[i][1] * self.Dt)# dan wei shi jian zhuan guo de jiao du
            action = np.array([self.x[i][3], self.x[i][4]])
#            action = np.array([self.x[i][3], self.u[i][1]*self.Dt])
#            print("action:")
#            print(action)
            return action
        else :
            action = np.array([0,0])
            return action

    def Goal_Cost(self, Goal, Pos):
        return sqrt((Pos[-1, 0] - Goal[0]) ** 2 + (Pos[-1, 1] - Goal[1]) ** 2)

    # 速度评价函数
    def Velocity_Cost(self, Pos):
        return self.V_Max - Pos[-1, 3]

    # 距离障碍物距离的评价函数
    def Obstacle_Cost(self, Pos, Obstacle=None, agents=None):
        MinDistance = float('Inf')  # 初始化时候机器人周围无障碍物所以最小距离设为无穷
        MinDistance_ob = float('Inf')  # 初始化时候机器人周围无障碍物所以最小距离设为无穷
        MinDistance_ag = float('Inf')  # 初始化时候机器人周围无障碍物所以最小距离设为无穷
        for i in range(len(Pos)):  # 对每一个位置点循环
            if Obstacle is not None:
                for j in range(len(Obstacle)):  # 对每一个障碍物循环
                    if Obstacle[j] is not None:
                        Current_Distance_ob = sqrt(
                            (Pos[i, 0] - Obstacle[j].x) ** 2 + (Pos[i, 1] - Obstacle[j].y) ** 2)  # 求出每个点和每个障碍物距离
                        if Current_Distance_ob < self.radius + Obstacle[
                            j].radius + self.inflation:  # 如果小于机器人自身的半径那肯定撞到障碍物了返回的评价值自然为无穷
                            return float('Inf')
                        if Current_Distance_ob < MinDistance:
                            MinDistance_ob = Current_Distance_ob
            else :
                pass

            if agents is not None:
                for j in range(len(agents)):  # 对每一个障碍物循环
                    Current_Distance_ag = sqrt((Pos[i, 0] - agents[j].pos_global_frame[0]) ** 2 +
                        (Pos[i, 1] - agents[j].pos_global_frame[1]) ** 2)  # 求出每个点和每个障碍物距离
                    if Current_Distance_ag < self.radius + agents[j].radius + self.inflation:  # 如果小于机器人自身的半径那肯定撞到障碍物了返回的评价值自然为无穷
                        return float('Inf')
                    if Current_Distance_ag < MinDistance:
                        MinDistance_ag = Current_Distance_ag

            MinDistance = min(MinDistance_ag, MinDistance_ob)
            # if Current_Distance_ob < MinDistance:
            #     MinDistance = Current_Distance_ob  # 得到点和障碍物距离的最小

        return 1 / MinDistance

    # 速度采用
    def V_Range(self, X):
        Vmin_Actual = X[3] - self.Va * self.Dt  # 实际在dt时间间隔内的最小速度
        Vmax_actual = X[3] + self.Va * self.Dt  # 实际载dt时间间隔内的最大速度
        Wmin_Actual = X[4] - self.Wa * self.Dt  # 实际在dt时间间隔内的最小角速度
        Wmax_Actual = X[4] + self.Wa * self.Dt  # 实际在dt时间间隔内的最大角速度
        VW = [max(self.V_Min, Vmin_Actual), min(self.V_Max, Vmax_actual), max(self.W_Min, Wmin_Actual),
              min(self.W_Max, Wmax_Actual)]  # 因为前面本身定义了机器人最小最大速度所以这里取交集
        return VW

    # 一条模拟轨迹路线中的位置，速度计算
    def Motion(self, X, u, dt):
        X[0] += u[0] * dt * cos(X[2])  # x方向上位置
        X[1] += u[0] * dt * sin(X[2])  # y方向上位置
        X[2] += u[1] * dt  # 角度变换
        X[3] = u[0]  # 速度
        X[4] = u[1]  # 角速度
        return X
#    def Motion(self, X, u, dt):
#        X[0] += u[0] * dt  # x方向上位置
#        X[1] = 0
#        X[2] += u[1] * dt  # 角度变换
#        X[3] = u[0]  # 速度
#        X[4] = u[1]  # 角速度
#        return X

    # 一条模拟轨迹的完整计算
    def Calculate_Traj(self, X, u):
        Traj = np.array(X)
        Xnew = np.array(X)
        time = 0
        while time <= self.Predict_Time:  # 一条模拟轨迹时间
            Xnew = self.Motion(Xnew, u, self.Dt)
            Traj = np.vstack((Traj, Xnew))  # 一条完整模拟轨迹中所有信息集合成一个矩阵
            time = time + self.Dt
        return Traj

    # DWA核心计算
    def dwa_Core(self, X, u, goal, agents=None, obstacles=None):
        vw = self.V_Range(X)
        best_traj = np.array(X)
        min_score = 10000.0  # 随便设置一下初始的最小评价分数
        for v in np.arange(vw[0], vw[1], self.Vreso):  # 对每一个线速度循环
            for w in np.arange(vw[2], vw[3], self.Wreso):  # 对每一个角速度循环
                traj = self.Calculate_Traj(X, [v, w])
                goal_score = self.Goal_Cost(goal, traj)
                vel_score = self.Velocity_Cost(traj)
                obs_score = self.Obstacle_Cost(traj, agents = agents)
#                print("obstacle_list")
#                print(self.agents[0].obstacle_list[0].x)
                score = goal_score + vel_score + obs_score
                if min_score >= score:  # 得出最优评分和轨迹
                    min_score = score
                    u = np.array([v, w])
                    best_traj = traj

        return u, best_traj

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_agents(self, agnets):
        self.agents = agnets
        self.x = [None for _ in range(len(agnets))]
        self.u = [None for _ in range(len(agnets))]
        self.flag = [False for _ in range(len(agnets))]



