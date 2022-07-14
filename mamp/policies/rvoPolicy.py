import numpy as np
from math import pi, sqrt, cos, sin, atan2, asin
from mamp.policies import Policy



class RVOPolicy(Policy):
    """ RVOPolicy Agents simply drive at pref speed toward the goal, ignoring other agents. """

    def __init__(self):
        Policy.__init__(self, str="RVOPolicy")
        self.type = "internal"
        self.now_goal = None
        self.agents = None
        self.obstacles = None

    def find_next_action(self, obs, agent):
        """ compute best velocity given the desired velocity, current velocity and workspace model"""
        v_des = self.compute_v_des(agent.pos_global_frame, agent.goal_global_frame, agent.pref_speed)
        # print(1111, v_des)
        rob_radius = agent.radius + 0.1
        v_opt = list(agent.speed)
        vA = [agent.speed[0], agent.speed[1]]
        pA = [agent.pos_global_frame[0], agent.pos_global_frame[1]]
        RVO_BA_all = []
        for a in self.agents:
            if a.id != agent.id:
                vB = [a.speed[0], a.speed[1]]
                pB = [a.pos_global_frame[0], a.pos_global_frame[1]]
                # use RVO
                transl_vB_vA = [pA[0] + 0.5 * (vB[0] + vA[0]), pA[1] + 0.5 * (vB[1] + vA[1])]
                # use VO
                # transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1]]
                dist_BA = self.distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
                if 2 * rob_radius > dist_BA:
                    dist_BA = 2 * rob_radius
                theta_BAort = asin(2 * rob_radius / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                theta_ort_right = theta_BA - theta_BAort
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                # use HRVO
                # dist_dif = distance([0.5*(vB[0]-vA[0]),0.5*(vB[1]-vA[1])],[0,0])
                # transl_vB_vA = [pA[0]+vB[0]+cos(theta_ort_left)*dist_dif, pA[1]+vB[1]+sin(theta_ort_left)*dist_dif]
                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2 * rob_radius]
                RVO_BA_all.append(RVO_BA)
        for obstacle in self.obstacles:
            # hole = [x, y, rad]
            vB = [0, 0]
            pB = [obstacle.x, obstacle.y]
            transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
            dist_BA = self.distance(pA, pB)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            # over-approximation of square to circular
            OVER_APPROX_C2S = 1.0
            rad = obstacle.r * OVER_APPROX_C2S
            if (rad + rob_radius) > dist_BA:
                dist_BA = rad + rob_radius
            theta_BAort = asin((rad + rob_radius) / dist_BA)
            theta_ort_left = theta_BA + theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            theta_ort_right = theta_BA - theta_BAort
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + rob_radius]
            RVO_BA_all.append(RVO_BA)
        vA_post = self.intersect(pA, v_des, RVO_BA_all)
        v_opt = vA_post[:]
        action = np.array([v_opt[0], v_opt[1]])
        # print("agent"+str(agent.id)+"'s action is:", action)
        return action

    def set_agents(self, agnets, obstacles):
        self.agents = agnets
        self.obstacles = obstacles
        # for a in self.agents:
        #     print(a.dic_start_pos['start'], a.dic_goal_pos['goal'])
        # for ob in self.obstacles:
        #     print(ob.x, ob.y, ob.r)

    def distance(self, pose1, pose2):
        """ compute Euclidean distance for 2D """
        return sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2) + 0.001

    def intersect(self, pA, vA, RVO_BA_all):
        # print('----------------------------------------')
        # print('Start intersection test')
        # print(11111, vA)
        norm_v = self.distance(vA, [0, 0])
        suitable_V = []
        unsuitable_V = []
        for theta in np.arange(0, 2 * pi, 0.1):      # 63 times  对应速度可取的角度2pi内分为63等分
            for rad in np.arange(0.02, norm_v + 0.02, norm_v / 5.0):     # 5 times  将最小速度和期望速度分为五等分
                new_v = [rad * cos(theta), rad * sin(theta)]
                suit = True
                for RVO_BA in RVO_BA_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
                    theta_dif = atan2(dif[1], dif[0])
                    theta_right = atan2(right[1], right[0])
                    theta_left = atan2(left[1], left[0])
                    if self.in_between(theta_right, theta_dif, theta_left):
                        suit = False
                        break
                if suit:
                    suitable_V.append(new_v)
                else:
                    unsuitable_V.append(new_v)
        new_v = vA[:]
        suit = True
        for RVO_BA in RVO_BA_all:
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
            theta_dif = atan2(dif[1], dif[0])
            theta_right = atan2(right[1], right[0])
            theta_left = atan2(left[1], left[0])
            if self.in_between(theta_right, theta_dif, theta_left):
                suit = False
                break
        if suit:
            suitable_V.append(new_v)
        else:
            unsuitable_V.append(new_v)
        # ----------------------
        if suitable_V:
            # print 'Suitable found'
            vA_post = min(suitable_V, key=lambda v: self.distance(v, vA))
        else:
            # print 'Suitable not found'
            tc_V = dict()       # 该速度下与障碍物发生碰撞的时间
            for unsuit_v in unsuitable_V:
                tc_V[tuple(unsuit_v)] = 0
                tc = []
                for RVO_BA in RVO_BA_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dist = RVO_BA[3]
                    rad = RVO_BA[4]
                    dif = [unsuit_v[0] + pA[0] - p_0[0], unsuit_v[1] + pA[1] - p_0[1]]      # 在遍历unsuit_v的时候已经计算过一次
                    theta_dif = atan2(dif[1], dif[0])                                       # 在遍历unsuit_v的时候已经计算过一次
                    theta_right = atan2(right[1], right[0])                                 # 在遍历unsuit_v的时候已经计算过一次
                    theta_left = atan2(left[1], left[0])                                    # 在遍历unsuit_v的时候已经计算过一次
                    if self.in_between(theta_right, theta_dif, theta_left):
                        small_theta = abs(theta_dif - 0.5 * (theta_left + theta_right))
                        if abs(dist * sin(small_theta)) >= rad:
                            rad = abs(dist * sin(small_theta))
                        big_theta = asin(abs(dist * sin(small_theta)) / rad)
                        dist_tg = abs(dist * cos(small_theta)) - abs(rad * cos(big_theta))
                        if dist_tg < 0:
                            dist_tg = 0
                        tc_v = dist_tg / self.distance(dif, [0, 0])
                        tc.append(tc_v)
                tc_V[tuple(unsuit_v)] = min(tc) + 0.001
            WT = 0.2
            vA_post = min(unsuitable_V, key=lambda v: ((WT / tc_V[tuple(v)]) + self.distance(v, vA)))
        return vA_post

    def in_between(self, theta_right, theta_dif, theta_left):
        if abs(theta_right - theta_left) <= pi:
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        else:
            if (theta_left < 0) and (theta_right > 0):
                theta_left += 2 * pi
                if theta_dif < 0:
                    theta_dif += 2 * pi
                if theta_right <= theta_dif <= theta_left:
                    return True
                else:
                    return False
            if (theta_left > 0) and (theta_right < 0):
                theta_right += 2 * pi
                if theta_dif < 0:
                    theta_dif += 2 * pi
                if theta_left <= theta_dif <= theta_right:
                    return True
                else:
                    return False

    def compute_v_des(self, x, goal, v_max):
        v_des = []
        dif_x = [goal[k] - x[k] for k in range(2)]
        norm = self.distance(dif_x, [0, 0])
        norm_dif_x = [dif_x[k] * v_max / norm for k in range(2)]
        v_des = norm_dif_x[:]  # a=norm_dif_x[:],相当于重新开辟一段地址
        if self.reach(x, goal, 0.1):
            v_des[0] = 0
            v_des[1] = 0
        return v_des

    def reach(self, p1, p2, bound=0.5):
        if self.distance(p1, p2) < bound:
            return True
        else:
            return False


if __name__ == "__main__":
    pass

