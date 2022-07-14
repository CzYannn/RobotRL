import numpy as np
from math import sqrt, sin, cos, atan2

from matp.policies import Policy
from matp.env_utils import Config

from matp.planner.mpcopt import agentQP, changeDestination, forwardState, transmitStates, Lambda, delta, A0, A, B


H = Config.H
h = Config.DT
agent_num = Config.NUM_AGENTS

"weight matrix"
Q = np.eye(3 * H)
R = np.eye(3 * H)
S = np.eye(3 * H)

"U constraints"
Aieq = np.vstack((-1 * np.eye(3 * H), np.eye(3 * H)))
bieq = 3 * np.ones((6 * H, 1))

"all agent states"
beta = 0.
alpha = 10.
ui_all = np.zeros((3, agent_num))  # all initial input

Adj = np.ones((agent_num, agent_num)) - np.eye(agent_num)

r_traj = 200.  # radius of leader trajectory
"rotationg formation"
class DMPRFCPolicy(Policy):

    def __init__(self):
        Policy.__init__(self, str="dmprfcPolicy")
        self.type = "internal"


    def find_next_action(self, agent, XallStates, p, v):

        p_x = p[:, 0][:, np.newaxis]
        p_y = p[:, 1][:, np.newaxis]
        p_z = p[:, 2][:, np.newaxis]

        V_x = v[:, 0][:, np.newaxis]
        V_y = v[:, 1][:, np.newaxis]
        V_z = v[:, 2][:, np.newaxis]

        u_x = np.zeros((agent_num, 1))
        u_y = np.zeros((agent_num, 1))
        u_z = np.zeros((agent_num, 1))
        # desired formation
        delta_body = np.array([[0, 0, 0, 0, 0],
                               [0, 60, -120, -180, 120],
                               [0, 0, 0, 0, 0]], dtype=float)


        if agent.id == 0:

            theta = atan2(XallStates[1, agent.id], XallStates[0, agent.id])
            p_xr = r_traj * cos(theta + np.pi/120)
            p_yr = r_traj * sin(theta + np.pi/120)
            p_zr = 0.
            Pdi = changeDestination(p_xr, p_yr, p_zr, H)
            u = agentQP(ui_all[:, agent.id], XallStates[:, agent.id], Pdi, H, Lambda, Q, delta, S, R, A0, Aieq, bieq)
            u0 = u[0: 3]
            # u0[2] = 0.
            XallStates[:, agent.id] = forwardState(XallStates[:, agent.id], u0, A, B)
            transmitStates(XallStates, p_x, p_y, p_z, V_x, V_y, V_z, agent_num)

            # global偏航角
            yaw = atan2(XallStates[4, agent.id], XallStates[3, agent.id])
            # global俯仰角
            pitch = -atan2(XallStates[5, agent.id], sqrt(pow(XallStates[3, agent.id], 2) + pow(XallStates[4, agent.id], 2)))
            roll = 0.0

            # body to global matrix
            Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]], dtype=float)
            Ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]], dtype=float)
            Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]], dtype=float)
            delta_global = Rx @ Ry @ Rz @ delta_body
            print([atan2(XallStates[1, i], XallStates[0, i]) for i in range(4)])

            # front-end feasible path planning(unnecessary)
            # for _ in range(200):
            #     for i in range(1, agent_num):
            #         u_x[i] = u_x[0] - alpha * (((p_x[i] - delta_global[0, i]) - p_x[0]) + beta * (V_x[i] - V_x[0]))
            #         u_y[i] = u_y[0] - alpha * (((p_y[i] - delta_global[1, i]) - p_y[0]) + beta * (V_y[i] - V_y[0]))
            #         u_z[i] = u_z[0] - alpha * (((p_z[i] - delta_global[2, i]) - p_z[0]) + beta * (V_z[i] - V_z[0]))
            #         for j in range(1, agent_num):
            #             u_x[i] -= Adj[i, j] * ((p_x[i] - p_x[j]) - (delta_global[0, i] - delta_global[0, j]) + beta * (V_x[i] - V_x[j]))
            #             u_y[i] -= Adj[i, j] * ((p_y[i] - p_y[j]) - (delta_global[1, i] - delta_global[1, j]) + beta * (V_y[i] - V_y[j]))
            #             u_z[i] -= Adj[i, j] * ((p_z[i] - p_z[j]) - (delta_global[2, i] - delta_global[2, j]) + beta * (V_z[i] - V_z[j]))
            #     # print(u_x.T,u_y.T,u_z.T)
            #     for i in range(1, agent_num):
            #         V_x[i] = V_x[i] + h * u_x[i]  # v = vo + at
            #         V_y[i] = V_y[i] + h * u_y[i]
            #         V_z[i] = V_z[i] + h * u_z[i]
            #
            #     for i in range(1, agent_num):
            #         p_x[i] = p_x[i] + h * V_x[i]  # x = xo + vt
            #         p_y[i] = p_y[i] + h * V_y[i]
            #         p_z[i] = p_z[i] + h * V_z[i]

            p_x[:, 0] = XallStates[0, 0] + delta_global[0, :]
            p_y[:, 0] = XallStates[1, 0] + delta_global[1, :]
            p_z[:, 0] = XallStates[2, 0] + delta_global[2, :]
            return u0, XallStates, np.hstack((p_x, p_y, p_z)), np.hstack((V_x, V_y, V_z))

        else:
            # back-end trajectory optimization problem
            Pdi = changeDestination(p_x[agent.id], p_y[agent.id], p_z[agent.id], H)
            u = agentQP(ui_all[:, agent.id], XallStates[:, agent.id], Pdi, H, Lambda, Q, delta, S, R, A0, Aieq, bieq)
            u0 = u[0: 3]
            # u0[2] = 0.
            XallStates[:, agent.id] = forwardState(XallStates[:, agent.id], u0, A, B)

            return u0, XallStates, p, v # MPC: using the first optimization input
