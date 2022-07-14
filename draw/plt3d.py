import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry import get_uav_model, get_car_model
from vis_util import plt_colors, base_fig_name


def draw_agent_3d(ax, pos_global_frame, heading_global_frame, my_agent_model, color='blue'):
    # heading_global_frame[0]: 偏航角，在xoy平面与x轴正方向的夹角，逆时针为正；
    # heading_global_frame[1]: 俯仰角，在yoz平面与y轴正方向的夹角，逆时针为正；
    # heading_global_frame[2]: 翻滚角，在xoz平面与x轴正方向的夹角，逆时针为正；
    agent_model = my_agent_model
    convert_to_actual_model_3d(agent_model, pos_global_frame, heading_global_frame)
    num_corner_point_per_layer = int(len(agent_model) / 2)
    x_list = []
    y_list = []
    z_list = []
    for layer in range(2):
        x_list.clear(), y_list.clear(), z_list.clear()
        for i in range(num_corner_point_per_layer):
            x_list.append(agent_model[i + layer * num_corner_point_per_layer][0])
            y_list.append(agent_model[i + layer * num_corner_point_per_layer][1])
            z_list.append(agent_model[i + layer * num_corner_point_per_layer][2])
        pannel = [list(zip(x_list, y_list, z_list))]
        ax.add_collection3d(Poly3DCollection(pannel, facecolors='goldenrod', alpha=0.9))  # , alpha=0.7

    for i in range(num_corner_point_per_layer - 1):
        x_list.clear(), y_list.clear(), z_list.clear()
        if i == 0:
            x_list.append(agent_model[0][0])
            x_list.append(agent_model[num_corner_point_per_layer - 1][0])
            x_list.append(agent_model[2 * num_corner_point_per_layer - 1][0])
            x_list.append(agent_model[num_corner_point_per_layer][0])
            y_list.append(agent_model[0][1])
            y_list.append(agent_model[num_corner_point_per_layer - 1][1])
            y_list.append(agent_model[2 * num_corner_point_per_layer - 1][1])
            y_list.append(agent_model[num_corner_point_per_layer][1])
            z_list.append(agent_model[0][2])
            z_list.append(agent_model[num_corner_point_per_layer - 1][2])
            z_list.append(agent_model[2 * num_corner_point_per_layer - 1][2])
            z_list.append(agent_model[num_corner_point_per_layer][2])
            pannel = [list(zip(x_list, y_list, z_list))]
            ax.add_collection3d(Poly3DCollection(pannel, facecolors=color, alpha=0.9))

        x_list.clear(), y_list.clear(), z_list.clear()
        x_list.append(agent_model[i][0])
        x_list.append(agent_model[i + 1][0])
        x_list.append(agent_model[i + 1 + num_corner_point_per_layer][0])
        x_list.append(agent_model[i + num_corner_point_per_layer][0])
        y_list.append(agent_model[i][1])
        y_list.append(agent_model[i + 1][1])
        y_list.append(agent_model[i + 1 + num_corner_point_per_layer][1])
        y_list.append(agent_model[i + num_corner_point_per_layer][1])
        z_list.append(agent_model[i][2])
        z_list.append(agent_model[i + 1][2])
        z_list.append(agent_model[i + 1 + num_corner_point_per_layer][2])
        z_list.append(agent_model[i + num_corner_point_per_layer][2])
        pannel = [list(zip(x_list, y_list, z_list))]

        # if i == 1:  # back1
        #     ax.add_collection3d(Poly3DCollection(pannel, facecolors='red'))
        # elif i == 2:  # back2
        #     ax.add_collection3d(Poly3DCollection(pannel, facecolors='red'))
        # else:
        #     ax.add_collection3d(Poly3DCollection(pannel))
        ax.add_collection3d(Poly3DCollection(pannel, facecolors=color, alpha=0.9))  # , alpha=0.7


def draw_circle_3d(ax, obstacles):
    for i, obstacle in enumerate(obstacles):
        alpha = 0.5
        color_ind = i % len(plt_colors)
        plt_color = plt_colors[color_ind]
        c = rgba2rgb(plt_color + [float(alpha)])
        if obstacle.shape == '3d-circle':
            center = obstacle.pos_global_frame
            radius = obstacle.radius
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            # surface plot rstride 值越大，图像越粗糙
            ax.plot_surface(x, y, z, rstride=8, cstride=8, color=c)
        else:
            raise NotImplementedError


def draw_traj_3d(ax, agents_info, agents_traj_list, step_num_list, current_step):
    for idx, agent_traj in enumerate(agents_traj_list):
        info = agents_info[idx]
        group = info['gp']
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num - 1:
            current_step = ag_step_num - 1

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        if group == 0:
            pos_z = np.ones_like(pos_x)
        else:
            pos_z = np.zeros_like(pos_x)
        alpha = agent_traj['alpha']
        beta  = np.zeros_like(alpha)
        gamma = np.zeros_like(alpha)

        # 绘制实线
        plt.plot(pos_x[:current_step], pos_y[:current_step], pos_z[:current_step], color=plt_color, ls='-', linewidth=2)
        # 绘制渐变线
        # colors = np.zeros((current_step, 4))
        # colors[:, :3] = plt_color
        # colors[:, 3] = np.linspace(0.2, 1., current_step)
        # colors = rgba2rgb(colors)
        # alphas = np.linspace(0.0, 1.0, current_step + 1)
        # for step in range(current_step):
        #     ax.scatter(pos_x[step], pos_y[step], pos_z[step], color=colors[step], s=3, alpha=alphas[step])
        # ax.scatter(pos_x[:current_step], pos_y[:current_step], pos_z[:current_step], color=colors, s=3, alpha=0.05)

        # # Also display circle at agent position at end of trajectory
        # ind = agent.step_num + last_index
        # alpha = 0.7
        # c = rgba2rgb(plt_color + [float(alpha)])
        # ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
        #                         radius=agent.radius, fc=c, ec=plt_color))
        # y_text_offset = 0.1
        # ax.text(agent.global_state_history[ind, 1] - 0.15,
        #         agent.global_state_history[ind, 2] + y_text_offset,
        #         '%d' % agent.id, color=plt_color)
        #####################################################################

        pos_global_frame = [pos_x[current_step], pos_y[current_step], pos_z[current_step]]
        heading_global_frame = [alpha[current_step], beta[current_step], gamma[current_step]]
        # print('pos_global_frame=', pos_global_frame)
        # print('heading_global_frame=', heading_global_frame)

        if group == 0:
            my_agent_model = get_uav_model()
        else:
            my_agent_model = get_car_model()
        draw_agent_3d(ax=ax,
                      pos_global_frame=pos_global_frame,
                      heading_global_frame=heading_global_frame,
                      my_agent_model=my_agent_model)


def plot_episode(agents_info, agents_traj_list, step_num_list, in_evaluate_mode=None, obstacles=None,
                 plot_save_dir=None, plot_policy_name=None, show_prob_matrix=False,
                 fig_size=(10, 8), show=False):

    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    # total_step = len(agents_traj_list[0])
    # print('num_agents:', num_agents, 'total_step:', total_step)

    while current_step < total_step:
        fig = plt.figure(0)
        fig.set_size_inches(fig_size[0], fig_size[1])

        # ax = fig.add_subplot(1, 1, 1)
        ax = Axes3D(fig)

        ax.set(xlabel='X',
               ylabel='Y',
               zlabel='Z',
               xlim=(-5, 10),
               ylim=(-5, 10),
               zlim=(0,   5),
               # xticks=np.arange(-5, 45, 1),
               # yticks=np.arange(-5, 45, 1),
               # zticks=np.arange(0, 5, 1)
               )

        # ax.view_init(elev=15,  # 仰角
        #              azim=60  # 方位角
        #              )

        # plt.grid()

        draw_traj_3d(ax, agents_info, agents_traj_list, step_num_list, current_step)
        if obstacles:
            pass

        fig_name = base_fig_name.format(
            policy="policy_name",
            num_agents=num_agents,
            test_case=str(0).zfill(3),
            step="_" + "{:05}".format(current_step),
            extension='png')
        filename = plot_save_dir + fig_name
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()
        print(filename)

        if show: plt.pause(0.0001)

        current_step += 5


def plot_episode_3d(agents_info, agents_traj_list, step_num_list, in_evaluate_mode=None, obstacles=None,
                    plot_save_dir=None, plot_policy_name=None, show_prob_matrix=False, config_info=None,
                    fig_size=(10, 8), show=False):
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    # total_step = len(agents_traj_list[0])
    # print('num_agents:', num_agents, 'total_step:', total_step)
    if show_prob_matrix:
        matrix = np.load('matrix.npy')
        grid = min(config_info['X_SEARCH'], config_info['Y_SEARCH'])
    else:
        matrix = None
        grid = None

    while current_step < total_step:
        fig = plt.figure(0)
        fig.set_size_inches(fig_size[0], fig_size[1])

        # ax = fig.add_subplot(1, 1, 1)
        ax = Axes3D(fig)

        ax.set(xlabel='X',
               ylabel='Y',
               zlabel='Z',
               xlim=(-5, 45),
               ylim=(-5, 45),
               zlim=(0, 5),
               # xticks=np.arange(-5, 45, 1),
               # yticks=np.arange(-5, 45, 1),
               # zticks=np.arange(0, 5, 1)
               )

        # ax.view_init(elev=15,  # 仰角
        #              azim=60  # 方位角
        #              )

        # plt.grid()
        s1 = time.time()
        if show_prob_matrix:
            draw_probability_matrix_3d(ax, matrix, grid, current_step)
        s2 = time.time()
        print('draw_prob', s2 - s1)

        draw_traj_3d(ax, agents_info, agents_traj_list, step_num_list, current_step)
        s3 = time.time()
        print('draw_traj', s3 - s2)
        if obstacles:
            pass

        fig_name = base_fig_name.format(
            policy="policy_name",
            num_agents=num_agents,
            test_case=str(0).zfill(3),
            step="_" + "{:05}".format(current_step),
            extension='png')
        filename = plot_save_dir + fig_name
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()
        # plt.savefig(filename, bbox_inches="tight")
        s4 = time.time()
        print('save', s4 - s3)
        print(filename)
        current_step += 5

        if show:
            plt.pause(0.0001)



def draw_probability_matrix_3d(ax, matrix, grid, current_step):
    current_matrix = matrix[current_step]
    for x_index, prob_list in enumerate(current_matrix):
        x_area = [x_index * grid, (x_index + 1) * grid]
        for y_index, prob in enumerate(prob_list):
            y_area = [y_index * grid, (y_index + 1) * grid]
            x_list = [x_area[0], x_area[0], x_area[1], x_area[1]]
            y_list = [y_area[0], y_area[1], y_area[1], y_area[0]]
            z_list = [0, 0, 0, 0]
            pannel = [list(zip(x_list, y_list, z_list))]
            ax.add_collection3d(Poly3DCollection(pannel, facecolors=(1, 0, 0), alpha=prob))

def draw_probability_matrix(ax, matrix, grid, current_step):
    current_matrix = matrix[current_step]
    for x_index, prob_list in enumerate(current_matrix):
        x_area = [x_index * grid, (x_index + 1) * grid]
        for y_index, prob in enumerate(prob_list):
            y_area = [y_index * grid, (y_index + 1) * grid]
            x_list = [x_area[0], x_area[0], x_area[1], x_area[1]]
            y_list = [y_area[0], y_area[1], y_area[1], y_area[0]]
            z_list = [0, 0, 0, 0]
            pannel = [list(zip(x_list, y_list, z_list))]
            ax.add_collection3d(Poly3DCollection(pannel, facecolors=(1, 0, 0), alpha=prob))

            # ax.plot_surface(X, Y, Z=X * 0 + 0.0,
            #                 # color='g',
            #                 color=(1, 0, 0),
            #                 alpha=prob
            #                 )
