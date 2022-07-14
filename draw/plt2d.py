import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from math import sin, cos, atan2, sqrt, pow

from geometry import get_2d_car_model, get_2d_uav_model
from vis_util import plt_colors, rgba2rgb

simple_plot = True

def convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame):
    alpha = heading_global_frame
    for point in agent_model:
        x = point[0]
        y = point[1]
        # 进行航向计算
        l = sqrt(pow(x, 2) + pow(y, 2))
        alpha_model = atan2(y, x)
        alpha_ = alpha + alpha_model - np.pi / 2  # 改加 - np.pi / 2 因为画模型的时候UAV朝向就是正北方向，所以要减去90°
        point[0] = l * cos(alpha_) + pos_global_frame[0]
        point[1] = l * sin(alpha_) + pos_global_frame[1]


def draw_agent_2d(ax, pos_global_frame, heading_global_frame, my_agent_model, color='blue'):
    agent_model = my_agent_model
    convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame)

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(agent_model, codes)

    # 第二步：创建一个patch，路径依然也是通过patch实现的，只不过叫做pathpatch
    patch = patches.PathPatch(path, facecolor='orange', lw=2)

    ax.add_patch(patch)


def draw_traj_2d(ax, agents_info, agents_traj_list, step_num_list, current_step):
    for idx, agent_traj in enumerate(agents_traj_list):
        agent_id = agents_info[idx]['id']
        agent_gp = agents_info[idx]['gp']
        agent_rd = agents_info[idx]['radius']
        agent_goal = agents_info[idx]['goal_pos']
        info     = agents_info[idx]
        group = info['gp']
        radius = info['radius'] / 1
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num-1:
            plot_step = ag_step_num - 1
        else:
            plot_step = current_step

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        alpha = agent_traj['alpha']

        # 绘制目标点
        plt.plot(agent_goal[0], agent_goal[1], color=plt_color, marker='*', markersize=20)

        # 绘制实线
        plt.plot(pos_x[:plot_step], pos_y[:plot_step], color=plt_color, ls='-', linewidth=2)
        # 绘制渐变线
        colors = np.zeros((plot_step, 4))
        colors[:, :3] = plt_color
        colors[:, 3] = np.linspace(0.2, 1., plot_step)
        colors = rgba2rgb(colors)
        alphas = np.linspace(0.0, 1.0, plot_step + 1)
        # for step in range(plot_step):
        #     ax.scatter(pos_x[step], pos_y[step], color=colors[step], s=3, alpha=alphas[step])
        # ax.scatter(pos_x[:plot_step], pos_y[:plot_step], color=colors, s=3, alpha=0.5)
        plt.grid()

        # # Also display circle at agent position at end of trajectory
        # ind = ag_step_num
        # alpha = 0.7
        # c = rgba2rgb(plt_color + [float(alpha)])
        # ax.add_patch(plt.Circle(agent_traj[ind, 1:3], radius=agent_rd, fc=c, ec=plt_color))
        #####################################################################
        if simple_plot:
            ax.add_patch(plt.Circle((pos_x[plot_step], pos_y[plot_step]), radius=radius, fc=plt_color, ec=plt_color))
            y_text_offset = 0.1
            ax.text(pos_x[plot_step] - 0.15, pos_y[plot_step] + y_text_offset, '%d' % agent_id, color=plt_color)
        else:
            if group == 0:
                my_model = get_2d_uav_model(size=agent_rd)
            else:
                my_model = get_2d_car_model(size=agent_rd)
            pos = [pos_x[plot_step], pos_y[plot_step]]
            heading = alpha[plot_step]
            draw_agent_2d(ax, pos, heading, my_model)


def plot_save_one_pic(agents_info, agents_traj_list, step_num_list, filename, current_step):
    fig = plt.figure(0)
    fig_size = (10, 8)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='X',
           ylabel='Y',
           # xlim=(-2, 10),
           # ylim=(-2, 10),
           )
    draw_traj_2d(ax, agents_info, agents_traj_list, step_num_list, current_step)
    fig.savefig(filename, bbox_inches="tight")
    # fig.savefig(filename)
    # plt.clear()
    plt.close()

def plot_episode(agents_info, traj_list, step_num_list,  plot_save_dir, base_fig_name, last_fig_name, show=False):
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    print('num_agents:', num_agents, 'total_step:', total_step)
    while current_step < total_step:
        fig_name = base_fig_name + "_{:05}".format(current_step) + '.png'
        filename = plot_save_dir + fig_name
        plot_save_one_pic(agents_info, traj_list, step_num_list, filename, current_step)
        print(filename)
        current_step += 1
    filename = plot_save_dir + last_fig_name
    plot_save_one_pic(agents_info, traj_list, step_num_list, filename, total_step)




