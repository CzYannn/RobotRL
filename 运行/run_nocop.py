#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.append('/home/miao/workspace/python_test/maros')
sys.path.append('/home/cjj/MACoordinate/MAROS')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ['GYM_CONFIG_PATH'] = '../mamp/configs/config.py'
import gym
gym.logger.set_level(40)

from mamp.agents.agent import Agent
# Policies
from mamp.policies import policy_dict
# Dynamics
from mamp.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from mamp.sensors import OtherAgentsObsSensor

def build_agents():
    start_x = 0
    start_y = 0
    goal_x  = 10
    goal_y  = 10
    radius  = 0.1
    pref_speed = 2.0
    initial_heading = np.pi / 2 # None #
    agents = [
        Agent(
                start_pos=[start_x+3, start_y+1],
                goal_pos=[goal_x+3, goal_y+1],
                name = 'Agent1',
                radius=radius, pref_speed=pref_speed,
                initial_heading=initial_heading,
                policy=policy_dict['noncoop'],
                dynamics_model=UnicycleDynamics,
                sensors=[OtherAgentsObsSensor],
                id=0),
        Agent(
                start_pos=[start_x, start_y],
                goal_pos=[goal_x, goal_y],
                name = 'Agent2',
                radius=radius, pref_speed=pref_speed,
                initial_heading=initial_heading,
                policy=policy_dict['noncoop'],
                dynamics_model=UnicycleDynamics,
                sensors=[OtherAgentsObsSensor],
                id=1),
        Agent(
                start_pos=[start_x+1, start_y+3],
                goal_pos=[goal_x+1, goal_y+3],
                name = 'Agent3',
                radius=radius, pref_speed=pref_speed,
                initial_heading=initial_heading,
                policy=policy_dict['noncoop'],
                dynamics_model=UnicycleDynamics,
                sensors=[OtherAgentsObsSensor],
                id=2),
        ]
    return agents


if __name__ == '__main__':
    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = build_agents()
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]

    env = gym.make("MAGazebo-v0")
    # env = gym.make("MultiAgentCollisionAvoidance-v0")
    env.set_agents(agents)

    epi_maximum = 1
    for epi in range(epi_maximum):
        env.reset()
        print("episode:", epi)
        game_over = False
        while not game_over:
            actions = {}
            obs, rewards, game_over, which_agents_done = env.step(actions)
        print("All agents finished!", env.episode_step_number)
    print("Experiment over.")



    log_save_dir = os.path.dirname(os.path.realpath(__file__)) + '/../draw/noncoop/log/'
    os.makedirs(log_save_dir, exist_ok=True)

    # trajectory
    writer = pd.ExcelWriter(log_save_dir + '/trajs.xlsx')
    for agent in agents:
        agent.history_info.to_excel(writer, sheet_name='agent' + str(agent.id))
    writer.save()

    # scenario information
    info_dict_to_visualize = {
        'all_agent_info': [],
        'all_obstacle': [],
    }
    for agent in agents:
        agent_info_dict = {'id': agent.id, 'gp': agent.group, 'radius': agent.radius, 'goal_pos': agent.goal_global_frame.tolist()}
        info_dict_to_visualize['all_agent_info'].append(agent_info_dict)

    info_str = json.dumps(info_dict_to_visualize, indent=4)
    with open(log_save_dir + '/env_cfg.json', 'w') as json_file:
        json_file.write(info_str)
    json_file.close()
