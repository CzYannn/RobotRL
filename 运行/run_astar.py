#!/usr/bin/env python
import rospkg
import rospy
import ros

import os
import gym
import numpy as np

gym.logger.set_level(40)

os.environ['GYM_CONFIG_CLASS'] = 'AStar'
os.environ['GYM_CONFIG_PATH'] = 'mamp/configs/config.py'


from mamp.agents.agent import Agent
from mamp.agents.obstacle import Obstacle
# Planner
from mamp.planner.astar import AStar
# Policies
from mamp.policies import policy_dict
# Dynamics
from mamp.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from mamp.sensors import Sensor


def build_agents():
    start_x = 0
    start_y = 0
    goal_x = 9
    goal_y = 9

    radius = 0.1
    pref_speed = 1.0
    initial_heading = np.pi

    agents = [
        Agent(start_pos=[start_x, start_y],
              goal_pos=[goal_x, goal_y],
              radius=radius,
              pref_speed=pref_speed,
              initial_heading=initial_heading,
              policy=policy_dict['noncoopl'],
              dynamics_model=UnicycleDynamics,
              sensors=[Sensor],
              id=0),
#        Agent(start_pos=[start_x + 2, start_y],
#              goal_pos=[goal_x, goal_y],
#              radius=radius,
#              pref_speed=pref_speed,
#              initial_heading=initial_heading,
#              policy=policy_dict['noncoopl'],
#              dynamics_model=UnicycleDynamics,
#              sensors=[Sensor],
#              id=1),
#        Agent(start_pos=[start_x + 4, start_y],
#              goal_pos=[goal_x - 6, goal_y],
#              radius=radius,
#              pref_speed=pref_speed,
#              initial_heading=initial_heading,
#              policy=policy_dict['noncoopl'],
#              dynamics_model=UnicycleDynamics,
#              sensors=[Sensor],
#              id=2),
#        Agent(start_pos=[start_x + 6, start_y],
#              goal_pos=[goal_x - 4, goal_y],
#              radius=radius,
#              pref_speed=pref_speed,
#              initial_heading=initial_heading,
#              policy=policy_dict['noncoopl'],
#              dynamics_model=UnicycleDynamics,
#              sensors=[Sensor],
#              id=3),
#        Agent(start_pos=[start_x + 8, start_y],
#              goal_pos=[goal_x - 6, goal_y],
#              radius=radius,
#              pref_speed=pref_speed,
#              initial_heading=initial_heading,
#              policy=policy_dict['noncoopl'],
#              dynamics_model=UnicycleDynamics,
#              sensors=[Sensor],
#              id=4),
#        Agent(start_pos=[goal_x - 6, goal_y],
#              goal_pos=[start_x + 8, start_y],
#              radius=radius,
#              pref_speed=pref_speed,
#              initial_heading=initial_heading,
#              policy=policy_dict['noncoopl'],
#              dynamics_model=UnicycleDynamics,
#              sensors=[Sensor],
#              id=5),
    ]
    return agents


def build_obstacles():
    obstacles = [
        Obstacle(pos=[3.0, 3.0], shape_dict={'shape': "circle", 'feature': 1.0}, id=0),
        Obstacle(pos=[3.0, 1.0], shape_dict={'shape': "circle", 'feature': 0.5}, id=1),
        Obstacle(pos=[2.0, 5.0], shape_dict={'shape': "circle", 'feature': 0.2}, id=2),
        Obstacle(pos=[6.0, 6.0], shape_dict={'shape': "circle", 'feature': 0.8}, id=3),
        Obstacle(pos=[9.0, 8.0], shape_dict={'shape': "circle", 'feature': 0.5}, id=4),
    ]
    return obstacles


if __name__ == '__main__':
    rospack = rospkg.RosPack()#ros
    rospy.init_node('nn_jackal',anonymous=False)#ros

    # Instantiate the environment
    env = gym.make("MultiAgentCollisionAvoidance-v0")  # MultiAgentCollisionAvoidance  MultiAgent

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/astar/')
    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = build_agents()
    obstacles = build_obstacles()
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    env.set_agents(agents, obstacles=obstacles)

    for a in agents:
        start_pos = a.pos_global_frame
        goal_pos = a.goal_global_frame
        planner = AStar(start=start_pos, end=goal_pos, danger=obstacles)
        a.set_planner(planner)

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 400
    obs = env.reset()  # Get agents' initial observations
    for i in range(num_steps):
        # Query the external agents' policies
        # e.g., actions[0] = external_policy(dict_obs[0])
        actions = {}

        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)
        rospy.spin()


        if game_over:
            print("All agents finished!")
            break

    env.reset()
