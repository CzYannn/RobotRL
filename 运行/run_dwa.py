#!/usr/bin/env python3
import sys
sys.path.append('/home/cjj/MACoordinate/MAROS')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import gym

import rospkg
import rospy
import threading

gym.logger.set_level(40)

os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ['GYM_CONFIG_PATH'] = '../mamp/configs/config.py'


from mamp.agents.agent import Agent
from mamp.agents.obstacle import Obstacle

# Policies
from mamp.policies import policy_dict
# Dynamics
from mamp.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from mamp.sensors import OtherAgentsObsSensor

def thread_job():
    rospy.spin()


def build_agents():
    start_x = 0
    start_y = 0
    goal_x = 9
    goal_y = 9
    radius = 0.1
    pref_speed = 2.0
    initial_heading = None # np.pi
    radius1 = 0.2
    pref_speed1 = 0.9
    agents = [
        Agent(
                start_pos=[start_x+3, start_y],
                goal_pos=[goal_x, goal_y],
                name = 'Agent1',
                radius=radius, pref_speed=pref_speed,
                initial_heading=initial_heading,
                policy=policy_dict['noncoop'],
                dynamics_model=UnicycleDynamics,
                sensors=[OtherAgentsObsSensor],
                id=0),
        Agent(
                start_pos=[start_x+3, start_y],
                goal_pos=[goal_x, goal_y],
                name = 'Agent2',
                radius=radius, pref_speed=pref_speed,
                initial_heading=initial_heading,
                policy=policy_dict['noncoop'],
                dynamics_model=UnicycleDynamics,
                sensors=[OtherAgentsObsSensor],
                id=1),
        Agent(
                start_pos=[start_x+3, start_y],
                goal_pos=[goal_x, goal_y],
                name = 'Agent3',
                radius=radius, pref_speed=pref_speed,
                initial_heading=initial_heading,
                policy=policy_dict['noncoop'],
                dynamics_model=UnicycleDynamics,
                sensors=[OtherAgentsObsSensor],
                id=2),
        ]
    return agents



def build_obstacles():
    obstacles = [
#        Obstacle(pos= [ 3.0, 3.0], shape_dict= {'shape': "circle", 'feature': 1.0}, id = 0),
#        Obstacle(pos= [ 3.0, 1.0], shape_dict= {'shape': "circle", 'feature': 0.5}, id = 1),
#        Obstacle(pos= [ 2.0, 5.0], shape_dict= {'shape': "circle", 'feature': 0.2}, id=2),
#        Obstacle(pos= [ 6.0, 6.0], shape_dict= {'shape': "circle", 'feature': 0.8}, id=3),
#        Obstacle(pos= [ 9.0, 8.0], shape_dict= {'shape': "circle", 'feature': 0.1}, id=4),
        # Obstacle(pos= [ 7.0, 8.0], shape_dict= {'shape': "circle", 'feature': 1.5}, id=5),

        # Obstacle(pos= [-1.0, 1.0], shape_dict= {'shape': "circle", 'feature': 0.5}, id = 1),
        # Obstacle(pos= [-1.0, 1.0], shape_dict= {'shape': "rect", 'feature': (0.5, 0.5)}, id = 3),
        ]
    return obstacles


if __name__ == '__main__':
    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = build_agents()
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    obstacles = build_obstacles()

    # for i, ag in enumerate(agents):
    #     ag.policy.set_obstacles(obstacles)
    #     ag.policy.set_agents(agents)

    env = gym.make("MultiAgentCollisionAvoidance-v0")
    # env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/dwa/')
    for ag in agents:
        ag.set_other_agents(ag.id, agents)

    env.set_agents(agents, obstacles = obstacles)
    obs = env.reset() # Get agents' initial observations

    add_thread = threading.Thread(target = thread_job)
    add_thread.start()
    epi_maximum = 10
    game_over = False
    for epi in range(epi_maximum):
        print(epi)
        while not game_over:
            actions = {}
            obs, rewards, game_over, which_agents_done = env.step(actions)
        env.set_agents(agents, obstacles = obstacles)
        env.reset()
        print("All agents finished!")
    print("Experiment over.")
