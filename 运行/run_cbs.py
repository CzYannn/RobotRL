import os
import gym
import numpy as np

gym.logger.set_level(40)

os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ['GYM_CONFIG_PATH'] = 'mamp/configs/config.py'

from mamp.agents.agent import Agent
from mamp.agents.obstacle import Obstacle
# Planner
from mamp.planner.cbs import Environment, CBS
# Policies
from mamp.policies import policy_dict
# Dynamics
from mamp.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from mamp.sensors import Sensor


def build_agents():
    # start_pos_list = [[2.0, 1.0], [4.0, 1.0]]
    # goal_pos_list = [[4.0, 3.0], [2.0, 3.0]]
    start_pos_list = [[4.0, 1.0], [7.0, 1.0], [1.0, 4.0], [3.0, 5.0], [6.0, 2.0], [5.0, 5.0]]
    goal_pos_list = [[6.0, 6.0], [0.0, 4.0], [6.0, 2.0], [5.0, 3.0], [3.0, 4.0], [3.0, 1.0]]
    radius = 0.1
    pref_speed = 1.0
    initial_heading = np.pi

    agents = [
        Agent(start_pos=start_pos_list[0],
              goal_pos=goal_pos_list[0],
              radius=radius, pref_speed=pref_speed, initial_heading=initial_heading,
              policy=policy_dict['tpp'],
              dynamics_model=UnicycleDynamics,
              sensors=[Sensor],
              id=0),
        Agent(start_pos=start_pos_list[1],
              goal_pos=goal_pos_list[1],
              radius=radius, pref_speed=pref_speed, initial_heading=initial_heading,
              policy=policy_dict['tpp'],
              dynamics_model=UnicycleDynamics,
              sensors=[Sensor],
              id=1),
        Agent(start_pos=start_pos_list[2],
              goal_pos=goal_pos_list[2],
              radius=radius, pref_speed=pref_speed, initial_heading=initial_heading,
              policy=policy_dict['tpp'],
              dynamics_model=UnicycleDynamics,
              sensors=[Sensor],
              id=2),
        Agent(start_pos=start_pos_list[3],
              goal_pos=goal_pos_list[3],
              radius=radius, pref_speed=pref_speed, initial_heading=initial_heading,
              policy=policy_dict['tpp'],
              dynamics_model=UnicycleDynamics,
              sensors=[Sensor],
              id=3),
        Agent(start_pos=start_pos_list[4],
              goal_pos=goal_pos_list[4],
              radius=radius, pref_speed=pref_speed, initial_heading=initial_heading,
              policy=policy_dict['tpp'],
              dynamics_model=UnicycleDynamics,
              sensors=[Sensor],
              id=4),
        Agent(start_pos=start_pos_list[5],
              goal_pos=goal_pos_list[5],
              radius=radius, pref_speed=pref_speed, initial_heading=initial_heading,
              policy=policy_dict['tpp'],
              dynamics_model=UnicycleDynamics,
              sensors=[Sensor],
              id=5),

    ]
    return agents


def build_obstacles():
    obstacles = [
        # Obstacle(pos=[2.0, 2.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=0),
        # Obstacle(pos=[4.0, 2.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=1),
        Obstacle(pos=[7.0, 3.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=0),
        Obstacle(pos=[0.0, 5.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=1),
        Obstacle(pos=[2.0, 5.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=2),
        Obstacle(pos=[7.0, 5.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=3),
        Obstacle(pos=[6.0, 0.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=4),
        Obstacle(pos=[0.0, 3.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=5),
        Obstacle(pos=[6.0, 1.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=6),
        Obstacle(pos=[1.0, 7.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=7),
        Obstacle(pos=[5.0, 6.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=8),
        Obstacle(pos=[3.0, 3.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=9),
        Obstacle(pos=[5.0, 1.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=10),
        Obstacle(pos=[1.0, 5.0], shape_dict={'shape': "rect", 'feature': (0.5, 0.5)}, id=11)
    ]
    return obstacles


if __name__ == '__main__':
    # Instantiate the environment
    env = gym.make("MultiAgentCollisionAvoidance-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(os.path.dirname(os.path.realpath(__file__)) + '/results/cbs/')
    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = build_agents()
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    obstacles = build_obstacles()
    env.set_agents(agents, obstacles=obstacles)

    obs = env.reset()  # Get agents' initial observations

    dimension = [12, 12]
    cbs_obstacles = []
    for obstacle in obstacles:
        cbs_obstacles.append((obstacle.x, obstacle.y))

    env_cbs = Environment(dimension, agents, cbs_obstacles)
    planner = CBS(env_cbs)
    env.set_cbs_planner(planner)

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 400
    for i in range(num_steps):
        # Query the external agents' policies
        # e.g., actions[0] = external_policy(dict_obs[0])
        actions = {}

        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)

        if game_over:
            print("All agents finished!")
            break

    env.reset()
