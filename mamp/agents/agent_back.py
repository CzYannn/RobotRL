#!/usr/bin/env python3
import math
import numpy as np
from mamp.envs import Config
from mamp.util import wrap

#from ford_msgs.msg import PedTrajVec, NNActions, PlannerMode, Clusters

#from obstacle import obstacle
from mamp.agents.obstacle import Obstacle
from mamp.ros_connect.ros_port import RosPort

class Agent(object):
    def __init__(self, name, radius, pref_speed, initial_heading, policy, dynamics_model, sensors,
    id,start_pos=None, goal_pos=None,group=0):
        self.policy = policy()
        self.dynamics_model = dynamics_model(self)
        self.sensors = [sensor() for sensor in sensors]

        # Store past selected actions
        self.chosen_action_dict = {}

        self.dic_start_pos = {'start': start_pos}
        self.dic_goal_pos = {'goal': goal_pos}
        self.dic_id = {'name': id}

        self.num_actions_to_store = 2
        self.action_dim = 2

        self.name = name
        self.id = id
        self.group = group
        self.dist_to_goal = 0.0
        self.near_goal_threshold = Config.NEAR_GOAL_THRESHOLD
        self.dt_nominal = Config.DT
        self.num_other_agents_observed = None
        range = getattr(Config, 'ENV_RANGE') if hasattr(Config, 'ENV_RANGE') else [[-20, 20], [-20, 20]]
        self.min_x = range[0][0]
        self.max_x = range[0][1]
        self.min_y = range[1][0]
        self.max_y = range[1][1]

        self.max_heading_change = np.pi / 3
        self.min_heading_change =  self.max_heading_change

        self.t_offset = None
        self.global_state_dim = 11
        self.ego_state_dim = 3

        self.planner = None
        self.path = []
        self.speed = [0, 0]     # vx, vy
        self.near_goal_reward = 0
        #ros
        self.ros_port = RosPort(self.name, self.id,)
        self.radianR = None
        self.radianP = None
        self.radianY = None
        self.goal_global_frame = None
        self.pos_global_frame = None
        self.action = None      #m
        self.other_agent_list = []


#        self.global_goal = PoseStamped()      #m
#        self.goal = PoseStamped()      #m
#        self.goal.pose.position.x = 0      #m
#        self.goal.pose.position.y = 0      #m
#        self.operation_mode = PlannerMode()      #m
#        self.operation_mode.mode = self.operation_mode.NN      #m
#        self.pose = PoseStamped()      #m
#        self.robot_pose = PoseStamped()
#        self.vel = Vector3()      #m
#        self.psi = 0.0      #m
#        self.num_poses = 0      #m
#        self.desired_action = np.zeros((2,))      #m
        self.d_min = 0.0      #m




#        self.reset(pos=start_pos, goal_pos=goal_pos, pref_speed=pref_speed, radius=radius, heading=initial_heading)
        self.reset(start_pos, goal_pos, pref_speed=pref_speed, radius=radius, heading=initial_heading)



    def reset(self, pos=None, goal_pos=None, pref_speed=None, radius=None, heading=None):
        """ Reset an agent with different states/goal, delete history and reset timer (but keep its dynamics, policy, sensors)

        :param px: (float or int) x position of agent in global frame at start of episode
        :param py: (float or int) y position of agent in global frame at start of episode
        :param gx: (float or int) desired x position of agent in global frame by end of episode
        :param gy: (float or int) desired y position of agent in global frame by end of episode
        :param pref_speed: (float or int) maximum speed of agent in m/s
        :param radius: (float or int) radius of circle describing disc-shaped agent's boundaries in meters
        :param heading: (float) angle of agent in global frame at start of episode

        """
        # Global Frame states
        if pos is not None:
            self.pos_global_frame = np.array(pos, dtype='float64')
        if goal_pos is not None:
            self.goal_global_frame = np.array(goal_pos, dtype='float64')
        else :
            self.goal_global_frame = None

        self.vel_global_frame = np.array([0.0, 0.0], dtype='float64')
        self.speed_global_frame = 0.0

        if heading is None and self.goal_global_frame is not None:
            vec_to_goal = self.goal_global_frame - self.pos_global_frame
            self.heading_global_frame = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        else:
            self.heading_global_frame = 0
#            self.heading_global_frame = heading
        self.delta_heading_global_frame = 0.0

        # Ego Frame states
        self.speed_ego_frame = 0.0
        self.heading_ego_frame = 0.0
        self.vel_ego_frame = np.array([0.0, 0.0])
        self.past_actions = np.zeros((self.num_actions_to_store, self.action_dim))

        # Other parameters
        if radius is not None:
            self.radius = radius
        if pref_speed is not None:
            self.pref_speed = pref_speed

#        self.straight_line_time_to_reach_goal = (np.linalg.norm(
#            self.pos_global_frame - self.goal_global_frame) - self.near_goal_threshold) / self.pref_speed
#        if Config.EVALUATE_MODE or Config.PLAY_MODE:
#            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
#        else:
#            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
#        self.time_remaining_to_reach_goal = max(self.time_remaining_to_reach_goal, self.dt_nominal)

        self.time_remaining_to_reach_goal = 100
        self.t = 0.0

        self.step_num = 0

        self.is_at_goal = False
        self.was_at_goal_already = False
        self.was_in_collision_already = False
        self.in_collision = False
        self.ran_out_of_time = False

        self.num_states_in_history = int(1.2 * self.time_remaining_to_reach_goal / self.dt_nominal)
        self.global_state_history = np.empty((self.num_states_in_history, self.global_state_dim))
        self.ego_state_history = np.empty((self.num_states_in_history, self.ego_state_dim))

        # self.past_actions = np.zeros((self.num_actions_to_store,2))
        self.past_global_velocities = np.zeros((self.num_actions_to_store, 2))
        self.past_global_velocities = self.vel_global_frame * np.ones((self.num_actions_to_store, 2))

        self.other_agent_states = np.zeros((Config.OTHER_AGENT_STATE_LENGTH,))
        self.other_agent_obs = np.zeros((Config.OTHER_AGENT_OBSERVATION_LENGTH,))

        self.dynamics_model.update_ego_frame()
        # self._update_state_history()
        # self._check_if_at_goal()
        # self.take_action([0.0, 0.0])

        self.min_dist_to_other_agents = np.inf

        self.turning_dir = 0.0
        self.is_done = False

    def __deepcopy__(self, memo):
        """ Copy every attribute about the agent except its policy (since that may contain MBs of DNN weights) """
        cls = self.__class__
        obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k != 'policy':
                setattr(obj, k, v)
        return obj

    def set_planner(self, planner):
        self.planner = planner
        path = self.planner.path()
        if len(path) > 1: self.path = path
        self.reset_states_in_history()

    def reset_states_in_history(self):
        if len(self.path) > 1:
            straight_line = 0
            for i in range(len(self.path) - 1):
                straight_line += np.linalg.norm(
                    np.array(self.path[i]) - np.array(self.path[i + 1])) - self.near_goal_threshold
            self.straight_line_time_to_reach_goal = straight_line / self.pref_speed
        else:
            self.straight_line_time_to_reach_goal = (np.linalg.norm(
                self.pos_global_frame - self.goal_global_frame) - self.near_goal_threshold) / self.pref_speed
        if Config.EVALUATE_MODE or Config.PLAY_MODE:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
        else:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
        self.time_remaining_to_reach_goal = max(self.time_remaining_to_reach_goal, self.dt_nominal)

        self.num_states_in_history = int(1.2 * self.time_remaining_to_reach_goal / self.dt_nominal)
        self.global_state_history = np.empty((self.num_states_in_history, self.global_state_dim))
        self.ego_state_history = np.empty((self.num_states_in_history, self.ego_state_dim))

    def _check_if_at_goal(self):
        """ Set :code:`self.is_at_goal` if norm(pos_global_frame - goal_global_frame) <= near_goal_threshold """


        if self.goal_global_frame is not None:
            distance_to_goal = (self.pos_global_frame[0] - self.goal_global_frame[0]) ** 2 + \
                               (self.pos_global_frame[1] - self.goal_global_frame[1]) ** 2
        else:
            distance_to_goal = 0

        distance_to_goal = np.sqrt(distance_to_goal)
        print("dis_to_goal:")
        print(distance_to_goal)

        if (distance_to_goal <= self.near_goal_threshold):      #m
            self.is_at_goal = True      #m
            return
        else:      #m
            self.is_at_goal = False
        is_near_goal = self.is_at_goal
        if is_near_goal:
            self.near_goal_reward = 0
        else:
            self.near_goal_reward = Config.REWARD_TO_GOAL_RATE * distance_to_goal  # -0.1*

    def set_state(self, px, py, vx=None, vy=None, heading=None):
        """ Without doing a full reset, update the agent's current state (pos, vel, heading).

        This is useful in conjunction with (:class:`~envs.dynamics.ExternalDynamics.ExternalDynamics`).
        For instance, if Agents in this environment should be aware of a real robot or one from a more realistic simulator (e.g., Gazebo),
        you could just call set_state each sim step based on what the robot/Gazebo tells you.

        If vx, vy not provided, will interpolate based on new&old position. Same for heading.

        Args:
            px (float or int): x position of agent in global frame right now
            py (float or int): y position of agent in global frame right now
            vx (float or int): x velocity of agent in global frame right now
            vy (float or int): y velocity of agent in global frame right now
            heading (float): angle of agent in global frame right now
        """
        if vx is None or vy is None:
            if self.step_num == 0:
                # On first timestep, just set to zero
                self.vel_global_frame = np.array([0, 0])
            else:
                # Interpolate velocity from last pos
                self.vel_global_frame = (np.array([px, py]) - self.pos_global_frame) / self.dt_nominal
        else:
            self.vel_global_frame = np.array([vx, vy])

        if heading is None:
            # Estimate heading to be the direction of the velocity vector
            heading = np.arctan2(self.vel_global_frame[1], self.vel_global_frame[0])
            self.delta_heading_global_frame = wrap(heading - self.heading_global_frame)
        else:
            self.delta_heading_global_frame = wrap(heading - self.heading_global_frame)

#        self.pos_global_frame = np.array([px, py])
        self.speed_global_frame = np.linalg.norm(self.vel_global_frame)
        self.heading_global_frame = heading

    def find_next_action(self, dict_obs, actions, dict_comm=None):
        print("find next action")
        self.pos_global_frame = self.ros_port.getRobotPose()
        self.radianR, self.radianP, self.radianY = self.ros_port.getEulerRadian()

        if self.ros_port.new_goal_received:
            self.ros_port.new_goal_received = False
            self.goal_global_frame = self.ros_port.getGoalPose()
        self._check_if_at_goal()
        if not self.is_at_goal:
            if self.policy.type == "external":
                action = self.policy.external_action_to_action(self, actions[self.id])
            elif self.policy.type == "internal":
                if Config.WITH_COMM:
                    action = self.policy.find_next_action(dict_obs, dict_comm, self)
                else:
                    action = self.policy.find_next_action(dict_obs, self)
            elif self.policy.type == "mixed":
                action = self.policy.produce_next_action(dict_obs, self, actions)
            else:
                raise NotImplementedError

    #        if self.dynamics_model.action_type == "R_THETA":
    #            if action[0] > self.pref_speed: action[0] = self.pref_speed
    #            if action[1] > self.max_heading_change: action[1] = self.max_heading_change
    #            if action[1] < self.min_heading_change: action[1] = self.min_heading_change

            return action
        else :
            action = np.array([0, 0])
            return action

    def do_action(self, action, dt):

        # Store past actions
        self.past_actions = np.roll(self.past_actions, 1, axis=0)
        self.past_actions[0, :] = action
        self.action = action      #m
        self.ros_port.pubTwist(action)
#        twist = Twist()      #m
#        twist.angular.z = action[1]      #m
#        twist.linear.x = action[0]      #m
#        print("linear.x:")
#        print(twist.linear.x)
#        print("angular.z:")
#        print(twist.angular.z)
#        self.pub_twist.publish(twist)      #m

        # Store info about the TF btwn the ego frame and global frame before moving agent
        if self.goal_global_frame is not None:
            goal_direction = self.goal_global_frame - self.pos_global_frame
            theta = np.arctan2(goal_direction[1], goal_direction[0])
        else :
            theta = np.arctan2(0, 0)
        self.T_global_ego = np.array([[np.cos(theta), -np.sin(theta), self.pos_global_frame[0]],
                                      [np.sin(theta), np.cos(theta), self.pos_global_frame[1]],
                                      [0, 0, 1]])
        self.ego_to_global_theta = theta

        # In the case of ExternalDynamics, this call does nothing,
        # but set_state should have been called instead
        self.dynamics_model.step(action, dt)

        self.dynamics_model.update_ego_frame()

        self._update_state_history()

        self._check_if_at_goal()

        self._store_past_velocities()

        # Update time left so agent does not run around forever
        self.time_remaining_to_reach_goal -= dt
        self.t += dt
        self.step_num += 1

        if self.time_remaining_to_reach_goal <= 0.0:  # and Config.TRAIN_MODE:
            self.ran_out_of_time = True
        return

    def take_action(self, action, dt):
        """ If not yet done, take action for dt seconds, check if done.

        Args:
            action (list): nominally a [delta heading angle, speed] command for this agent (but probably could be anything that the dynamics_model.step can handle)
            dt (float): time in seconds to execute :code:`action`

        """
        # Agent is done if any of these conditions hold (at goal, out of time, in collision). Stop moving if so & ignore the action.


        if self.is_at_goal or self.ran_out_of_time or self.in_collision:
            if self.is_at_goal:
                self.was_at_goal_already = True
#                self.ros_port.stop_moving()

            if self.in_collision:
                self.was_in_collision_already = True
#                self.ros_port.stop_moving()
            self.vel_global_frame = np.array([0.0, 0.0])
            self._store_past_velocities()
            return
        else:
            self.do_action(action, dt)
        return

    def sense(self, agents, agent_index, top_down_map):
        """ Call the sense method of each Sensor in self.sensors, store in self.sensor_data dict keyed by sensor.name.

        Args:
            agents (list): all :class:`~gym_collision_avoidance.envs.agent.Agent` in the environment
            agent_index (int): index of this agent (the one with this sensor) in :code:`agents`
            top_down_map (2D np array): binary image with 0 if that pixel is free space, 1 if occupied

        """
        self.sensor_data = {}
        for sensor in self.sensors:
            sensor_data = sensor.sense(agents, agent_index, top_down_map)
            self.sensor_data[sensor.name] = sensor_data

    def _update_state_history(self):
        global_state, ego_state = self.to_vector()
        self.global_state_history[self.step_num, :] = global_state
        self.ego_state_history[self.step_num, :] = ego_state

    def print_agent_info(self):
        """ Print out a summary of the agent's current state. """
        print('----------')
        print('Global Frame:')
        print('(px,py):', self.pos_global_frame)
        print('(gx,gy):', self.goal_global_frame)
        print('(vx,vy):', self.vel_global_frame)
        print('speed:', self.speed_global_frame)
        print('heading:', self.heading_global_frame)
        print('Body Frame:')
        print('(vx,vy):', self.vel_ego_frame)
        print('heading:', self.heading_ego_frame)
        print('----------')

    def to_vector(self):
        """ Convert the agent's attributes to a single global state vector. """
        if self.goal_global_frame is None:
            goal_pose = np.array([0, 0])
        else :
            goal_pose = self.goal_global_frame

        global_state = np.array([self.t,
                                 self.pos_global_frame[0],
                                 self.pos_global_frame[1],
                                 goal_pose[0],
                                 goal_pose[1],
                                 self.radius,
                                 self.pref_speed,
                                 self.vel_global_frame[0],
                                 self.vel_global_frame[1],
                                 self.speed_global_frame,
                                 self.heading_global_frame])

        ego_state = np.array([self.t, self.dist_to_goal, self.heading_ego_frame])
        return global_state, ego_state

    def get_communication_dict(self):
        communication = {}
        for comm in Config.COMM_OBJ:
            communication[comm] = self.get_agent_data(comm)
        return communication

    def get_sensor_data(self, sensor_name):
        """ Extract the latest measurement from the sensor by looking up in the self.sensor_data dict (which is populated by the self.sense method.

        Args:
            sensor_name (str): name of the sensor (e.g., 'laserscan', I think from Sensor.str?)

        """
        if sensor_name in self.sensor_data:
            return self.sensor_data[sensor_name]

    def get_agent_data(self, attribute):
        """ Grab the value of self.attribute (useful to define which states sensor uses from config file).
        Args:
            attribute (str): which attribute of this agent to look up (e.g., "pos_global_frame")
        """
        return getattr(self, attribute)

    def get_agent_data_equiv(self, attribute, value):
        """ Grab the value of self.attribute and return whether it's equal to value (useful to define states sensor uses from config file).

        Args:
            attribute (str): which attribute of this agent to look up (e.g., "radius")
            value (anything): thing to compare self.attribute to (e.g., 0.23)

        Returns:
            result of self.attribute and value comparison (bool)

        """
        return eval("self." + attribute) == value

    def get_observation_dict(self):
        observation = {}
        for state in Config.STATES_IN_OBS_MULTI[self.group]:
            observation[state] = np.array(eval("self." + Config.STATE_INFO_DICT[state]['attr']))
        return observation

    def get_ref(self):
        """ Using current and goal position of agent in global frame, compute coordinate axes of ego frame.

        Ego frame is defined as: origin at center of agent, x-axis pointing from agent's center to agent's goal (right-hand rule, z axis upward).
        This is a useful representation for goal-conditioned tasks, since many configurations of agent-pos-and-goal in the global frame map to the same ego setup.

        Returns:
        2-element tuple containing

        - **ref_prll** (*np array*): (2,) with vector corresponding to ego-x-axis (pointing from agent_position->goal)
        - **ref_orth** (*np array*): (2,) with vector corresponding to ego-y-axis (orthogonal to ref_prll)

        """
        if self.goal_global_frame is not None:
            goal_direction = self.goal_global_frame - self.pos_global_frame
        else:
            goal_direction = np.array([0,0])

        self.goal_direction = goal_direction
        self.dist_to_goal = math.sqrt(goal_direction[0] ** 2 + goal_direction[1] ** 2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / self.dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg
        return ref_prll, ref_orth

    def _store_past_velocities(self):
        self.past_global_velocities = np.roll(self.past_global_velocities, 1, axis=0)
        self.past_global_velocities[0, :] = self.vel_global_frame

    def ego_pos_to_global_pos(self, ego_pos):
        """ Convert a position in the ego frame to the global frame.
        This might be useful for plotting some of the perturbation stuff.
        Args:
            ego_pos (np array): if (2,), it represents one (x,y) position in ego frame
                if (n,2), it represents n (x,y) positions in ego frame
        Returns:
            global_pos (np array): either (2,) (x,y) position in global frame or (n,2) n (x,y) positions in global frame
        """
        if ego_pos.ndim == 1:
            ego_pos_ = np.array([ego_pos[0], ego_pos[1], 1])
            global_pos = np.dot(self.T_global_ego, ego_pos_)
            return global_pos[:2]
        else:
            ego_pos_ = np.hstack([ego_pos, np.ones((ego_pos.shape[0], 1))])
            global_pos = np.dot(self.T_global_ego, ego_pos_.T).T
            return global_pos[:, :2]

    def global_pos_to_ego_pos(self, global_pos):
        """ Convert a position in the global frame to the ego frame.

        Args:
            global_pos (np array): one (x,y) position in global frame

        Returns:
            ego_pos (np array): (2,) (x,y) position in ego frame

        """
        ego_pos = np.dot(np.linalg.inv(self.T_global_ego), np.array([global_pos[0], global_pos[1], 1]))
        return ego_pos[:2]
#ros function
    def find_vmax(self, d_min, heading_diff):
        # Calculate maximum linear velocity, as a function of error in
        # heading and clear space in front of the vehicle
        # (With nothing in front of vehicle, it's not important to
        # track MPs perfectly; with an obstacle right in front, the
        # vehicle must turn in place, then drive forward.)
        d_min = max(0.0,d_min)
        x = 0.3
        margin = 0.3
        # y = max(d_min - 0.3, 0.0)
        y = max(d_min, 0.0)
        # making sure x < y
        if x > y:
            x = 0
        w_max = 1
        # x^2 + y^2 = (v_max/w_max)^2
        v_max = w_max * np.sqrt(x**2 + y**2)
        v_max = np.clip(v_max,0.0,self.pref_speed)
        # print 'V_max, x, y, d_min', v_max, x, y, d_min
        if abs(heading_diff) < np.pi / 18:
            return self.pref_speed
        return v_max

    def set_other_agents(self, host_id, agents):
        host_agent = agents[host_id]
        self.obstacle_list = [None for _ in range(len(agents))]
        for i, other_agent in enumerate(agents):
            if i == host_id:
                continue
            self.other_agent_list.append(other_agent)
            self.obstacle_list[other_agent.id] = Obstacle(pos = other_agent.pos_global_frame, shape_dict = {'shape' : 'circle', 'feature' : other_agent.radius})



#    def set_obstacles(self):
#        for i, other_agent in enumerate(self.other_agent_list):
#            self.obstacle_list.append[other_agent.id] = Obstacle(pos = obstacle.pos_global_frame, shape_dict = {'shape' : 'circle', 'feature' : obstacle.radius})



