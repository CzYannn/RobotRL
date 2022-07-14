#!/usr/env/bin python

import math
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point, PoseWithCovarianceStamped

class AgentPort():
    def __init__(self, id, name):
        self.id   = id
        self.name = name
        self.taken_action = None
        self.pos_global_frame = None
        self.radianR = None
        self.radianP = None
        self.radianY = None
        self.angleR = None
        self.angleP = None
        self.angleY = None
        self.goal_global_frame = None
        self.new_goal_received = False

        self.str_pub_twist = '/' + self.name + '/cmd_vel'
        self.pub_twist = rospy.Publisher(self.str_pub_twist,Twist,queue_size=1)

        self.str_pose = '/' + self.name + '/pose'
        self.str_odom = '/' + self.name + '/odom'
        self.str_goal = '/' + self.name + '/move_base_simple/goal'
        self.str_robot_pose = '/' + self.name + '/robot_pose'
        self.sub_odom = rospy.Subscriber(self.str_odom, Odometry, self.cbOdom)
#        self.sub_pose = rospy.Subscriber(self.str_pose,PoseWithCovarianceStamped,self.cbPose)
        # self.sub_robot_pose = rospy.Subscriber(self.str_robot_pose, PoseStamped, self.cbRobotPose)
#        self.sub_other_robot_pose1 = rospy.Subscriber('other_robot_pose1',PoseStamped, self.cbRobotPose1)
#        self.sub_other_robot_pose2 = rospy.Subscriber('other_robot_pose2',PoseStamped, self.cbRobotPose2)
#        self.obstacle_list = [None for _ in range(len(other_agent_list))]
#        for i, agent in other_agent_list:
#            sub_val_name = 'self.sub'+agent.name+'Pose'
#            sub_name = '/' + agent.name + '/robot_pose'
#            sub_fun_name = 'self.' + cbRobotPose + agent.id

#            def eval(sub_fun_name)(self,msg):
#                self.obstacle_list[i] = Obstacle(pos = np.array([msg.pose.position.x, msg.pose.position.y]),
#                shape_dict = { 'shape' : "circle", 'feature' : 0.2})
#            eval(sub_val_name) = rospy.Subscriber(sub_name, PoseStamped, sub_fun_name)

    def cbOdom(self, msg):
        self.odom = msg
        self.pos_global_frame = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.radianR = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        self.radianP = math.asin(2 *  (qw * qy - qz * qx))
        self.radianY = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz * qz + qy * qy))

        self.angleR = self.radianR * 180 / math.pi # 横滚角
        self.angleP = self.radianP * 180 / math.pi # 俯仰角
        self.angleY = self.radianY * 180 / math.pi # 偏航角

        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z

        self.vel_global_frame = [math.cos(self.angleY), math.sin(self.angleY)] * self.v_x 


    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)

    def pubTwist(self, action, dt):
        self.taken_action = action
        twist = Twist()
        twist.linear.x  = action[0]
        twist.angular.z = action[1] / dt
        self.pub_twist.publish(twist)

#    def cbRobotPose1(self, msg):
#        self.obstacle_list[0] = Obstacle(pos = np.array([msg.pose.position.x, msg.pose.position.y]), shape_dict = { 'shape' : "circle", 'feature' : 0.2})

#    def cbRobotPose2(self, msg):
#        self.obstacle_list[1] = Obstacle(pos = np.array([msg.pose.position.x, msg.pose.position.y]), shape_dict = { 'shape' : "circle", 'feature' : 0.2})

    def getRobotPose(self):
        return self.pos_global_frame

    def getEulerRadian(self):
        return self.radianR, self.radianP, self.radianY

    def getGoalPose(self):
        return self.goal_global_frame

#     def set_other_agents(self, agents):
#         host_agent = agents[self.agent.id]
# #        self.obstacle_list = [None for _ in range(len(agents))]
#         self.other_agent_list = []
#         for i, other_agent in enumerate(agents):
#             if i == self.agent.id:
#                 continue
#             self.other_agent_list.append(other_agent)
# #            self.obstacle_list[other_agent.id] = Obstacle(pos = other_agent.pos_global_frame, shape_dict = {'shape' : 'circle', 'feature' : other_agent.radius})
