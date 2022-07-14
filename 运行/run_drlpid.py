#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.getcwd())) ## add path 
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os 
os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ['GYM_CONFIG_PATH'] = '../mamp/configs/config.py'
import gym 
import numpy as np 
import rospy 
from algos.sac.sac_torch import SAC
from mrobot.srv import StartUp, StartUpRequest # service call
from mamp.util import str2bool
import argparse
import torch as T
from mamp.agents.leader import Leader
from mamp.agents.follower import Follower


def build_agents(leadr_policy, follower_policy):
    pref_speed = 1.0
    agents = [
        Leader(
                name = 'Agent1',
                pref_speed=pref_speed,
                policy=leadr_policy,
                id=0),]
    for i in range(2):
        follower = Follower(
                            name = 'Agent'+str(i+2),
                            pref_speed=pref_speed,
                            policy=follower_policy,
                            id=i+1)
        agents.append(follower)
    return agents

if __name__ == '__main__':
    #parameter setting
    parser = argparse.ArgumentParser(description='track_car')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(tao) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--reward_scale', type=int, default=15, metavar='N',
                        help='reward_scale (default:10)')
    parser.add_argument('--train', type=str2bool, default=True, metavar='N',
                        help='if train (default:True)')
    parser.add_argument('--episode', type=int, default=1000, metavar='N',
                        help='episode (default:1000)')
    parser.add_argument('--load', type=str2bool, default=False, metavar='N',
                        help='if load (default:False)')
    parser.add_argument('--warmup',type=str2bool, default=False, metavar='N',
                        help='if warmup (default:False')
    parser.add_argument('--RL',type=str2bool, default=True,metavar='N',
                        help = 'if use RL(defaul:True)')
    parser.add_argument('--action',type=int, default=6, metavar='N',
                        help='action number(default=6)')
    parser.add_argument('--seed',type=int, default=123456, metavar='N',
                    help='action number(default=123456)')
    args = parser.parse_args()

    #set random seed 
    T.manual_seed(args.seed)
    np.random.seed(args.seed)
    T.cuda.manual_seed(args.seed)

    load_point = args.load
    train_mode = args.train
    episode = args.episode
    # 激活环境检测节点
    rospy.wait_for_service('/Activate')
    service_call = rospy.ServiceProxy('/Activate', StartUp)
    response = service_call(True)
    print(response)

    leader_policy = SAC(alpha=args.lr,beta=args.lr,n_actions=args.action,gamma=args.gamma,
                reward_scale=args.reward_scale, layer1_size=args.hidden_size,
                layer2_size=args.hidden_size, tau=args.tau,batch_size=args.batch_size)
    
    follower_policy = SAC(alpha=args.lr,beta=args.lr,gamma=args.gamma, input_dims=[5],
                reward_scale=args.reward_scale, layer1_size=args.hidden_size,n_actions=2,
                layer2_size=args.hidden_size, tau=args.tau,batch_size=args.batch_size)

    agents = build_agents(leader_policy, follower_policy)
    
    env = gym.make('LineFollower-v0')
    # neighbor_info = [
    #                 {1: [-1, 1], 2:[-1, -1]},
    #                 {0: [-1, 1], 4:[2, 0]},
    #                 {0: [-1, -1], 3:[2, 0]},
    #                 {2: [-2, 0], 4: [0, -2]},
    #                 {1: [-2, 0], 3: [0, 2]},
    #                 ]
    neighbor_info = [
                    {0: [0, 0]},
                    {0: [-1, 1]},
                    {0: [-1, -1]},
                    ]
    env.set_agents(agents, neighbor_info)
    
    def shutdown():
        """
        shutdown properly
        """
        print("shutdown!")
        env.stop()
        service_call(False)

    rospy.on_shutdown(shutdown)
    if load_point:
        agents[0].policy.load_models()

    score_history = []
    score_save = []
    mean_error = []
    best_score = 0

    for i in range(episode):
        ob = env.reset()
        done = False
        score = 0
        step = 0 
        while not done:
            action, command = agents[0].find_next_action(ob, warmup = args.warmup, 
                            eval = args.load, action_num = args.action, RL=args.RL)
            ob_, reward, done, _ = env.step(command)
            score += reward 
            agents[0].policy.remember(ob, action, reward, ob_, done)
            ob = ob_
            step +=1 
            if train_mode:
                agents[0].policy.learn()

        score_history.append(score)
        score_save.append(score)
        avg_score = np.mean(score_history[-50:])
        if best_score <= avg_score and train_mode and i>=50:
            best_score = avg_score
            agents[0].policy.save_models()

        print('episode = ', i, 'score = %.2f' % score, 'avg_score = %.2f' % avg_score,'step: ',step)
        if (i+1) % 50 == 0:
            with open('data/train/score_data.txt','a') as file_object:
                file_object.write(str(score_save)+'\n')
            with open('data/train/success_record.txt','a') as file_object:
                file_object.write(str(env.success_record)+'\n')
            score_save=[]
            env.success_record=[]
        # record lose
        if (i+1) % 10 == 0 :
            with open('data/train/value_loss_list.txt','a') as file_object:
                file_object.write(str(agents[0].policy.value_loss_list)+'\n')
            with open('data/train/actor_loss_list.txt','a') as file_object:
                file_object.write(str(agents[0].policy.actor_loss_list)+'\n')
            with open('data/train/critic_loss_list.txt','a') as file_object:
                file_object.write(str(agents[0].policy.critic_loss_list)+'\n')
            agents[0].policy.value_loss_list=[]
            agents[0].policy.actor_loss_list=[]
            agents[0].policy.critic_loss_list=[]







