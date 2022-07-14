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


def build_agents(leader_policy, follower_policy_list):
    pref_speed = 2.0
    agents = []
    leader = Leader(
                    name = 'Agent1',
                    pref_speed=pref_speed,
                    policy=leader_policy,
                    id=0,
                    RL=False)
    agents.append(leader)
    for i in range(args.agent_num-1):
        follower = Follower(
                            name = 'Agent'+str(i+2),
                            pref_speed=pref_speed,
                            policy=follower_policy_list[i],
                            id=i+1,
                            RL=True)
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
                        help='reward_scale (default:15)')
    parser.add_argument('--train', type=str2bool, default=True, metavar='N',
                        help='if train (default:True)')
    parser.add_argument('--episode', type=int, default=1000, metavar='N',
                        help='episode (default:1000)')
    parser.add_argument('--load_L', type=str2bool, default=False, metavar='N',
                        help='if load leader_model(default:False)')
    parser.add_argument('--load_F', type=str2bool, default=False, metavar='N',
                        help='if load follower_model(default:False)')
    parser.add_argument('--warmup',type=str2bool, default=False, metavar='N',
                        help='if warmup (default:False')
    parser.add_argument('--RL',type=str2bool, default=True,metavar='N',
                        help = 'if use RL(defaul:True)')
    parser.add_argument('--action',type=int, default=6, metavar='N',
                        help='action number(default=6)')
    parser.add_argument('--seed',type=int, default=123456, metavar='N',
                    help='action number(default=123456)')
    parser.add_argument('--agent_num',type=int, default=3, metavar='N',
                        help='agent number(default=4)')                    
    args = parser.parse_args()

    #set random seed 
    T.manual_seed(args.seed)
    np.random.seed(args.seed)
    T.cuda.manual_seed(args.seed)

    load_L = args.load_L
    load_F = args.load_F
    train_mode = args.train
    episode = args.episode
    # 激活环境检测节点
    rospy.wait_for_service('/Activate')
    service_call = rospy.ServiceProxy('/Activate', StartUp)
    response = service_call(True)
    print(response)

    follower_policy_list = []
    leader_policy = SAC(alpha=args.lr,beta=args.lr,n_actions=args.action,gamma=args.gamma,
                reward_scale=args.reward_scale, layer1_size=args.hidden_size,
                layer2_size=args.hidden_size, tau=args.tau,batch_size=args.batch_size, id=0)
    for i in range(args.agent_num-1):
        follower_policy_list.append(SAC(alpha=args.lr,beta=args.lr,gamma=args.gamma, input_dims=[7],
                reward_scale=args.reward_scale, layer1_size=args.hidden_size,n_actions=2,
                layer2_size=args.hidden_size, tau=args.tau,batch_size=args.batch_size, id=i+1))

    agents = build_agents(leader_policy, follower_policy_list)
    env = gym.make('LineFollower-v0')
    # neighbor_info = [
    #                 {1: [-1, 1], 2:[-1, -1]},
    #                 {0: [-1, 1], 4:[2, 0]},
    #                 {0: [-1, -1], 3:[2, 0]},
    #                 {2: [-2, 0], 4: [0, -2]},
    #                 {1: [-2, 0], 3: [0, 2]},
    #                 ]
    neighbor_info = [
                    {0:[0, 0]},
                    {0:[-1, 1]},
                    {0:[-1, -1]},
                    # {0:[-2, 0]},
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
    if load_L:
        agents[0].policy.load_models()
    if load_F:
        for ag in agents:
            if ag.group == 1:
                ag.policy.load_models()

    score_history = [[] for i in range(len(agents))]
    score_save = [[] for i in range(len(agents))]
    mean_error = [0 for i in range(len(agents))]
    best_score = [0 for i in range(len(agents))]
    avg_score = [0 for i in range(len(agents))]

    # training loop

    for i in range(episode):
        obs = env.reset()
        scores = np.zeros(len(agents))
        step = 0 
        while not env.game_over:
            all_commands = []
            all_actions = []
            for idx, ag in enumerate(agents):
                if ag.group == 0 :
                    action = ag.act(obs[idx], warmup = args.warmup, 
                                    eval = args.load_F, action_num = args.action)
                else:
                    action = ag.act(obs[idx], warmup = args.warmup, 
                                    eval = args.load_L, action_num = 2)
                all_actions.append(action)
            obs_, rewards, dones, _ = env.step(all_actions)
            for idx, ag in enumerate(agents):
                if not ag.quit :
                    scores[idx] += rewards[idx]
                    ag.policy.remember(obs[idx], all_actions[idx], rewards[idx], obs_[idx], dones[idx])
                    if train_mode:
                        ag.policy.learn()
            obs = obs_
            step +=1 

            # record info
        for idx, ag in enumerate(agents):        
            score_history[idx].append(scores[idx])
            score_save[idx].append(scores[idx])
            avg_score[idx] = np.mean(score_history[idx][-50:])
            if best_score[idx] <= avg_score[idx] and train_mode and i>=50:
                best_score[idx] = avg_score[idx]
                ag.policy.save_models()
            print(f'{ag.name}, episode ={i}, score = {scores[idx]:.2f}, avg_score = {avg_score[idx]:.2f}, step: {step}')
            # record loss and score
            if load_F==True:
                with open(f'data/test/point_data_{idx}.txt', 'w') as file_object:
                    file_object.write(str(ag.pose_list)+'\n')
                with open(f'data/test/error_data_{idx}.txt','w') as file_object:
                    file_object.write(str(ag.error_list)+'\n')
            if args.RL == False:
                with open(f'data/test/pid_error_data_{idx}.txt','w') as file_object:
                    file_object.write(str(ag.error_list)+'\n')

            if train_mode == True:
                if (i+1) % 10 == 0:
                    with open(f'data/train/score_data{idx}.txt','a') as file_object:
                        file_object.write(str(score_save[idx])+'\n')
                        score_save[idx] = []
                    with open(f'data/train/value_loss_list{idx}.txt','a') as file_object:
                        file_object.write(str(agents[idx].policy.value_loss_list)+'\n')
                    with open(f'data/train/actor_loss_list{idx}.txt','a') as file_object:
                        file_object.write(str(agents[idx].policy.actor_loss_list)+'\n')
                    with open(f'data/train/critic_loss_list{idx}.txt','a') as file_object:
                        file_object.write(str(agents[idx].policy.critic_loss_list)+'\n')
                    agents[idx].policy.value_loss_list=[]
                    agents[idx].policy.actor_loss_list=[]
                    agents[idx].policy.critic_loss_list=[]
        if train_mode == True:
            if (i+1) % 50 == 0:
                with open(f'data/train/success_record.txt','a') as file_object:
                    file_object.write(str(env.success_record)+'\n')
                    env.success_record=[]







