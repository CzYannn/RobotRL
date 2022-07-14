import os
import torch as T
import torch.nn.functional as F
import numpy as np
from .buffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork, ValueNetwork

class SAC():
    def __init__(self, id, alpha=0.0003, beta=0.0003, input_dims=[14],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=15):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.id = id
        self.actor = ActorNetwork(alpha=alpha, input_dims=np.array(input_dims)-1, n_actions=n_actions,
                    name='actor', max_action=1, id=id)
        self.critic_1 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions,
                    name='critic_1', id=id)
        self.critic_2 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions,
                    name='critic_2', id=id)
        self.value = ValueNetwork(beta=beta, input_dims=input_dims, name='value', id=id)
        self.target_value = ValueNetwork(beta=beta, input_dims=input_dims, name='target_value', id=id)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        #record loss
        self.value_loss_list = np.array([])
        self.critic_loss_list = np.array([])
        self.actor_loss_list = np.array([])

    def choose_action(self, observation, warmup=False, evaluate=False, action=6):
        observation = observation[:-1]
        state = T.Tensor([observation]).to(self.actor.device)
        if evaluate:
            return self.actor.sample_mu(state).cpu().detach().numpy()[0]
        if warmup and self.memory.mem_cntr<=self.batch_size:
            actions =  (T.rand(1,action)-0.5)*2
        else:
            actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def get_avg_loss(self):
        """
        计算loss的平均值
        """
        avg_pi_loss = np.mean(self.actor_loss_list)
        avg_q_loss = np.mean(self.critic_loss_list)
        avg_v_loss = np.mean(self.value_loss_list)
        return avg_pi_loss, avg_q_loss, avg_v_loss

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)
        actor_state = T.zeros(self.batch_size,self.input_dims[0]-1).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        for i in range(len(state)):
            actor_state[i] = state[i][:-1]

        actions, log_probs = self.actor.sample_normal(actor_state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(actor_state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        #record loss 
        critic_loss = critic_loss.cpu().detach().numpy()
        value_loss = value_loss.cpu().detach().numpy()
        actor_loss = actor_loss.cpu().detach().numpy()
        self.critic_loss_list = np.append(self.critic_loss_list,critic_loss)
        self.actor_loss_list = np.append(self.actor_loss_list, actor_loss)
        self.value_loss_list = np.append(self.value_loss_list, value_loss)


