import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class InterestOptionCriticContrastive(nn.Module):
    def __init__(self, state_dim, action_dim, num_options, interest_dim, hidden_dim=128, lr=1e-3, gamma=0.99, tau=0.01):
        super(InterestOptionCriticContrastive, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.interest_dim = interest_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        
        # Interest Network
        self.interest_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, interest_dim)
        )
        
        # Option Policy Network
        self.option_policy_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + interest_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(num_options)
        ])
        
        # Critic Network
        self.critic_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + interest_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_options)
        ])
        
        # Target Networks
        self.target_critic_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + interest_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_options)
        ])
        
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.interest_optimizer = optim.Adam(self.interest_net.parameters(), lr=lr)
        self.option_policy_optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.option_policy_net]
        self.critic_optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.critic_net]
        
    def forward(self, state):
        interest = self.interest_net(state)
        return interest
    
    def get_option_policy(self, state, interest, option):
        input_tensor = torch.cat([state, interest], dim=-1)
        return self.option
    
    def get_critic_value(self, state, interest, action, option):
        input_tensor = torch.cat([state, interest, action], dim=-1)
        return self.critic_net[option](input_tensor)
    
    def update_critic(self, state, action, reward, next_state, done, option):
        interest = self.interest_net(state).detach()
        next_interest = self.interest_net(next_state).detach()
        
        current_q = self.get_critic_value(state, interest, action, option)
        
        with torch.no_grad():
            next_action_probs = self.option_policy_net[option](torch.cat([next_state, next_interest], dim=-1))
            next_actions = torch.multinomial(next_action_probs, 1)
            next_action_one_hot = F.one_hot(next_actions.squeeze(-1), num_classes=self.action_dim).float()
            next_q = self.target_critic_net[option](torch.cat([next_state, next_interest, next_action_one_hot], dim=-1))
            target_q = reward + (1 - done) * self.gamma * next_q
        
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizers[option].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[option].step()
        
        return critic_loss.item()
    
    def update_option_policy(self, state, option):
        interest = self.interest_net(state).detach()
        input_tensor = torch.cat([state, interest], dim=-1)
        
        action_probs = self.option_policy_net[option](input_tensor)
        actions = torch.multinomial(action_probs, 1)
        action_one_hot = F.one_hot(actions.squeeze(-1), num_classes=self.action_dim).float()
        
        q_value = self.get_critic_value(state, interest, action_one_hot, option)
        policy_loss = -q_value.mean()
        
        self.option_policy_optimizers[option].zero_grad()
        policy_loss.backward()
        self.option_policy_optimizers[option].step()
        
        return policy_loss.item()
    
    def update_interest(self, state, option):
        interest = self.interest_net(state)
        input_tensor = torch.cat([state, interest], dim=-1)
        
        action_probs = self.option_policy_net[option](input_tensor)
        actions = torch.multinomial(action_probs, 1)
        action_one_hot = F.one_hot(actions.squeeze(-1), num_classes=self.action_dim).float()
        
        q_value = self.get_critic_value(state, interest, action_one_hot, option)
        interest_loss = -q_value.mean()
        
        self.interest_optimizer.zero_grad()
        interest_loss.backward()
        self.interest_optimizer.step()
        
        return interest_loss.item()
    
    def soft_update_target_networks(self):
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_net, net in zip(self.target_critic_net, self.critic_net):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)