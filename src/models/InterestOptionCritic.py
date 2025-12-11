import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class InterestOptionCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_options, interest_dim, hidden_dim=128, lr=1e-3, gamma=0.99, tau=0.01):
        super(InterestOptionCritic, self).__init__()
        
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

    def forward_interest(self, state):
        interest = self.interest_net(state)
        return interest
    
    def forward_option_policy(self, state, interest, option):
        input_tensor = torch.cat([state, interest], dim=-1)
        action_probs = self.option_policy_net[option](input_tensor)
        return action_probs

    def forward_critic(self, state, interest, action, option):
        input_tensor = torch.cat([state, interest, action], dim=-1)
        q_value = self.critic_net[option](input_tensor)
        return q_value
    
    def get_target_critic_value(self, state, interest, action, option):
        input_tensor = torch.cat([state, interest, action], dim=-1)
        q_value = self.target_critic_net[option](input_tensor)
        return q_value
    
    def update_target_networks(self):
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        input_tensor = torch.cat([state, interest, action], dim=-1)
        q_value = self.critic_net[option](input_tensor)
        return q_value
    
    def get_option_policy(self, state, interest, option):
        input_tensor = torch.cat([state, interest], dim=-1)
        return self.option_policy_net[option](input_tensor)
    
    def get_critic_value(self, state, interest, action, option):
        input_tensor = torch.cat([state, interest, action], dim=-1)
        return self.critic_net[option](input_tensor)
    
    def get_target_critic_value(self, state, interest, action, option):
        input_tensor = torch.cat([state, interest, action], dim=-1)
        return self.target_critic_net[option](input_tensor)
    