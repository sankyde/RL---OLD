
import torch
import torch.nn as nn
import torch.multiprocessing
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from copy import deepcopy
LOG_STD_MAX=2
LOG_STD_MIN=-20



# class CNNetwork(nn.Module):
#     def __init__(self,input_shape,output_shape,hidden_dim=128):
#         super(CNNetwork, self).__init__()
#



class _NeuralNetwork(nn.Module):
    def __init__(self,input_shape,output_shape,act_limit,hidden_dim):
        super().__init__()

        self.model=nn.ModuleList()

        assert isinstance(input_shape,int) and isinstance(output_shape,int)
        self.net=nn.Sequential(
            nn.Linear(input_shape,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer=nn.Linear(hidden_dim,output_shape)
        self.log_std_layer=nn.Linear(hidden_dim,output_shape)
        self.act_limit=act_limit

    def forward(self,obs,with_logprob=True):
        net_out=self.net(obs)
        mu=self.mu_layer(net_out)
        log_std=self.log_std_layer()
        log_std=torch.clamp(log_std,LOG_STD_MIN,LOG_STD_MAX)
        std=torch.exp(log_std)

        dist=Normal(mu,std)

        action=dist.rsample()

        if with_logprob:

            logp_pi= dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2*(np.log(2)-action-F.softplus(-2 * action))).sum(axis=1)
        else:
            logp_pi=None

        action=torch.tanh(action)
        action=self.act_limit * action

        return action,logp_pi

class _Q_function(nn.Module):
    def __init__(self,input_shape,action_dim,hidden_dim):
        super().__init__()

        self.Q=nn.Sequential(
            nn.Linear(input_shape+action_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
        )

    def forward(self,obs,act):
        q=self.Q(torch.cat([obs,act],dim=-1))
        return torch.squeeze(q,-1)

class ReplayBuffer:
    def __init__(self,obs_dim,act_dim,size):
        self.obs_states=np.zeros((size,obs_dim),dtype=np.float32)
        self.obs_nstates=np.zeros((size,obs_dim),dtype=np.float32)
        self.actions=np.zeros((size,act_dim),dtype=np.float32)
        self.rewards=np.zeros(size,dtype=np.float32)
        self.dones=np.zeros(size,dtype=np.float32)
        self.ptr,self.size,self.max_size=0,0,size

    def store(self,obs,act,rew,next_obs,done):
        self.obs_states[self.ptr] = obs
        self.obs_nstates[self.ptr] = next_obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.dones[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1,self.max_size)

    def sample_batch(self,batch_size=32):
        idxs= np.random.randint(0,self.size,size=batch_size)
        s = torch.as_tensor(self.obs_states[idxs],dtype=torch.float32)
        s_= torch.as_tensor(self.obs_nstates[idxs],dtype=torch.float32)
        a = torch.as_tensor(self.actions[idxs],dtype=torch.float32)
        r = torch.as_tensor(self.rewards[idxs],dtype=torch.float32)
        d = torch.as_tensor(self.dones[idxs],dtype=torch.float32)

        return s,a,r,s_,d





class Actor_Critic(nn.Module):
    def __init__(self,obs_space,action_space,action_space_high,hidden_dims):
        super().__init__()
        obs_dim=obs_space
        act_dim=action_space
        act_limit=action_space_high
        self.hidden_size=hidden_dims

        self.actor=_NeuralNetwork(obs_dim,act_dim,act_limit,self.hidden_size)
        self.Q1 = _Q_function(obs_dim,act_dim,hidden_dims)
        self.Q2= _Q_function(obs_dim, act_dim, hidden_dims)
        self.targetQ1 = deepcopy(self.Q1)
        self.targetQ2=deepcopy(self.Q2)

    def act(self,obs):
        with torch.no_grad():
            a,_=self.actor.forward(obs,with_logprob=True)
            return a.numpy()
















