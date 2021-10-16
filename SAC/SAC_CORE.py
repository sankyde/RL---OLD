
from SAC_Networks import Actor_Critic,ReplayBuffer
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import timeit
import itertools


def sac():
    polyak_avg=0.995
    lr=1e-3
    alpha=0.2
    gamma=0.99
    #(env.observation_space)
    #print(env.action_space)
    obs_space=8
    act_space=2
    ac=Actor_Critic(obs_space,act_space,env.action_space.high,128)
    ac_target= deepcopy(ac)
    replay=ReplayBuffer(env.observation_space.shape[0],env.action_space.shape[0],int(1e6))

    for p in ac_target.parameters():
        p.requires_grad=False


    def compute_q_loss(data):
        s,a,r,s_,d = data
        q1=ac.Q1(s,a)
        q2=ac.Q2(s,a)

        with torch.no_grad():

            actor_out,actor_logp=ac.actor(s_)

            q1_pi_target=ac_target.Q1(s_,actor_out)
            q2_pi_target=ac_target.Q2(s_,actor_out)

            q_pi_targ=torch.min(q1_pi_target,q2_pi_target)
            backup=r+gamma*(1-d)*(q_pi_targ- alpha*actor_logp)

        lossq1=((q1-backup)**2).mean()
        lossq2=((q2-backup)**2).mean()

        lossq=lossq1+lossq2

        return lossq

    def compute_loss_actor(data):

        s, a, r, s_, d = data

        actor_out,actor_logp=ac.actor(s)
        q1_output=ac.Q1(s,actor_out)
        q2_output=ac.Q2(s,actor_out)

        q_final= torch.min(q1_output,q2_output)

        loss_actor=(alpha*actor_logp - q_final).mean()

        return loss_actor

    actor_opt= Adam(ac.actor.parameters(),lr=lr)
    # q1_opt=Adam(ac.Q1.parameters(),lr=lr)
    # q2_opt=Adam(ac.Q2.parameters(),lr=lr)
    q_parameters=itertools.chain(ac.Q1.parameters(),ac.Q2.parameters())
    q_opt=Adam(q_parameters,lr=lr)

    def update(data):

        q_opt.zero_grad()
        loss_q=compute_q_loss(data)
        loss_q.backward()
        q_opt.step()

        for p in q_parameters:
            p.requires_grad=False

        actor_opt.zero_grad()
        loss_actor=compute_loss_actor(data)
        loss_actor.backward()
        actor_opt.step()

        for xyz in q_parameters:
            xyz.requires_grad=True

        with torch.no_grad():
            for p,p_targs in zip(ac.parameters(),ac_target.parameters()):
                p_targs.data.mul_(polyak_avg)
                p_targs.data.add_((1-polyak_avg) * p.data)



    def test_agent():
        N=100
        obs = env.reset()
        for eps in range(N):
            done_test=False
            reward=0
            print(obs)
            o,r,done_test,_=env.step(ac.act(torch.as_tensor(obs)))
            reward += r
            obs=o

    test_agent()










if __name__=="__main__":
    env=gym.make('LunarLanderContinuous-v2')
    #print(env.action_space)
    sac()




