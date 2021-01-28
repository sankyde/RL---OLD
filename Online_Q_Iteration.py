import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import gym
import matplotlib.pyplot as plt
env=gym.make('LunarLander-v2')
gamma=0.95


class Model(nn.Module):
    def __init__(self,ins,outs):
        super(Model,self).__init__()
        self.l1=nn.Linear(ins,256)
        self.l2=nn.Linear(256,128)
        self.l3=nn.Linear(128,outs)

    def forward(self,state):
        out=F.relu(self.l1(state))
        out=F.relu(self.l2(out))
        out=F.relu(self.l3(out))
        outputs=F.softmax(out,dim=-1)
        return outputs

if __name__=='__main__':
    rew=[]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(8,4).to(device)

    optimizer =Adam(model.parameters(), lr=0.01)
    N=100000
    for i in range(N):

        done=False
        state=env.reset()
        eps=1
        rew_sum=0
        while not done:

            rand=np.random.rand()

            if rand < eps:
                act=env.action_space.sample()
            else:
                act=model.forward(torch.FloatTensor(state).to(device))
                act=torch.argmax(act).detach().cpu().numpy()

            next_state,reward,done,_=env.step(act)

            Q_target = reward + gamma * model.forward(torch.FloatTensor(next_state).to(device))
            #print(torch.argmax(Q_target))
            #print(act)
            Q_target=Q_target.cpu()
            loss=torch.Tensor(np.array(torch.tensor(act)-torch.argmax(Q_target)+1,dtype=np.float32))
            #print("loss",loss)
            loss=torch.tensor(loss,requires_grad=True)
            # loss=torch.Tensor(,requires_grad=True)
            # print(loss)
            #loss=np.sum(np.power((act, np.argmax(Q_target)),2))

            #loss= torch.FloatTensor(loss,requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rew_sum += reward
            state=next_state
            eps -= eps/0.005

        print(rew_sum)
        rew.append(rew_sum)









