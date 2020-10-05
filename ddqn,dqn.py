import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
class Buffer:
    def __init__(self,mem_size,input_size,output_size):
        self.max_size=mem_size
        self.counter=0
        self.state_mem=np.zeros((mem_size,input_size),dtype=np.float32)
        self.action_mem=np.zeros((mem_size,output_size),dtype=np.float32)
        self.reward_mem=np.zeros(mem_size,dtype=np.int32)
        self.next_state_mem=np.zeros((mem_size,input_size),dtype=np.float32)
        self.terminal_mem=np.zeros(mem_size)

    def store_mem(self,s,a,r,s_,terminal):
        index= self.counter % self.max_size
        if self.counter>self.max_size:
            self.counter=0
        self.state_mem[index]=s
        self.action_mem[index]=a
        self.reward_mem[index]=r
        self.next_state_mem[index]=s_
        self.terminal_mem[index]=terminal
        self.counter +=1
    def sample_mem(self,batch):
        indices=np.random.choice(np.arange(self.counter),batch)
        s= self.state_mem[indices]
        a=self.action_mem[indices]
        r=self.reward_mem[indices]
        s_=self.next_state_mem[indices]
        d=self.terminal_mem[indices]
        return s,a,r,s_,d

def NeuralNetwork(inp_shape,out_shape,lr=0.01):
    model=Sequential()
    model.add(Input(inp_shape))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(out_shape, activation='softmax'))
    model.compile(optimizer='adam',loss='mse')
    return model



dueling=False
Q=True
inputs=8
outputs=4
tau=0.001
eps=1
min_eps=0.005
eps_decay=0.0005
gamma=0.99
buffer=Buffer(10000,inputs,outputs)
env=gym.make('LunarLander-v2')
eval_network=NeuralNetwork(inputs,outputs)
target_network=NeuralNetwork(inputs,outputs)
num_games=5000
for i in range(num_games):
    init=env.reset()
    done=False
    reward_sum=0
    while not done:
        if eps < np.random.random():
            act = env.action_space.sample()

        else:
            act = target_network.predict(np.expand_dims(init, axis=0))
            #print(act)

        obs,reward,done,_=env.step(np.argmax(act))
        buffer.store_mem(init,act,reward,obs,done)
        reward_sum += reward
        init = obs
        eps -= eps*eps_decay

        if i>3:
            s,a,r,s_,terminal=buffer.sample_mem(32)
            curr_Q=eval_network.predict(s)
            next_q=eval_network.predict(s_)
            q_targ=target_network.predict(s_)
            if dueling:
                for k in range(int(curr_Q.shape[0])):
                    curr_Q[k,np.arange(outputs)]=r[k]+ gamma*q_targ[k,int(np.argmax(next_q[k]))]
            if Q:
                for k in range(int(curr_Q.shape[0])):
                    curr_Q[k,np.arange(outputs)]=r[k]+ gamma*np.max(q_targ[k])
            
            eval_network.train_on_batch(s,curr_Q)
            st = target_network.get_weights()
            et = eval_network.get_weights()
            weights = []
            for target_param, param in zip(st, et):
                weights.append(tau * param + (1 - tau) * target_param)
            target_network.set_weights(weights)



    print(reward_sum)



