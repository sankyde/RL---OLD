import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential,Input
import numpy as np
import gym
from xlwt import Workbook





def NeuralNetwork(input_dims,output_dims):
    model=Sequential()
    model.add(Input(input_dims))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dims, activation='softmax'))
    model.compile(optimizer='adam',loss='mse')
    return model
def AdvNeuralNetwork(input_dims):
    model=Sequential()
    model.add(Input(input_dims))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    return model
#constants
num_games=5000
gamma=0.99

inputs=4
outputs=2

#defs
policy_network=NeuralNetwork(inputs,outputs)
env=gym.make('CartPole-v1')
advnetwork=AdvNeuralNetwork(inputs)

#funcs
def GAE(r,next_state,state):

    deltas= np.array(r) + gamma * np.squeeze(advnetwork.predict(np.array(next_state))) - np.squeeze(advnetwork.predict(np.array(state)))
    adv=[]

    for idx,j in enumerate(deltas):
        delta_sum=0
        pow=0
        for idx1,l in enumerate(deltas,start=idx):
            delta_sum += l * (gamma ** pow)
            pow += 1
        adv.append(delta_sum)

    # print(np.array(adv).shape)
    adv=np.array(adv)

    return adv/adv.mean()


def RTG(r_mem):
    disc_rewards=[]
    for t in range(len(r_mem)):
        Gt=0
        pw=0
        for k in range(t,len(r_mem)):
            Gt += (gamma**pw)*r_mem[k]
            pw += 1
        disc_rewards.append(Gt)
    disc_rewards=np.array(disc_rewards)
    disc_rewards = (disc_rewards-np.mean(disc_rewards))/ disc_rewards.std()

    return disc_rewards



for i in range(num_games):
    reward_mem=[]
    state_mem=[]
    next_state_mem=[]
    action_mem=[]
    initial=env.reset()
    reward_sum=0
    done=False
    while not done:
        act=policy_network.predict(np.expand_dims(initial,axis=0))
        obs,reward,done,_=env.step(np.argmax(act))
        action_mem.append(act)
        state_mem.append(initial)
        reward_mem.append(reward)
        next_state_mem.append(obs)
        reward_sum += reward
        initial=obs

    disc_rewards1=RTG(reward_mem)

    gae=GAE(reward_mem,next_state_mem,state_mem)
    disc_rewards =gae
    
    out=[]
   
    for d_r,action in zip(disc_rewards,action_mem):
        out.append(-tf.keras.backend.log(action)*d_r)
   
    policy_network.train_on_batch(np.array(state_mem),np.squeeze(out))
    advnetwork.train_on_batch(np.array(state_mem),disc_rewards1)
    print(reward_sum)
    






