

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential,Input
import numpy as np
import gym
import matplotlib.pyplot as plt


def NeuralNetwork(input_dims,output_dims):
    model=Sequential()
    model.add(Input(input_dims))
    model.add(Dense(256,activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dims, activation='softmax'))
    model.compile(optimizer=tf.optimizers.Adam(lr=0.001),loss='mse',)
    return model
def AdvNeuralNetwork(input_dims,output_dims):
    model=Sequential()
    model.add(Input(input_dims))
    model.add(Dense(256,activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dims, activation='softmax'))
    model.compile(optimizer='adam',loss='mse')
    return model
#constants
num_games=5000
gamma=0.99
adv_ON=False
Q_ON=True
Gt_ON=False
inputs=8
outputs=4
multiplier=None

#defs
policy_network=NeuralNetwork(inputs,outputs)
adv_network=AdvNeuralNetwork(inputs,1)
env=gym.make('LunarLander-v2')


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
    disc_rewards = (disc_rewards-disc_rewards.mean())/ disc_rewards.std()

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

        rtg = RTG(reward_mem)

        if adv_ON:
            Q = (np.sum(np.array((gamma ** len(reward_mem)) * adv_network.predict(np.array(next_state_mem))))) + rtg

            Q = np.array(Q)
            f = adv_network.predict(np.array(state_mem))

            multiplier= (np.expand_dims(Q,axis=1)-f)
            
        if Gt_ON:
            multiplier=RTG(reward_mem)
        if Q_ON:

            Q=(np.sum(np.array((gamma ** len(reward_mem)) * adv_network.predict(np.array(next_state_mem))))) + rtg

            multiplier=np.array(Q)


            
       

        out=[]
        rtg=RTG(reward_mem)
        
        for d_r,action in zip(multiplier,action_mem):
            out.append(-tf.keras.backend.log(action)*d_r)
       
        policy_network.train_on_batch(np.array(state_mem),np.squeeze(np.array(out)))


        adv_network.train_on_batch(np.array(state_mem),np.array(rtg))
        #y_vals.append(reward_sum)
        #x_vals.append(i)
        print(reward_sum)



    #print(timeit.timeit(setup=mysetup,stmt=code,number=10))




