import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential,Input
import numpy as np
import gym

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
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam',loss='mse')
    return model
#constants
num_games=5000
gamma=0.99

inputs=4
outputs=2

#defs
policy_network=NeuralNetwork(inputs,outputs)
adv_network=AdvNeuralNetwork(inputs,1)
env=gym.make('CartPole-v1')

#def estimate_value():


#funcs
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
        reward_sum += reward
        initial=obs

    disc_rewards=RTG(reward_mem)
    # print(disc_rewards.shape)
    # print(np.squeeze(np.array(action_mem).shape))
    #out=disc_rewards * -np.log(np.array(action_mem))

    out=[]

    for d_r,action in zip(disc_rewards,action_mem):
        out.append(-tf.keras.backend.log(action)*d_r)
    #print(out)
    policy_network.train_on_batch(np.array(state_mem),np.squeeze(out))
    print(reward_sum)


# 10.0
# 11.0
# 17.0
# 30.0
# 21.0
# 26.0
# 15.0
# 23.0
# 36.0
# 50.0
# 58.0
# 25.0
# 15.0
# 210.0
# 402.0
# 17.0
# 11.0
# 10.0
# 22.0
# 10.0
# 14.0
# 22.0
# 27.0
# 27.0
# 22.0
# 14.0
# 12.0
# 44.0
# 22.0
# 16.0
# 25.0
# 38.0
# 25.0
# 35.0
# 27.0
# 16.0
# 61.0
# 31.0
# 52.0
# 20.0
# 25.0
# 27.0
# 190.0
# 17.0
# 206.0
# 19.0
# 30.0
# 238.0
# 61.0
# 500.0
# 36.0
# 500.0
# 10.0
# 10.0
# 20.0
# 29.0
# 19.0
# 39.0
# 31.0
# 32.0
# 8.0
# 24.0
# 32.0
# 10.0
# 26.0
# 9.0
# 39.0
# 10.0
# 45.0
# 64.0
# 41.0
# 36.0
# 499.0
# 44.0
# 500.0
# 42.0
# 76.0
# 72.0
# 176.0
# 406.0
# 12.0
# 43.0
# 23.0
# 10.0
# 12.0
# 24.0
# 19.0
# 21.0
# 33.0
# 38.0
# 10.0
# 24.0
# 25.0
# 12.0
# 26.0
# 21.0
# 26.0
# 14.0
# 12.0
# 48.0
# 38.0
# 43.0
# 41.0
# 42.0
# 51.0
# 35.0
# 22.0
# 21.0
# 422.0
# 47.0
# 52.0
# 27.0
# 10.0
# 338.0
# 49.0
# 483.0
# 22.0
# 500.0
# 202.0
# 254.0
# 18.0
# 26.0
# 137.0
# 14.0
# 114.0
# 175.0
# 59.0
# 24.0
# 29.0
# 26.0
# 10.0
# 200.0
# 17.0
# 12.0
# 10.0
# 20.0
# 12.0
# 24.0
# 17.0
# 60.0
# 79.0
# 112.0
# 38.0
# 152.0
# 77.0
# 15.0
# 25.0
# 252.0
# 275.0
# 120.0
# 14.0
# 500.0
# 500.0
# 142.0
# 29.0
# 16.0
# 490.0
# 22.0
# 37.0
# 22.0
# 98.0
# 15.0
# 32.0
# 18.0
# 16.0
# 17.0
# 20.0
# 15.0
# 21.0
# 19.0
# 49.0
# 29.0
# 27.0
# 23.0
# 17.0
# 34.0
# 19.0
# 14.0
# 20.0
# 65.0
# 23.0
# 19.0
# 20.0
# 40.0
# 18.0
# 10.0
# 9.0
# 25.0
# 10.0
# 27.0
# 22.0
# 12.0
# 203.0
# 167.0
# 16.0
# 22.0
# 45.0
# 24.0
# 15.0
# 88.0
# 116.0


