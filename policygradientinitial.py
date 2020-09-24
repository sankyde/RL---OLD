

import tensorflow as tf
import gym
from tensorflow.keras import Sequential,Input,Model
from tensorflow.keras.layers import Dense
from keras import backend as K
import numpy as np

def NN(inputs,outputs,lr,advantages):
    def custom_loss(y_true,y_pred):
      out=-K.log(y_pred)*y_true
      return K.sum(out*advantages)
    inp=Input(shape=(inputs))
    inpa=Input(shape=advantages)
    nx=Dense(256,activation='relu')(inp)
    nx=Dense(128,activation='relu')(nx)
    out=Dense(outputs,activation='softmax')(nx)
    model=Model(inputs=[inp,inpa],outputs=[out])
    policy=Model(inputs=[inp],outputs=[out])
    model.compile(optimizer=tf.optimizers.Adam(lr=lr),loss=custom_loss)

    return model,policy


env=gym.make('CartPole-v1')
observation_space=4
action_space=2
gamma=0.95
advantage=1
learning_rate=0.1
adv,policy=NN(observation_space,action_space,learning_rate,advantage)
reward_mem=[]
action_mem=[]
state_mem=[]
mean_reward=0
num_games=10000
for i in range(num_games):
    initial=env.reset()
    reward_sum=0
    done=False   

    while not done:
        act=np.argmax(policy.predict(np.expand_dims(initial,axis=0)))
        obs,reward,done,_=env.step(act)
        reward_mem.append(reward)
        state_mem.append(initial)
        action_mem.append(tf.one_hot(act,action_space))
        reward_sum += reward
        initial=obs
    
    #print('Mean rollout reward',rollout_reward/len(reward_mem))
    mean_reward += reward_sum    



    G=np.zeros_like(reward_mem)


    for step in range(len(reward_mem)):
      discount=1
      G_sum=0
      for t in range(step,len(reward_mem)):
        G_sum += reward_mem[t]*discount
        discount *= gamma
      G[t]=G_sum
    mean=np.mean(G)
    std = np.std(G) if np.std(G) > 0 else 1
    G = (G - mean) / std
    #G +=reward_mem[step]*np.power(gamma,step)
    

    #print(out.shape)

    adv.train_on_batch([tf.convert_to_tensor(state_mem),G],tf.convert_to_tensor(action_mem,dtype=np.float32))
    #print('Printing Out')
    #print(out)
    if i%32==0 and i>2:
      print('Mean_reward',mean_reward/i)
    state_mem=[]
    reward_mem=[]
    action_mem=[]
