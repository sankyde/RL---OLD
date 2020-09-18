
import tensorflow as tf
import gym
from tensorflow.keras.layers import Dense,Conv2D,Flatten
import numpy as np
from tensorflow.keras import Sequential



def pp(img):
    return tf.image.rgb_to_grayscale(img)


env=gym.make('Breakout-v4')

n_actions=4


initial=env.reset()
initial=pp(initial)


class Mem():
    
    def __init__(self,input_shape,output_shape,maxlen):
        self.maxlen=maxlen
        self.mem_cntr=0
        self.state_memory=np.zeros((maxlen,*input_shape),dtype=np.float32)
        self.next_state_memory=np.zeros((maxlen,*input_shape),dtype=np.float32)
        self.reward_memory=np.zeros(maxlen,dtype=np.float32)
        self.action_memory=np.zeros((maxlen,output_shape),dtype=np.uint8)
        self.terminal_memory=np.zeros(maxlen,dtype=np.float32)
        
        
    def store_mem(self,s,a,r,s_,terminal):
        
        if self.mem_cntr > self.maxlen:
            self.mem_cntr=0
        
        self.state_memory[self.mem_cntr]=s
        self.next_state_memory[self.mem_cntr]=s_
        self.reward_memory[self.mem_cntr]=r
        self.action_memory[self.mem_cntr]=a
        self.terminal_memory[self.mem_cntr]=terminal
        
        self.mem_cntr += 1
        
    def random_sample(self,batch_size):
        
        length=self.mem_cntr
        
        batch=np.random.choice(length,size=batch_size,replace=False)
        
        r=self.reward_memory[batch]
        s_=self.next_state_memory[batch]
        s=self.state_memory[batch]
        done=self.terminal_memory[batch]
        a=self.action_memory[batch]
        
        
        return s,a,r,s_,done

        
        
    def ret(self):
        
        return self.reward_memory,self.next_state_memory,self.state_memory,self.terminal_memory,self.action_memory



# mem=Mem(initial.shape,n_actions,100)

# for i in range(100):
    
#     act=env.action_space.sample()
#     obs,reward,done,_=env.step(act)
#     mem.store_mem(initial,act,reward,pp(obs),done)
#     initial=pp(obs)



# s,a,r,s_,done=mem.random_sample(4)

class NeuralNetwork():
    
    def dqn(self,input_shape,output_shape,lr=0.01):
        model=Sequential()
        model.add(Conv2D(64,4,strides=4,input_shape=input_shape,activation='relu'))
        model.add(Conv2D(32,2,strides=2,activation='relu'))
        model.add(Conv2D(32,2,strides=2,activation='relu'))
        model.add(Flatten())
        model.add(Dense(256,activation='relu'))
        model.add(Dense(output_shape))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),loss='mse',)
        return model
    

# nn=NeuralNetwork().dqn(initial.shape,n_actions)
# x=nn.predict(tf.cast(np.expand_dims(pp(env.reset()),axis=0),dtype=tf.float32))

# nn.train_on_batch(s_,a)
    
class Agent():
    def __init__(self,gamma,arr_size):
        
        self.target_network=NeuralNetwork().dqn(initial.shape,n_actions)
        self.learning_network=NeuralNetwork().dqn(initial.shape,n_actions)
        
        self.gamma=gamma
        self.replay_mem=Mem(initial.shape,n_actions,arr_size)
        self.eps=1
        self.eps_min=0.1
        self.eps_linearity=0.005
        
    def select_action(self,obs):
        
        
        
        obs=tf.cast(np.expand_dims(obs,axis=0),dtype=tf.float32)
        
        eps=np.random.random()
        
        if eps < self.eps:
            act = env.action_space.sample()
            
        else:
            act=self.target_network.predict(obs)                     
        
        return act
    
    
    def replace_networks(self):
        self.target_network.set_weights(self.learning_network.get_weights())
        
    def learn(self,batch_size=2000):
        
        s,a,r,s_,done=self.replay_mem.random_sample(batch_size)
        
        
        
        indices=[i for i in range(batch_size)]
        
        out=self.learning_network.predict(s_)
        
        y=out
        
        y[indices,a]=r + self.gamma*np.max(out,axis=1)*(1-done)
        
        self.learning_network.train_on_batch(s,y)
        
        
        
        
# x=Agent(0.95,100)

# indices=np.arange(4)

# y=x.learning_network.predict(s_)

# y[indices,a]=reward + 0.95* np.max(y,axis=1) * 1-done


N=100000
num_games=100
agent=Agent(0.95,N)
MAX_STEPS=20000
for i in range(num_games):    
    
    
    done=False
    
    first=env.reset()
    first=pp(first)
    
    k=env.reset()
    reward_sum=0
    
    for i in range(MAX_STEPS):
        
        act=agent.select_action(first)
        act=np.argmax(act)
        obs,reward,done,_=env.step(np.argmax(act))
        
        
        
        agent.replay_mem.store_mem(first,act,reward,pp(obs),done)
        
        reward_sum += reward
        
        first=pp(obs)
    
    
        
    if agent.eps<agent.eps_min:
           agent.eps=agent.eps_min
    else:
           agent.eps -= agent.eps_linearity*0.95
    
    print(reward_sum)
    
    if i% 16 == 0:
        agent.replace_networks()
    
        
    
        
        



















    
    
    