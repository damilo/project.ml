import numpy as np
import random as rn
import os
import tensorflow as tf
from keras import backend as K
from keras import models

from .dqn import MLPQNet
from .memory import Memory


### CLASS MLPQNAgent w/ experience replay, target network
class MLPQNAgent ():
    
    
    def __init__ (self, state_shape, action_size, init_info=False):
        # rl hyperparameters
         # exploration probability
        self.__eps_max = 1.0 # exploration probability at start
        self.__eps_min = 0.01 # minimum exploration probability
        self.__eps_decay = 0.00005 # exponential decay rate for exploration prob
        self.explore_p = self.__eps_max - self.__eps_min
         # discount factor
        self.__gamma = 0.9
        
        # agent's memory (for experience replay)
        self.__memory = Memory (100000)
        
        # agent's brain
        self.__lr = 0.001 # hyperparameter: learning rate
        self.__brain = (MLPQNet (state_shape, action_size, self.__lr)).model

        # target estimator network (for target network)
        self.__target_estimator = (MLPQNet (state_shape, action_size, self.__lr)).model #models.clone_model (self.__brain)
        self.__target_estimator.set_weights (self.__brain.get_weights ())
        self.__update_target_estimator_every = 2000 # parameter: how many learn cycles until updating the target estimator network
        self.__tau = 0.125
        
        # environment
        self.__action_size = action_size
        self.__learn_cnt = 1 # parameter: learning counter

        if (init_info):
            print ('rl hyperparameters:')
            print (' epsilon: max = {}, min = {}, decay = {}'.format (self.__eps_max, self.__eps_min, self.__eps_decay))
            print (' gamma: value = {}'.format (self.__gamma))
            print ('nn hyperparameters:')
            print (' learning rate: value = {}'.format (self.__lr))
            print ('nn architecture:')
            self.print_qnet ()
            print ('target nn parameters:')
            print (' update target every learn cycle: {}'.format (self.__update_target_estimator_every))
            print (' tau: value = {}'.format (self.__tau))
    
    
    def select_action (self, state):
        action = None
        if self.explore_p > np.random.rand ():
            # Make a random action
            action = rn.randint (0, self.__action_size-1)
        else:
            # Get action from Q-network
            action_probs = self.__brain.predict (state)
            action = np.argmax (action_probs)

        return action
    
    
    def store (self, observation):
        self.__memory.add (observation)
    
    
    def learn (self, batch_size):
        # because of this if, no pre-initialization of buffer is needed
        if (self.__memory.length () < batch_size):
            return
        
        batch = self.__memory.sample (batch_size)
        
        # shape of each array: (batch_size, each_shape)
        states = np.array ([each[0] for each in batch])
        actions = np.array ([each[1] for each in batch])
        rewards = np.array ([each[2] for each in batch])
        next_states = np.array ([each[3] for each in batch])
        
        # NEW learn w/ target network

        # get Q-table of next states from target estimator network
        qtable_next_states = self.__target_estimator.predict_on_batch (next_states)
        # set Q-table to 0 for states where episode ends
        episode_ends = (next_states == np.zeros (states[0].shape)).all (axis=1)
        qtable_next_states[episode_ends] = np.zeros (self.__action_size)
        
        # NEW get new target values (done info is needed, then no need to do the 'episode_ends' stuff above)
        #target_values = rewards + np.invert (dones, dtype=np.float32) * self.__gamma * np.amax (qtable_next_states, axis=1)
        # OLD get new target values
        target_values = rewards + self.__gamma * np.amax (qtable_next_states, axis=1)
        
        # update Q-table of states with target values
        # due to this, a weight update of the neural network can be performed
        qtable_states = self.__brain.predict_on_batch (states)
        for new, qtable_state, action in zip (range (batch_size), qtable_states, actions):
            qtable_state[action] = target_values[new]
        
        # update target estimator network
        if self.__learn_cnt % self.__update_target_estimator_every == 0:
            self.__update_target_estimator ()

        
        # train network with updated Q-tables
        loss = self.__brain.train_on_batch (states, qtable_states)

        # update exploration
        self.explore_p = self.__eps_min + (self.__eps_max - self.__eps_min)*np.exp (-self.__eps_decay*self.__learn_cnt)
        
        # update learning cycle count
        self.__learn_cnt += 1
        
        return loss
    
    
    def print_qnet (self):
        return self.__brain.summary ()
    
    
    def __update_target_estimator (self):
        # copy weights from brain to target estimator network
        # note: if weights are copied: no compile of target model needed to get same predictions
        weights = self.__brain.get_weights ()
        target_weights = self.__target_estimator.get_weights ()
        for i in range (len (target_weights)):
            target_weights[i] = weights[i] * self.__tau + target_weights[i] * (1 - self.__tau)

        self.__target_estimator.set_weights (target_weights)
        
        print ('>>> target estimator network updated ({})'.format (self.__learn_cnt))
    
    
    def reset (self):
        slf.__learn_cnt = 0
    
    
    def get_memory_size (self):
        return (self.__memory.length ())