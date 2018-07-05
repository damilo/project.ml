import numpy as np
from collections import defaultdict
import random

class Agentv2:

    def __init__ (self, state_size, action_size):
        # hyperparameters
        # exploration
        self.explore_start = 1.0 # exploration probability at start
        self.explore_stop = 0.01 # minimum exploration probability
        self.decay_rate = 0.0001 # exponential decay rate for exploration prob
        self.explore_p = 0
        # NN
        self.hidden_size = 64 # number of units in each Q-network hidden layer
        self.learning_rate = 0.0001 # Q-network learning rate
        # RL
        self.gamma = 0.8
        
        # Agent's Memory and Brain
        self._memory = Memory ()
        self._brain = QNetwork (name='brain',
            learning_rate=self.learning_rate,
            state_size=state_size, action_size=action_size,
            hidden_size=self.hidden_size)
        
        # Environment
        self.state_size = state_size
        self.action_size = action_size
        

    def act (self, state, step_num, sess):
        action = None
        self.explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp (-self.decay_rate*step_num)
        if self.explore_p > np.random.rand ():
            # Make a random action
            # TODO Sample of action space e.g.
            action = random.randint (0, self.action_size-1)
            #action = self._env.action_space.sample ()
        else:
            # Get action from Q-network
            feed = {self._brain.inputs_: state.reshape ((1, *state.shape))}
            Qs = sess.run (self._brain.output, feed_dict=feed)
            action = np.argmax (Qs)

        return action
        

    def store (self, observation):
        self._memory.add (observation)


    def learn (self, batch_size, sess):
        # Sample mini-batch from memory
        batch = self._memory.sample (batch_size)
        states = np.array ([each[0] for each in batch])
        actions = np.array ([each[1] for each in batch])
        rewards = np.array ([each[2] for each in batch])
        next_states = np.array ([each[3] for each in batch])
        
        # Train network
        target_Qs = sess.run (self._brain.output, feed_dict={self._brain.inputs_: next_states})
        
        # Set target_Qs to 0 for states where episode ends
        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0)
        
        targets = rewards + self.gamma * np.max (target_Qs, axis=1)

        loss, _ = sess.run ([self._brain.loss, self._brain.opt],
                            feed_dict={self._brain.inputs_: states,
                                        self._brain.targetQs_: targets,
                                        self._brain.actions_: actions})
        
        return loss


from collections import deque
class Memory:
    def __init__ (self, max_size=1000):
        self.buffer = deque (maxlen=max_size)
    
    def add (self, sample):
        self.buffer.append (sample)
            
    def sample (self, batch_size):
        idx = np.random.choice (np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]



import tensorflow as tf
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10, name='QNetwork'):
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None)
            
            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

class MLPQNet ():

    def __init__ (self, state_size, action_size, learning_rate):
        self.Net = Sequential ()
        # input layer = feature vector x(s_t)
        self.Net.add (Dense (10, input_dim=state_size, activation='relu', name='Input'))
        # hidden layer
        self.Net.add (Dense (10, activation='relu', name='Hidden_1'))
        # output layer = p(a|x(s_t)), array of action probabilities given feature vector x(s_t)
        self.Net.add (Dense (action_size, activation='sigmoid', name='Output'))

        # value function approximation with stochastic gradient descent
        # since this is linear function approximation, there's only one global optimum > no momentum needed
        sgd = optimizers.SGD (lr=learning_rate, momentum=0.0)
        adam = optimizers.Adam (lr=learning_rate)
        # goal of DQN: minimize mean squared value error VE = (v_true - v_approx)^2
        self.Net.compile (loss='mean_squared_error', optimizer=sgd)
    
    def learn ():
        pass

    def print_architecture (self):
        self.Net.summary ()
    
    # TODO
    def save (self, filepath=None):
        #if (filepath == None):
        #    filepath = 
        self.Net.save_weights (filepath)
    
    def load (self, filepath):
        self.Net.load_weights (filepath)


class Agentv3 ():

    def __init__ (self, state_size, action_size):
        # hyperparameters
        # exploration
        self.epsilon_max = 1.0 # exploration probability at start
        self.epsilon_min = 0.01 # minimum exploration probability
        self.epsilon_decay = 0.0001 # exponential decay rate for exploration prob
        self.explore_p = 0
        # NN
        #self.hidden_size = 64 # number of units in each Q-network hidden layer
        #self.learning_rate = 0.0001 # Q-network learning rate
        # RL
        self.gamma = 0.8
        
        # Agent's Memory and Brain
        self.Memory = Memory ()
        self.Brain = MLPQNet (state_size, action_size, 0.0001)
        
        # Environment
        #self.state_size = state_size
        self.action_size = action_size
        

    def act (self, state, step_num):
        action = None
        explore_p = self.epsilon_min + (self.epsilon_max - self.epsilon_min)*np.exp (-self.epsilon_decay*step_num)
        self.explore_p = explore_p
        
        if explore_p > np.random.rand ():
            # Make a random action
            action = random.randint (0, self.action_size-1)
        else:
            # Get action from Q-network
            action_probs = self.Brain.Net.predict (state)
            action = np.argmax (action_probs)

        return action
        

    def store (self, observation):
        self.Memory.add (observation)


    def learn (self, batch_size):

        batch = self.Memory.sample (batch_size)
        hist = []
        for state, action, reward, next_state in batch:
            target = reward
            if not ((next_state == np.zeros (state.shape)).all ()):
              target = reward + self.gamma * np.amax(self.Brain.Net.predict(np.reshape(next_state, [1, 4]))[0])
            target_f = self.Brain.Net.predict(np.reshape(state, [1, 4]))
            target_f[0][action] = target
            hist.append (self.Brain.Net.fit (np.reshape(state, [1, 4]), target_f, epochs=1, verbose=0))
        return hist

        """
        # Sample mini-batch from memory
        batch = self.Memory.sample (batch_size)
        states = np.array ([each[0] for each in batch])
        actions = np.array ([each[1] for each in batch])
        rewards = np.array ([each[2] for each in batch])
        next_states = np.array ([each[3] for each in batch])

        # target values - the values we want to become
        target_psa = self.Brain.Net.predict (next_states)
        # set target values to 0 for states where episode ends
        episode_ends = (next_states == np.zeros (states[0].shape)).all (axis=1)
        target_psa[episode_ends] = np.zeros (self.action_size)
        
        target_updates = rewards + 0.7 * np.max (target_psa, axis=1)

        # get approximate values
        approx_psa = self.Brain.Net.predict (states)

        # update approximate value for taken action to target value
        for i in range (approx_psa.shape[0]):
            approx_psa[i,actions[i]] = target_updates[i]
        

        # train our brain
        return self.Brain.Net.fit (states, approx_psa, epochs=1, batch_size=batch_size, verbose=0)
        """