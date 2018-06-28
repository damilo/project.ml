import numpy as np
from collections import defaultdict

class Agent:

    def __init__ (self):

        self.nA = 3
        self.Q = defaultdict (lambda: np.zeros(self.nA))
        
        ### hyperparameters
        self.epsilon = 1.0
        self.epsilon_counter = 1
        self.gamma = 0.3
        self.alpha = 0.5
        
    def epsilon_greedy_probs (self, Q_s, epsilon, nA):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones (nA) * epsilon / nA
        best_a = np.argmax (Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        
        return policy_s
    
    # Sarsa (on-policy TD control)
    def td_sarsa (self, state, action, reward, next_state):
        
        next_action = self.select_action (next_state)
        
        td_error = (reward+self.gamma*self.Q[next_state][next_action]) - self.Q[state][action]
        
        return self.alpha*td_error
    
    # -----
    
    def select_action (self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        policy_s = self.epsilon_greedy_probs (self.Q[state], self.epsilon, self.nA)
        action = np.random.choice (np.arange (self.nA), p=policy_s)
        
        return action

    def step (self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        #if done:
            # TD control requires action value of terminal state to be zero
            # BUT: for this task, the terminal states are also starting states
            # so we don't do anything with it
        
        self.Q[state][action] += self.td_sarsa (state, action, reward, next_state)
        
        # update epsilon per step
        self.epsilon = (self.epsilon*1. / self.epsilon_counter)
        # -> this prevented from learning better: if self.epsilon > 0.05 else 0.05
        self.epsilon_counter+=1