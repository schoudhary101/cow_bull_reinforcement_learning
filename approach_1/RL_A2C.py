import sys
import matplotlib.pyplot as plt
import numpy as np
import pylab

from A2CAgent import A2CAgent
from Cowbull_v2 import Cowbull_v2

EPISODES = 100000
STEPS = 20

N_SIZE = 4
N_LONG = STEPS
N_OPTION = 10

if __name__ == "__main__":

    env = Cowbull_v2(N_LONG, N_SIZE)

    agent = A2CAgent(N_LONG, N_SIZE, N_OPTION)

    rewards = []
    
    num_wins = 0
    
    for episode in range(EPISODES):

        done = False
        reward = 0
        sum_reward = 0
        action = []
        state = np.zeros((N_LONG, N_SIZE+1))
        next_state = np.zeros((N_LONG, N_SIZE+1))
        env.reset()

        for step in range(STEPS):

            action = agent.get_action(state)
            
            reward, done = env.step(action)
            
            if (agent.render):
                env.render()
            
            next_state[step, :(N_SIZE)] = np.array(action)
            next_state[step, N_SIZE] = reward
            
            agent.train_model(state, action, reward, next_state)
            
            state = next_state
            
            sum_reward += reward
            
            if reward==5040:
                num_wins+=1
            
            if done:
                break
            
        rewards.append(sum_reward/STEPS)
        if episode%100==0:
            print("On Episode: ", episode+1)
            plt.plot(rewards)
            plt.savefig('plot.png', dpi = 150)
            print("No. of Wins:",num_wins)

