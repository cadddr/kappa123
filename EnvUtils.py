'''
Utils for manipulating RL environments and visualizing learned value functions.
'''

import logging
import random
from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from tqdm import tqdm

def sampleEpisode(env, policy, render=False, **kwargs):
    states = []
    actions = []
    rewards = []

    observation = env.reset()

    done = False
    while not done:
        if render:
            clear_output()
            env.render()
        states.append(observation)

        action = policy(observation, **kwargs)

        observation, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

    return states, actions, rewards

def episodeToDataframe(states, actions, rewards):
    return pd.DataFrame(states, columns=['playerSum', 'dealerShowing', 'usableAce'])\
        .join(pd.Series(actions, name='action'))\
        .join(pd.Series(rewards, name='reward'))


def plotStateActionValueActions3D(stateActionValues, usableAce=True):
    stateValues = pd.DataFrame(stateActionValues).T.max(axis=1)\
        .reset_index()\
        .rename(columns={'level_0': 'playerSum', 'level_1': 'dealerShowing', 'level_2': 'usableAce', 0: 'value'})\

    stateValues = stateValues[stateValues.usableAce==usableAce]\
        .sort_values(['playerSum', 'dealerShowing'])

    ax = plt.axes(projection='3d')
    ax.view_init(elev=45., azim=-30)

    X = stateValues.playerSum.drop_duplicates().values
    Y = stateValues.dealerShowing.drop_duplicates().values
    
    Z = stateValues.value.values.reshape(len(X), len(Y))

    X, Y = np.meshgrid(Y, X)
    
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
    plt.ylabel('playerSum')
    plt.xlabel('dealerShowing')

def plotStateActionValues2D(env, stateActionValues, shape=(1, -1)):
    stateValues = [max([stateActionValues[state][action] 
        for action in range(env.action_space.n)]) 
        for state in range(env.observation_space.n)]

    plt.imshow(np.reshape(stateValues, shape))
    plt.show()

