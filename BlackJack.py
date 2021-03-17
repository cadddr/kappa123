import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import gym
import random
from EnvUtils import sampleEpisode

def episodeToDataframe(states, actions, rewards):
    return pd.DataFrame(states, columns=['playerSum', 'dealerShowing', 'usableAce'])\
        .join(pd.Series(actions, name='action'))\
        .join(pd.Series(rewards, name='reward'))


def plotStateActionValueActions3D(stateActionValues, usableAce=True):
    stateValues = pd.DataFrame(stateActionValues).T.max(axis=1)\
        .reset_index()\
        .rename(columns={'level_0': 'playerSum', 'level_1': 'dealerShowing', 'level_2': 'usableAce', 0: 'value'})\

    stateValues = stateValues[stateValues.usableAce==usableAce].sort_values(['playerSum', 'dealerShowing'])

    ax = plt.axes(projection='3d')
    ax.view_init(elev=45., azim=-30)

    X = stateValues.playerSum.drop_duplicates().values
    Y = stateValues.dealerShowing.drop_duplicates().values
    
    Z = stateValues.value.values.reshape(len(X), len(Y))

    X, Y = np.meshgrid(Y, X)
    
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
    plt.ylabel('playerSum')
    plt.xlabel('dealerShowing')


