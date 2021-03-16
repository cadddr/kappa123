import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import gym
import random

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def sampleEpisode(env, policy, epsilon=None):
    states = []
    actions = []
    rewards = []

    observation = env.reset()

    done = False
    while not done:
        states.append(observation)

        action = policy(observation, epsilon)

        observation, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

    return states, actions, rewards

def evaluatePolicy(episode, stateActionValues, visitCounts):
    for i, row in episode.iterrows():
        state = (row.playerSum, row.dealerShowing, row.usableAce)
        cumulativeReturn = episode.iloc[i:].reward.sum()

        visitCounts[state][row.action] += 1
        stateActionValues[state][row.action] += (cumulativeReturn - stateActionValues[state][row.action]) / visitCounts[state][row.action] 

def makeEpsilonGreedyPolicy(stateActionValues, visitCounts, actions, N0=100):
    def chooseEpsiloneGreedyAction(state, epsilon=None):
        if not epsilon:
            stateVisitCount = sum(visitCounts[state].values())
            epsilon = N0 / (N0 + stateVisitCount)

        QValuesByAction = [stateActionValues[state][action] for action in actions]
        maxQValue = max(QValuesByAction)

        def epsilonGreedyProb(q): 
            return epsilon / len(actions) + (1 - epsilon) * (q == maxQValue)

        probs = [epsilonGreedyProb(q) for q in QValuesByAction]
        probs = np.array(probs) / np.sum(probs)

        #logging.debug(f'QValues={QValuesByAction}, epsilon={epsilon}, probs={probs}')
        
        return np.random.choice(actions, p=probs)

    return chooseEpsiloneGreedyAction


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


def runMonteCarlo(doLearning=True, 
                  stateActionValues=None, 
                  visitCounts=None, 
                  policy=None, 
                  numEpisodes=1000000, 
                  actions=[0, 1], 
                  epsilon=None, 
                  N0=100):

    env = gym.make("Blackjack-v0")

    if not (stateActionValues and visitCounts):
        stateActionValues = defaultdict(lambda: defaultdict(float))
        visitCounts = defaultdict(lambda: defaultdict(int))

    if not policy:
        def policy(*args): return random.choice([0,1])

    meanEpisodeReward = 0

    with tqdm(range(numEpisodes)) as pbar:
        for i in pbar:
            states, actions, rewards = sampleEpisode(env, policy)

            episode = pd.DataFrame(states, columns=['playerSum', 'dealerShowing', 'usableAce'])\
                .join(pd.Series(actions, name='action'))\
                .join(pd.Series(rewards, name='reward'))

            meanEpisodeReward += (episode.iloc[-1].reward - meanEpisodeReward) / (i + 1)

            pbar.set_postfix(meanReward=meanEpisodeReward, refresh=False)
            pbar.update(0)

            if doLearning:
                evaluatePolicy(episode, stateActionValues, visitCounts)

                policy = makeEpsilonGreedyPolicy(stateActionValues, visitCounts, actions, N0)

    return stateActionValues, visitCounts, policy

if __name__ == '__main__':
    stateActionValues, visitCounts, policy = runMonteCarlo()
    plotStateActionValueActions3D(stateActionValues)






