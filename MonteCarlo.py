import numpy as np
from collections import defaultdict
from tqdm import tqdm
from EnvUtils import sampleEpisode
import random
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

def evaluatePolicy(states, actions, rewards, stateActionValues, visitCounts):
    for i, (state, action) in enumerate(zip(states, actions)):
        cumulativeReturn = sum(rewards[i:])

        visitCounts[state][action] += 1

        stateActionValues[state][action] += \
            (cumulativeReturn - stateActionValues[state][action]) \
                / visitCounts[state][action] 

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

        logging.debug(f'QValues={QValuesByAction}, epsilon={epsilon}, probs={probs}')
        
        return np.random.choice(actions, p=probs)

    return chooseEpsiloneGreedyAction

def runMonteCarlo(env, 
                  stateActionValues=None, 
                  visitCounts=None, 
                  policy=None, 
                  numEpisodes=100000, 
                  doLearning=True, 
                  epsilon=None, 
                  N0=100):

    actions = list(range(env.action_space.n))

    if not (stateActionValues and visitCounts):
        stateActionValues = defaultdict(lambda: defaultdict(float))
        visitCounts = defaultdict(lambda: defaultdict(int))

    if not policy:
        def policy(*args): return random.choice(actions)

    meanEpisodeReward = 0

    with tqdm(range(numEpisodes)) as pbar:
        for i in pbar:
            states, actions, rewards = sampleEpisode(env, policy)

            meanEpisodeReward += (rewards[-1] - meanEpisodeReward) / (i + 1)

            pbar.set_postfix(meanReward=meanEpisodeReward, refresh=False)
            pbar.update(0)

            if doLearning:
                evaluatePolicy(states, actions, rewards, stateActionValues, visitCounts)

                policy = makeEpsilonGreedyPolicy(stateActionValues, visitCounts, actions, N0)

    return stateActionValues, visitCounts, policy
