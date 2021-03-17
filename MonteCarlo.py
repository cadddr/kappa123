'''
Monte Carlo policy evaluation with epsilon-greedy action selection.
'''


import numpy as np
from collections import defaultdict
from tqdm import tqdm
from EnvUtils import sampleEpisode
import random
import mlflow
from collections.abc import Iterable

def loggingHelper(label, value, *args):
    argStrings = [
        '_'.join(map(str, arg)) 
            if isinstance(arg, Iterable) 
            else str(arg) 
        for arg in args
    ]

    if isinstance(value, Iterable):
        for i, val in enumerate(value):
            mlflow.log_metric(f"{label}_{'_'.join(argStrings)}_{i}", val)
    else:
        mlflow.log_metric(f"{label}_{'_'.join(argStrings)}", value)

def evaluatePolicy(states, actions, rewards, stateActionValues, visitCounts):
    for i, (state, action) in enumerate(zip(states, actions)):
        cumulativeReturn = sum(rewards[i:])

        visitCounts[state][action] += 1
        loggingHelper('N', visitCounts[state][action], state, action)

        stateActionValues[state][action] += \
            (cumulativeReturn - stateActionValues[state][action]) \
                / visitCounts[state][action] 

        loggingHelper('Q', stateActionValues[state][action], state, action)

def makeEpsilonGreedyPolicy(stateActionValues, visitCounts, actions, N0=100):
    def chooseEpsiloneGreedyAction(state, epsilon=None):
        if not epsilon:
            stateVisitCount = sum(visitCounts[state].values())
            epsilon = N0 / (N0 + stateVisitCount)
            loggingHelper('epsilon', epsilon, state)

        QValuesByAction = [stateActionValues[state][action] for action in actions]
        maxQValue = max(QValuesByAction)

        def epsilonGreedyProb(q): 
            return epsilon / len(actions) + (1 - epsilon) * (q == maxQValue)

        probs = [epsilonGreedyProb(q) for q in QValuesByAction]
        probs = np.array(probs) / np.sum(probs)
        loggingHelper('actionProbs', probs, state)
        
        return np.random.choice(actions, p=probs)

    return chooseEpsiloneGreedyAction

def runMonteCarlo(env, 
                  stateActionValues=None, 
                  visitCounts=None, 
                  policy=None, 
                  numEpisodes=100000, 
                  render=False,
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

    mlflow.start_run()

    with tqdm(range(numEpisodes)) as pbar:
        for i in pbar:
            states, actions, rewards = sampleEpisode(env, policy, render)

            meanEpisodeReward += (rewards[-1] - meanEpisodeReward) / (i + 1)
            loggingHelper('meanReward', meanEpisodeReward)

            pbar.set_postfix(meanReward=meanEpisodeReward, refresh=False)
            pbar.update(0)

            if doLearning:
                evaluatePolicy(states, actions, rewards, stateActionValues, visitCounts)

                policy = makeEpsilonGreedyPolicy(stateActionValues, visitCounts, actions, N0)

    mlflow.end_run()
    return stateActionValues, visitCounts, policy
