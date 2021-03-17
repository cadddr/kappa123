def sampleEpisode(env, policy, **kwargs):
    states = []
    actions = []
    rewards = []

    observation = env.reset()

    done = False
    while not done:
        states.append(observation)

        action = policy(observation, **kwargs)

        observation, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

    return states, actions, rewards