# A sample of REINFORCE algorithm.
# Using continuous action and continuous state

import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCarContinuous-v0')
np.random.seed(1)

#action_dim = env.action_space.n
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.high.shape[0]

# Initialize weight
weight = np.random.rand(state_dim + 1, action_dim)

def policy(state, w):
    standard_deviation = w[0] ** 2
    mean = np.dot(w[1:].T, state)
    return np.random.normal(mean, standard_deviation)

def policy_grad(state, action, w):
    grads = np.zeros(w.shape)
    sigma = w[0]
    mu = w[1:]
    try:
        grads[0] = ((action - np.dot(mu.T, state)) ** 2 - sigma ** 2) / (sigma ** 3)
        A = action - np.dot(mu.T, state)
        B = A * state
        C = B / (sigma ** 2)
        grads[1:] = C.reshape((2, 1))
        #grads[1:] = (np.dot(action - np.dot(mu.T, state), state)) / (sigma ** 2).reshape(grads[1:].shape)
    except ValueError:
        pass
    except OverflowError:
        pass
    return grads

# Hyperparameters
learning_rate = 0.00002
num_episodes = 10000
gamma = 0.99

episode_score = []
for e in range(num_episodes):
    score = 0.0
    record = []
    state = env.reset()
    while True:
        action = policy(state, weight)
        #print("action: " + str(action))
        next_state, reward, done, info = env.step(action)
        record.append((action, state, reward))
        score += reward
        if done:
            break
        state = next_state
    print("Episode: {}, Score: {}".format(e, score))
    episode_score.append(score)
    rewards = zip(*record)[2]
    average_reward = np.sum(rewards) / float(len(rewards)) # baseline is average reward
    for i, data in enumerate(record):
        action, state, reward = data
        grad = policy_grad(state, action, weight)
        discounted_reward = sum([ r * (gamma ** t) for t, r in enumerate(rewards[i:]) ])
        R = (discounted_reward - average_reward)
        weight = weight + learning_rate * grad * R
    if e % 1000 == 0:
        np.savetxt('out__{}.csv'.format(e), weight, delimiter=',')

plt.plot(np.arange(len(episode_score)), episode_score)
plt.show()
env.close()
print(weight)
np.savetxt('out.csv', weight, delimiter=',')

