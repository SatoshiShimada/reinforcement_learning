# A sample of REINFORCE algorithm.
# Using discrete action and continuous state

import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
np.random.seed(1)

action_dim = env.action_space.n
state_dim = env.observation_space.high.shape[0]

# Initialize weight
weight = np.random.rand(state_dim, action_dim)

def softmax(x):
    shift_x = x - np.max(x)
    exps = np.exp(shift_x)
    return exps / np.sum(exps)

def policy(state, w):
    z = state.dot(w)
    return softmax(z)

def policy_grad(state, action, w):
    """
    Calculate gradient by nemerical differentiation.
    """
    epsilon = 1e-8
    grads = np.zeros(w.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w1 = np.copy(w)
            w2 = np.copy(w)
            w1[i, j] += epsilon
            w2[i, j] -= epsilon
            grads[i, j] = (np.log(policy(state, w1)[action]) - np.log(policy(state, w2)[action])) / (2.0 * epsilon)
    return grads

# Hyperparameters
learning_rate = 0.002
num_episodes = 1500
gamma = 0.99

episode_score = []
for e in range(num_episodes):
    score = 0.0
    record = []
    state = env.reset()
    while True:
        action_probs = policy(state, weight)
        action = np.random.choice(action_dim, p=action_probs)
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

plt.plot(np.arange(len(episode_score)), episode_score)
plt.show()
env.close()

