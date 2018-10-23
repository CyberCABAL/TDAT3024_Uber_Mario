import gym
import numpy as np
import random
from IPython.display import clear_output

#actions = ["south", "north", "east", "west", "pickup", "dropoff"]
#Based on https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

env = gym.make("Taxi-v2").env

#env.render()

Q = np.zeros([env.observation_space.n, env.action_space.n]);
α = 0.1	# Learn rate
ϵ = 0.1	# Randomness
γ = 0.6	# Future importance


def update_Q(α, γ, action, state, n_state, reward):
    Q[state, action] = (1 - α) * Q[state, action] + α * (reward + γ * np.max(Q[n_state]));



#env.s = 328  # set environment to illustration's state
for i in range(0, 100000):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    #frames = []

    done = False

    while not done:
        action = (env.action_space.sample() if (random.uniform(0, 1) < ϵ) else np.argmax(Q[state]))   # Explore or exploit

        #action = env.action_space.sample()	#random
        #state, reward, done, info = env.step(action)
        n_state, reward, done, info = env.step(action)

        update_Q(α, γ, action, state, n_state, reward);

        if reward == -10:
            penalties += 1

        state = n_state
        epochs += 1

    if i % 1000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

total_epochs, total_penalties = 0, 0
episodes = 100

for i in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        if (i == episodes - 1):
            clear_output(wait=True)
            env.render()

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes}:")
print(f"Timesteps per episode: {total_epochs / episodes}")
print(f"Penalties per episode: {total_penalties / episodes}")
