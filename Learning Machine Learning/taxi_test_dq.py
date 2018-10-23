import gym
import numpy as np
import random
import keras as k
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Dropout
from IPython.display import clear_output
from collections import deque


#actions = ["south", "north", "east", "west", "pickup", "dropoff"]
# Based on https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
# and https://keon.io/deep-q-learning/

env = gym.make("Taxi-v2").env

α = 0.0001	# Learn rate
ϵ = 1.0	# Randomness
γ = 0.95	# Future importance

ϵ_min = 0.001
ϵ_decay = 0.9925

model = k.Sequential()
model.add(Dense(48, input_dim=env.observation_space.n, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.075))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=α))

memory = deque(maxlen=2000)

epoch_limit = 7000

overtimes = 0
def big_array(data, size, value = 1):
    data_array = np.zeros((1, size))
    data_array[0][data] = value
    return data_array

for i in range(0, 100):       # divided by 1000
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    #frames = []

    done = False
    print("i: ", i)

    while not done and epochs <= epoch_limit:
        action = (env.action_space.sample() if (random.uniform(0, 1) < ϵ) else np.argmax(model.predict(np.array(big_array(state, env.observation_space.n)))[0]))   # Explore or exploit
        n_state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        if (epochs == epoch_limit):
            overtimes += 1
            reward -= 30
        memory.append((np.array([state]), action, reward, np.array([n_state]), done))
        #print(action)

        state = n_state
        epochs += 1

    r_batch = random.sample(memory, 128)
    for state, action, reward, n_state, done in r_batch:
        target = reward
        if not done:
            target = reward + γ * np.amax(model.predict(big_array(n_state, env.observation_space.n)))
        target_f = model.predict(big_array(state, env.observation_space.n))
        target_f[0][action] = target

        model.fit(big_array(state, env.observation_space.n), target_f, epochs=1, verbose=0)
        if ϵ > ϵ_min:
            ϵ *= ϵ_decay

    print("Fails: ", penalties)

    if i % 1000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Overtimes: ", overtimes)

total_epochs, total_penalties = 0, 0
episodes = 10

for i in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(model.predict(big_array(state, env.observation_space.n))[0])
        state, reward, done, info = env.step(action)
        print(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        if (i == episodes - 1):
            clear_output(wait=True)
            env.render()
        if epochs % 10 == 0:
            clear_output(wait=True)
            print(f"Epoch: {epochs}")

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes}:")
print(f"Timesteps per episode: {total_epochs / episodes}")
print(f"Penalties per episode: {total_penalties / episodes}")
