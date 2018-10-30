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

α = 0.0000000000000000000000000001	# Learn rate
ϵ = 1.0		# Randomness
γ = 0.999	# Future importance

ϵ_min = 0.05
ϵ_decay = 0.99985

"""def process(map_n):
    res = []
    i = 0
    for c in map_n:
        if (c != "\n" and i < 106):
            res.append(ord(c))
            i += 1
    return res
    #k.preprocessing.text.text_to_word_sequence(map_n, filters='', split='\n')"""


size = 4

model = k.Sequential()
model.add(Dense(12, input_dim=size))
#model.add(Dense(75))
#model.add(Dropout(0.01))
#model.add(Dense(8, input_dim=8))
model.add(Dense(36, input_dim=12))
model.add(Dense(18, input_dim=36))
model.add(Dense(env.action_space.n, input_dim=18, activation='softmax'))
model.compile(loss='mse', optimizer=Adam(lr=α, clipvalue=1))	#categorical_crossentropy

memory = deque(maxlen=1000000)

epoch_limit = 100000
#late_limit = 8000

overtimes = 0
"""def big_array(data, size, value = 1):
    data_array = np.zeros((1, size))
    data_array[0][data] = value
    return data_array"""

for i in range(0, 1000):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False
    print("i: ", i)

    while not done and epochs <= epoch_limit:
        #map = np.array(process(env.render("ansi").getvalue()))
        action = (env.action_space.sample() if (random.uniform(0, 1) < ϵ) else np.argmax(model.predict(np.array([list(env.decode(state))]))[0]))   # Explore or exploit
        n_state, reward, done, info = env.step(action)

        #n_map = np.array(process(env.render("ansi").getvalue()))

        if reward == -10:
            penalties += 1
            #reward += 9

        #if reward == 20:
        #    reward += 30

        #if (epochs > late_limit):
        #    reward -= 1
        #if (epochs == epoch_limit):
            #overtimes += 1
            #reward -= 50
            #print(reward, "for", action)
        memory.append((state, action, reward, n_state, done))
        #print(action)

        state = n_state
        epochs += 1

    r_batch = random.sample(memory, 128)
    for state, action, reward, n_state, done in r_batch:
        target = reward
        if not done:
            target = reward + γ * np.amax(model.predict(np.array([list(env.decode(n_state))])))
        target_f = model.predict(np.array([list(env.decode(state))]))
        target_f[0][action] = target

        model.fit(np.array([list(env.decode(state))]), target_f, epochs=3, verbose=0)
        if ϵ > ϵ_min:
            ϵ *= ϵ_decay

    print("Fails: ", penalties)
    print("Epochs: ", epochs)

    if i % 1000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

#print("Overtimes: ", overtimes)

print(model.get_weights())

total_epochs, total_penalties = 0, 0
episodes = 100

for i in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done and epochs < 10000:
        action = np.argmax(model.predict(np.array([list(env.decode(state))]))[0])
        state, reward, done, info = env.step(action)
        print(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        if (i == episodes - 1):
            clear_output(wait=True)
            env.render()
        if (epochs % 100 == 0):
            clear_output(wait=True)
            print(f"Epoch: {epochs}")

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes}:")
print(f"Timesteps per episode: {total_epochs / episodes}")
print(f"Penalties per episode: {total_penalties / episodes}")
env.close()

model.save_weights("Weight");
