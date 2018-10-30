from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import random
from collections import deque
import keras as k
import overwrite

Moves = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['down'],
    ['up'],
]

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v1")	#Same as gym.make
env = BinarySpaceToDiscreteSpaceEnv(env, Moves)

action_amount = env.action_space.n

α = 0.001	# Learn rate
ϵ = 1.0		# Randomness
γ = 0.999	# Future importance

ϵ_min = 0.025
ϵ_decay = 0.975

sub_bottom = 26
sub_top = 26
sub_right = 0
sub_left = 0

x_state = 256
y_state = 240
x_state_reduced = x_state - sub_right - sub_left
y_state_reduced = y_state - sub_bottom - sub_top

dim = y_state_reduced * x_state_reduced



# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def preprocess(image):
    if (sub_bottom > 0):
        if (sub_left > 0):
            temp = image[sub_top: -sub_bottom, sub_right: -sub_left]  # [:, subtract_top : -subtract_bottom]
        else:
            temp = image[sub_top: -sub_bottom]
    elif (sub_left > 0):
        temp = image[:, sub_right: -sub_left]
    else:
        temp = image

    new_image = np.zeros(temp.shape[:2])
    for i in range(y_state_reduced):
        for j in range(x_state_reduced):
            new_image[i][j] = int(np.dot(temp[i][j][:], [0.299, 0.587, 0.114]))
    return np.reshape(new_image, dim)


model = k.Sequential()
model.add(k.layers.Dense(256, input_dim=dim))
model.add(k.layers.Dense(128, input_dim=256, activation='relu'))
#model.add(k.layers.Dense(32, input_dim=128, activation='relu'))
model.add(k.layers.Dense(action_amount, input_dim=128, activation='relu'))
model.compile(loss='mse', optimizer=k.optimizers.Adam(lr=α, clipvalue=1))	#categorical_crossentropy

memory = deque(maxlen=100000)

done = False
for i in range(0, 1000):
    state = preprocess(env.reset())
    for i in range(1000):
        action = (env.action_space.sample() if (random.uniform(0, 1) < ϵ) else (np.argmax(model.predict(np.array([state])))))
        n_state, reward, done, info = env.step(action)
        proc_n_state = preprocess(n_state)
        env.render()

        memory.append((state, action, reward, proc_n_state, done))
        #print(reward)
        state = proc_n_state
        if done:
            state = preprocess(env.reset())

    r_batch = random.sample(memory, 32)
    for state, action, reward, n_state, done in r_batch:
        target = reward
        if not done:
            target = reward + γ * np.amax(model.predict(np.array([n_state])))
        target_f = model.predict(np.array([state]))
        target_f[0][action] = target

        model.fit(np.array([state]), target_f, epochs=3, verbose=0)
        if ϵ > ϵ_min:
            ϵ *= ϵ_decay

episodes = 10

for i in range(episodes):
    state = env.reset()
    epochs = 0

    done = False

    while not done:
        action = np.argmax(model.predict(np.array([list(env.decode(state))]))[0])
        state, reward, done, info = env.step(action)

        epochs += 1
        if (i == episodes - 1):
            env.render()
        if (epochs % 100 == 0):
            print(f"Epoch: {epochs}")

env.close()
