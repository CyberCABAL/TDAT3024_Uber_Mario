from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import random
from collections import deque
import keras as k
import overwrite

import cv2

#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb

Moves = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['A'],
    ['left'],
    ['left', 'A'],
#    ['down'],
#    ['up'],
]

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v1")	#Same as gym.make
env = BinarySpaceToDiscreteSpaceEnv(env, Moves)

action_amount = env.action_space.n

α = 0.00025	# Learn rate
ϵ = 1.0		# Randomness
γ = 0.9	# Future importance

ϵ_min = 0.05
ϵ_decay = 0.99975

# 16x16 = 1 cell
sub_bottom = 30
sub_top = 112  #80
sub_right = 32
sub_left = 8

x_state = 256
y_state = 240
x_state_r = x_state - sub_right - sub_left
y_state_r = y_state - sub_bottom - sub_top

#dim = y_state_r * x_state_r

stack_amount = 4
#dim_n = dim * stack_amount

def preprocess(image):
    new_image = cv2.cvtColor(image[sub_top : y_state - sub_bottom, sub_left : x_state - sub_right], cv2.COLOR_RGB2GRAY)
    cv2.imshow("vision", new_image)
    cv2.waitKey(1)
    return new_image

model = k.Sequential()
#model.add(k.layers.Dense(256, input_dim=dim_n))
#model.add(k.layers.Dense(128, input_dim=256, activation='relu'))
#model.add(k.layers.Dense(32, input_dim=128))
#model.add(k.layers.Dense(action_amount, input_dim=32))
#model.compile(loss='mse', optimizer=k.optimizers.Adam(lr=α, clipvalue=1))	#categorical_crossentropy

model.add(k.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding="VALID", activation='elu',
                          name="c0", input_shape=(y_state_r, x_state_r, stack_amount)))
model.add(k.layers.BatchNormalization(epsilon=0.00000001))
model.add(k.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding="VALID", activation='elu',
                          name="c1"))
model.add(k.layers.Dropout(0.01))
model.add(k.layers.BatchNormalization(epsilon=0.00000001))
model.add(k.layers.Conv2D(filters=128, kernel_size=[2, 2], strides=[2, 2], padding="VALID", activation='elu',
                          name="c2"))
model.add(k.layers.BatchNormalization(epsilon=0.00000001))
model.add(k.layers.Flatten())
model.add(k.layers.Dense(512, activation='elu'))
model.add(k.layers.Dense(action_amount))

model.compile(loss='mse', optimizer=k.optimizers.Adam(lr=α, clipvalue=1))

memory = deque(maxlen=1000000)

stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)
done = False
#limit_low = 50
limit_high = 600
for i in range(0, 300):
    state = preprocess(env.reset())
    for n in range(stack_amount):
        stack.append(state)
    stacked_state = np.stack(stack, axis=2)
    max_x = 0
    j = 0
    while not done:
        if (random.uniform(0, 1) < ϵ):
            action = env.action_space.sample()
        else:
            arg = model.predict(np.array([stacked_state]))
            action = np.argmax(arg)
            #print(action)
        #action = (env.action_space.sample() if (random.uniform(0, 1) < ϵ) else (np.argmax(model.predict(np.array([stacked_state])))))

        n_state, reward, done, info = env.step(action)
        proc_n_state = preprocess(n_state)

        if j >= limit_high:
            done = True

        stack.append(proc_n_state)
        stacked_state_n = np.stack(stack, axis=2)

        pos = info["x_pos"]
        if (pos > max_x):
            max_x = pos
        if pos < 16:
           reward = -10

        memory.append((stacked_state, action, reward, stacked_state_n, done))
        state = proc_n_state
        stacked_state = stacked_state_n
        #if done:
        #    state = preprocess(env.reset())
        #if (i % 10 == 0):
        env.render()
        j += 1

    print("i:", i)
    #limit_high += 50
    print("Best x position:", max_x)

    r_batch = random.sample(memory, 128)
    for state, action, reward, n_state, done in r_batch:
        target = reward
        q_target = model.predict(np.array([state]))
        if not done:
            target = reward + γ * np.amax(model.predict(np.array([n_state])))
        q_target[0][action] = target

        model.fit(np.array([state]), q_target, epochs=3, verbose=0)
        if ϵ > ϵ_min:
            ϵ *= ϵ_decay

episodes = 5

stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)

for i in range(episodes):
    state = env.reset()
    epochs = 0
    for n in range(stack_amount):
        stack.append(preprocess(state))
    stacked_state = np.stack(stack, axis=2)

    done = False

    while not done:
        action = np.argmax(model.predict(np.array([stacked_state])))
        state, reward, done, info = env.step(action)
        stack.append(preprocess(state))
        stacked_state = np.stack(stack, axis=2)

        epochs += 1
        env.render()
        #if (epochs % 100 == 0):
        #    print(f"Epoch: {epochs}")
    print("i:", i)
    print("x-position:", info["x_pos"])

env.close()
cv2.destroyAllWindows()
