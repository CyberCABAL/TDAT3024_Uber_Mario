from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import random
from collections import deque
import keras as k
import overwrite

#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb

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

α = 0.0001	# Learn rate
ϵ = 1.0		# Randomness
γ = 0.95	# Future importance

ϵ_min = 0.05
ϵ_decay = 0.995

sub_bottom = 15
sub_top = 80  #26
sub_right = 0
sub_left = 0

x_state = 256
y_state = 240
x_state_r = x_state - sub_right - sub_left
y_state_r = y_state - sub_bottom - sub_top

#dim = y_state_r * x_state_r

stack_amount = 3
#dim_n = dim * stack_amount

# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def preprocess(image):
    if (sub_bottom > 0):
        if (sub_left > 0):
            temp = image[sub_top : -sub_bottom, sub_right : -sub_left]  # [:, subtract_top : -subtract_bottom]
        else:
            temp = image[sub_top : -sub_bottom]
    elif (sub_left > 0):
        temp = image[:, sub_right : -sub_left]
    else:
        temp = image

    new_image = np.zeros(temp.shape[:2])
    for i in range(y_state_r):
        for j in range(x_state_r):
            new_image[i][j] = int(np.dot(temp[i][j][:], [0.299, 0.587, 0.114])) / 255.0
    #return np.array([np.reshape(new_image, dim)])
    return new_image

model = k.Sequential()
#model.add(k.layers.Dense(256, input_dim=dim_n))
#model.add(k.layers.Dense(128, input_dim=256, activation='relu'))
#model.add(k.layers.Dense(32, input_dim=128))
#model.add(k.layers.Dense(action_amount, input_dim=32))
#model.compile(loss='mse', optimizer=k.optimizers.Adam(lr=α, clipvalue=1))	#categorical_crossentropy

model.add(k.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding="VALID", activation='elu',
                          name="c0", input_shape=(y_state_r, x_state_r, stack_amount)))
model.add(k.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding="VALID", activation='elu',
                          name="c1"))
model.add(k.layers.Conv2D(filters=128, kernel_size=[2, 2], strides=[2, 2], padding="VALID", activation='elu',
                          name="c2"))
model.add(k.layers.Flatten())
model.add(k.layers.Dense(512, activation='elu'))
model.add(k.layers.Dense(action_amount))

model.compile(loss='mse', optimizer=k.optimizers.Adam(lr=α, clipvalue=1))

memory = deque(maxlen=500000)

stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)
done = False
#limit_low = 50
limit_high = 250
for i in range(0, 250):
    state = preprocess(env.reset())
    for n in range(stack_amount):
        stack.append(state)
    stacked_state = np.stack(stack, axis=2)
    max_x = 0
    j = 0
    while not done and j < limit_high:
        action = (env.action_space.sample() if (random.uniform(0, 1) < ϵ) else (np.argmax(model.predict(np.array([stacked_state])))))
        n_state, reward, done, info = env.step(action)
        proc_n_state = preprocess(n_state)

        stack.append(proc_n_state)
        stacked_state_n = np.stack(stack, axis=2)

        pos = info["x_pos"]
        if (pos > max_x):
            max_x = pos

        memory.append((stacked_state, action, reward, stacked_state_n, done))
        state = proc_n_state
        stacked_state = stacked_state_n
        if done:
            state = preprocess(env.reset())
        #if (i % 10 == 0):
        env.render()
        j += 1

    print("i:", i)
    limit_high += 50
    print("Best x position:", max_x)

    r_batch = random.sample(memory, 64)
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
        action = np.argmax(model.predict(np.array([preprocess(state)])))
        state, reward, done, info = env.step(action)

        epochs += 1
        if (i == episodes - 1):
            env.render()
        if (epochs % 100 == 0):
            print(f"Epoch: {epochs}")
    print("i:", i)
    print("x-position:", info["x_pos"])

env.close()
