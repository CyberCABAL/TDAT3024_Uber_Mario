from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import random
from collections import deque
import keras as k
#import overwrite

import cv2

#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb

#https://www.researchgate.net/figure/DQN-DDQN-and-Duel-DDQN-performance-Results-were-normalized-by-subtracting-the-a-random_fig1_309738626

#https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py

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

α = 0.0015	# Learn rate
ϵ = 1.0		# Randomness
γ = 0.975	# Future importance

ϵ_min = 0.125
ϵ_decay = 0.99975

# 16x16 = 1 cell
sub_bottom = 28
sub_top = 64  #80
sub_right = 16
sub_left = 0

x_state = 256
y_state = 240

calc1 = x_state - sub_right
calc2 = y_state - sub_bottom

x_state_r = calc1 - sub_left
y_state_r = calc2 - sub_top

#dim = y_state_r * x_state_r

stack_amount = 4
#dim_n = dim * stack_amount

def preprocess(image):
    #new_image = cv2.cvtColor(image[sub_top : calc2, sub_left : calc1], cv2.COLOR_RGB2GRAY)
    #cv2.imshow("vision", new_image)
    #cv2.waitKey(1)
    #return new_image
    return np.array(cv2.cvtColor(image[sub_top : calc2, sub_left : calc1], cv2.COLOR_RGB2GRAY), dtype="uint8") # / 255

def get_model():
    model = k.Sequential()

    model.add(k.layers.Conv2D(filters=32, kernel_size=[4, 4], kernel_initializer='glorot_uniform', strides=[4, 4], padding="VALID", activation='relu',
                              name="c0", input_shape=(y_state_r, x_state_r, stack_amount)))
    model.add(k.layers.BatchNormalization(epsilon=0.000001, axis=1))
    model.add(k.layers.Conv2D(filters=64, kernel_size=[4, 4], kernel_initializer='glorot_uniform', strides=[2, 2], padding="VALID", activation='relu',
                              name="c1"))
    model.add(k.layers.BatchNormalization(epsilon=0.000001, axis=1))
    # model.add(k.layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(k.layers.Dropout(0.025))
    model.add(k.layers.Conv2D(filters=128, kernel_size=[2, 2], kernel_initializer='glorot_uniform', strides=[2, 2], padding="VALID", activation='relu',
                              name="c2"))
    model.add(k.layers.BatchNormalization(epsilon=0.000001, axis=1))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(k.layers.Dense(action_amount, kernel_initializer='glorot_uniform'))

    #model.compile(loss="mse", optimizer=k.optimizers.Adam(lr=α, clipvalue=1))
    model.compile(loss="logcosh", optimizer=k.optimizers.RMSprop(lr=α, clipvalue=1))
    return model

Q = get_model()
Q_target = get_model()

def update_target():
    Q_target.set_weights(Q.get_weights())


update_freq = 75000
upd = 0
memory = deque(maxlen=50000)

stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)
#limit_low = 50
#limit_high = 1000
for i in range(0, 5000):
    state = preprocess(env.reset())
    for n in range(stack_amount):
        stack.append(state)
    stacked_state = np.stack(stack, axis=2)
    done = False
    max_x = 0
    j = 0
    while not done:
        action = (env.action_space.sample() if (random.uniform(0, 1) < ϵ) else np.argmax(Q.predict(np.array([stacked_state]))))

        n_state, reward, done, info = env.step(action)
        proc_n_state = preprocess(n_state)

        #if j >= limit_high:
        #    done = True

        stack.append(proc_n_state)
        stacked_state_n = np.stack(stack, axis=2)

        pos = info["x_pos"]
        if (pos > max_x):
            max_x = pos

        memory.append((stacked_state, action, reward, stacked_state_n, done))
        state = proc_n_state
        stacked_state = stacked_state_n

        #if (i % 10 == 0):
        env.render()
        #j += 1
        if upd % update_freq == 0:
            update_target()
        upd += 1

    print("i:", i)
    print("Best x position:", max_x)
    #print("Queue size:", len(memory))

    r_batch = random.sample(memory, 128)
    for state, action, reward, n_state, done in r_batch:
        #cv2.imshow("Replay", state)
        #cv2.waitKey(1)
        target = reward
        q_target_value = Q.predict(np.array([state]))
        if not done:
            target += γ * Q_target.predict(np.array([n_state]))[0][np.argmax(Q.predict(np.array([n_state])))]
        q_target_value[0][action] = target

        Q.fit(np.array([state]), q_target_value, epochs=1, verbose=0)
        if ϵ > ϵ_min:
            ϵ *= ϵ_decay


cv2.imwrite("Training.png", state)
episodes = 10

stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)

for i in range(episodes):
    state = env.reset()
    #epochs = 0
    for n in range(stack_amount):
        stack.append(preprocess(state))
    stacked_state = np.stack(stack, axis=2)

    done = False

    while not done:
        action = np.argmax(Q.predict(np.array([stacked_state])))
        state, reward, done, info = env.step(action)
        stack.append(preprocess(state))
        stacked_state = np.stack(stack, axis=2)

        #epochs += 1
        env.render()
    print("i:", i)
    print("x-position:", info["x_pos"])

cv2.imwrite("Result.png", state)

env.close()
cv2.destroyAllWindows()
