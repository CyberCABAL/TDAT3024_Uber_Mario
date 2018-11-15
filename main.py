from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import random
from collections import deque
import keras as k
from sys import argv
#from sys import getsizeof
import threading
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

α = 0.00025	# Learn rate
ϵ = 1.0		# Randomness
γ = 0.975	# Future importance

ϵ_min = 0.125
ϵ_decay = 0.9975

# 16x16 = 1 cell
sub_bottom = 28
sub_top = 64  #80
sub_right = 16
sub_left = 8

x_state = 256
y_state = 240

calc1 = x_state - sub_right
calc2 = y_state - sub_bottom

resize_factor = 0.25    # 1/n² the amount of RAM used, little information lost

x_state_r = int((calc1 - sub_left) * resize_factor)
y_state_r = int((calc2 - sub_top) * resize_factor)

stack_amount = 3
visual = False
colour = False

upd = 0

#continue_learning = True

net_input_shape = (y_state_r, x_state_r, stack_amount)
stack_axis = 2


def preprocess(image):
    if colour:
        new_image = np.array(cv2.resize(
            image[sub_top: calc2, sub_left: calc1], None, fx=resize_factor, fy=resize_factor), dtype="uint8")
    else:
        new_image = np.array(cv2.cvtColor(cv2.resize(
            image[sub_top: calc2, sub_left: calc1], None, fx=resize_factor, fy=resize_factor), cv2.COLOR_RGB2GRAY), dtype="uint8")
    if visual:
        cv2.imshow("Vision", new_image)
        cv2.waitKey(1)

    return new_image


def get_model():
    model = k.Sequential()

    model.add(k.layers.Conv2D(filters=32, kernel_size=[4, 4], kernel_initializer='glorot_uniform', strides=[4, 4], padding="VALID", activation='relu',
                              name="c0", input_shape=net_input_shape))
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


def update_target(network, target):
    target.set_weights(network.get_weights())
    print("Target updated")


def step(stacked_state, network):
    action = env.action_space.sample() if (random.uniform(0, 1) < ϵ) else np.argmax(network.predict(np.array([stacked_state])))
    return action, env.step(action)


def replay(batches, network, target_network):
    #while batches >= memory_size:
    #    batches = int(batches / 2)
    r_batch = random.sample(memory, batches)
    for state, action, reward, n_state, done in r_batch:
        if visual:
            if colour:
                cv2.imshow("Replay", state[:, :, :3])
            else:
                cv2.imshow("Replay", state)
            cv2.waitKey(1)
        target = reward
        q_target_value = network.predict(np.array([state]))
        if not done:
            target += γ * target_network.predict(np.array([n_state]))[0][np.argmax(network.predict(np.array([n_state])))]
        q_target_value[0][action] = target

        network.fit(np.array([state]), q_target_value, epochs=1, verbose=0)


def stack_the_state(stack_0):
    res_state = np.stack(stack_0, axis=stack_axis)
    return res_state if not colour else np.reshape(res_state, (y_state_r, x_state_r, stack_amount * 3))


def simulation(network):
    stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)

    global memory, upd
    state = preprocess(env.reset())
    for n in range(stack_amount):
        stack.append(state)
    stack_state = stack_the_state(stack)
    done = False
    max_x = 0
    while not done:
        action, data = step(stack_state, network)
        n_state, reward, done, info = data
        proc_n_state = preprocess(n_state)

        stack.append(proc_n_state)
        stack_state_n = stack_the_state(stack)

        pos = info["x_pos"]

        memory.append((stack_state, action, reward, stack_state_n, done))
        #state = proc_n_state
        stack_state = stack_state_n

        if pos > max_x:
            max_x = pos

        env.render()
        upd += 1

    return max_x


def learn(n, t):
    global upd
    if upd > update_freq:
        update_target(n, t)
        upd = 0

    if len(memory) > 256:
        replay(256, n, t)



def multiple_sim(amount, network, target):
    for i in range(0, amount):
        #th = threading.Thread(target=learn, args=(network, target))
        #th.start()
        max_x = simulation(network)

        learn(network, target)

        global ϵ
        if ϵ > ϵ_min:
            ϵ *= ϵ_decay

        print("i:", i)
        print("Best x position:", max_x)
        #th.join()
        # print("Queue size:", getsizeof(memory), "Elements:", len(memory))


def tests(episodes=5):
    stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)

    for i in range(episodes):
        state = env.reset()
        for n in range(stack_amount):
            stack.append(preprocess(state))
        stacked_state = stack_the_state(stack)

        done = False

        while not done:
            action = np.argmax(Q.predict(np.array([stacked_state])))
            state, reward, done, info = env.step(action)
            stack.append(preprocess(state))
            stacked_state = stack_the_state(stack)
            env.render()

        print("i:", i)
        print("x-position:", info["x_pos"])

    return state


if __name__ == "__main__":
    if "-v" in argv:
        visual = True
    if "-c" in argv:
        colour = True
        net_input_shape = (y_state_r, x_state_r, stack_amount * 3)
        stack_axis = 3

    Q = get_model()
    Q_target = get_model()

    update_freq = 25000
    memory = deque(maxlen=125000)

    multiple_sim(1000, Q, Q_target)

    last_state = tests(10)

    cv2.imwrite("Result.png", last_state)

    env.close()
    cv2.destroyAllWindows()

    Q.save_weights("weights")
    Q_target.save_weights("t_weights")
