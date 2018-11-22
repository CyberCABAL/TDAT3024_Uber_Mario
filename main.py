from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import numpy as np
import random
from collections import deque
import keras as k
from sys import argv
import resource
import cv2

from stack_queue import StackQueue
import compression

from multiprocessing.pool import ThreadPool


def memory_use():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

#import threading
#import overwrite

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


def make_env(world, level, v = "v1"):
    env_0 = gym_super_mario_bros.make("SuperMarioBros-" + str(world) + "-" + str(level) + "-" + v)	#Same as gym.make
    return BinarySpaceToDiscreteSpaceEnv(env_0, Moves)


def make_random_env(v = "v1"):
    return make_env(1, random.randint(1, 3))


env = make_random_env()
action_amount = env.action_space.n

pool = ThreadPool(processes=3)

α = 0.00025	# Learn rate
ϵ = 1.0		# Randomness
γ = 0.9875	# Future importance

ϵ_min = 0.2
ϵ_decay = 0.999925

# 16x16 = 1 cell
sub_bottom = 12
sub_top = 52  #80
sub_right = 22
sub_left = 14

x_state = 256
y_state = 240

calc1 = x_state - sub_right
calc2 = y_state - sub_bottom

resize_factor = 0.25    # 1/n² the amount of RAM used

x_state_r = int((calc1 - sub_left) * resize_factor)
y_state_r = int((calc2 - sub_top) * resize_factor)

memory_size = 175000
batch_size = 128
stack_amount = 3
stack_axis = 2

visual = False
colour = False

upd = 0
seen = 0
render_limit = 2500

#continue_learning = True

net_input_shape = (y_state_r, x_state_r, stack_amount)


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

    model.add(k.layers.Conv2D(filters=32, kernel_size=[8, 8], kernel_initializer='glorot_uniform', strides=[4, 4], padding="VALID", activation='relu',
                              name="c0", input_shape=net_input_shape))
    model.add(k.layers.BatchNormalization(epsilon=0.000001, axis=1))
    model.add(k.layers.Conv2D(filters=64, kernel_size=[4, 4], kernel_initializer='glorot_uniform', strides=[2, 2], padding="VALID", activation='relu',
                              name="c1"))
    model.add(k.layers.BatchNormalization(epsilon=0.000001, axis=1))
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
    print("Target updated, update counter reset to", upd)
    target.set_weights(network.get_weights())


def step(env_0, stacked_state, network, ϵ):
    """if (random.uniform(0, 1) < ϵ):
        action = env.action_space.sample()
    else:
        action = np.argmax(network.predict(np.array([stacked_state])))
        print("Action:", action)"""
    action = env_0.action_space.sample() if (random.uniform(0, 1) < ϵ) else np.argmax(network.predict(np.array([stacked_state])))
    return action, env_0.step(action)


def replay(batches, network, target_network):
    r_batch = memory.random_stacks(stack_axis, batches)
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


def save(data, memory_0):
    memory_0.append((compression.run_length_c(data[0], colour), data[1], data[2],
                     compression.run_length_c(data[3], colour), data[4]))


def simulation(env_0, network, render):
    stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)

    global upd, memory
    state = preprocess(env_0.reset())
    for n in range(stack_amount):
        stack.append(state)
    done = False
    max_x = 0
    while not done:
        upd += 1
        action, data = step(env_0, stack_the_state(stack), network, ϵ)
        n_state, reward, done, info = data
        n_state = preprocess(n_state)

        stack.append(n_state)
        pool.apply_async(save, ((state, action, reward, n_state, done), memory))    # Do compression in a thread.
        #save(data)

        state = n_state

        pos = info["x_pos"]
        if pos > max_x:
            max_x = pos
        if render:
            env_0.render()

    return max_x


def learn(n, t):
    global upd
    if upd >= update_freq:
        upd = 0
        update_target(n, t)

    if len(memory) > batch_size:
        replay(batch_size, n, t)
        global seen
        seen += batch_size


#def reset_env(env_0):
#    env_0.close()
#    return make_random_env()


def multiple_sim(amount, network, target):
    global ϵ, env
    render = False
    for i in range(0, amount):
        if i > render_limit:
            render = True
        max_x = simulation(env, network, render)

        learn(network, target)

        if ϵ > ϵ_min:
            ϵ *= ϵ_decay

        if i % 1000 == 0:
            network.save_weights("weights")
            target.save_weights("t_weights")

        #env = reset_env(env)
        env.close()
        env = make_random_env()

        #print(i, "Best x position:", max_x, "Memory use:", memory_use(), "Queue:", len(memory), "ϵ =", "{0:.3f}".format(ϵ))
        print(i, "Best x:", max_x, "Learn Frames:", seen, "Memory:", memory_use(), "Queue:", len(memory), "ϵ =",
              "{0:.3f}".format(ϵ))
        #th.join()


def tests(network, episodes=5):
    stack = deque([np.zeros((y_state_r, x_state_r)) for i in range(stack_amount)], maxlen=stack_amount)

    global env
    for i in range(episodes):
        state = preprocess(env.reset())
        for n in range(stack_amount):
            stack.append(state)

        done = False
        while not done:
            action, data = step(env, stack_the_state(stack), network, 0)
            state, reward, done, info = env.step(action)
            stack.append(preprocess(state))
            env.render()

        print(i, "Final x:", info["x_pos"])
        env.close()
        env = make_random_env()

    return state


if __name__ == "__main__":
    train = False

    if "-v" in argv:
        visual = True
    if "-t" in argv:
        train = True
    if "-c" in argv:
        colour = True
        net_input_shape = (y_state_r, x_state_r, stack_amount * 3)
        stack_axis = 3

    Q = get_model()
    
    if train:
        Q_target = get_model()
        #Q.load_weights("weights")
        #Q_target.load_weights("t_weights")
        memory = StackQueue(memory_size, stack_amount, (y_state_r, x_state_r), colour)
        update_freq = 75000
        multiple_sim(100001, Q, Q_target)
    else:
        Q.load_weights("weights")

    """colour = True
    test = env.reset()
    c = compression.run_length_c(test, colour)
    d = compression.run_length_d(c)
    print((test == d).all())"""

    """test = preprocess(env.reset())
    n_state, reward, done, info = env.step(0)
    test = compression.run_length_c(test, colour)
    n = compression.run_length_c(preprocess(n_state), colour)

    for i in range(memory_size):
        memory.append((test, 0, reward, n, done))
        if i % 10000 == 0:
            print("Memory:", memory_use())

    print("Memory:", memory_use())"""

    last_state = tests(Q, 10)

    cv2.imwrite("Result.png", last_state)

    print("Memory:", memory_use())

    cv2.destroyAllWindows()
    if train:
        Q.save_weights("weights_final")
        Q_target.save_weights("t_weights_final")

    env.close()

