#nytest
import random
import gym
import keras
import numpy as np
from keras.models import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam

EPISODESSSS___ = 1000

class agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state,action,reward,next_state,done in minibatch:
            target = reward
            if not done:
                target = (reward+self.gamma*np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self ,name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)

if __name__ =='__main__':

    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = agent(state_size, action_size)

    done = False
    batch_size = 32

    for e in range(EPISODESSSS___):
        state = env.reset()
        
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1,state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode:{}/{}, score: {}, e: {:.2}".format(e, EPISODESSSS___, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            #if(e%50 == 49):
            #   env.render()
    
        
        
    
        
        
