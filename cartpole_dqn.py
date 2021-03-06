from __future__ import print_function

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))  # input dimension = #states
        model.add(Dense(self.action_size, activation='linear'))  # output nodes = #action
        model.compile(loss='mse', optimizer=Adam(lr=1e-2, decay=1e-5))

        print(model.summary())
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # another method to explore/exploit
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):  # done==True if this is the ending move
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        X, Y = [], []
        minibatch = random.sample(self.memory,
                                  batch_size)  # random action yields better result compared to sequential processing
        for state, action, reward, next_state, done in minibatch:
            target_f = self.model.predict(state)
            target_f[0][action] = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            X.append(state[0])
            Y.append(target_f[0])
        self.model.train_on_batch(np.array(X), np.array(Y))

        if self.epsilon > self.epsilon_min:  # gradually change from explore to exploit
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("{} actions, {}-dim state".format(action_size, state_size))

    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    emax = 5000
    for e in range(emax):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        success = False
        for time in range(200):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, emax, time, agent.epsilon))  # score == time
                if time > 195:
                    success = True
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if success:
            success_count += 1
        else:
            success_count = 0
        if success_count > 100:
            print('Cartpole-v0 SOLVED!!!')
            break
