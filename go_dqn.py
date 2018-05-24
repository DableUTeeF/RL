from __future__ import print_function

from keras import backend as K
import random
import numpy as np
from goes import GoEnv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.optimizers import Adam, SGD

K.set_image_data_format('channels_first')


def sigmoid1to7(x):
    return int((K.sigmoid(x) * 6)+1)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.  # exploration rate
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.iteration = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, 3, input_shape=self.state_size))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(self.action_size, activation='sigmoid'))  # output nodes = #action
        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=1e-2, decay=1e-5, momentum=0.9))

        print(model.summary())
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # another method to explore/exploit
            return random.randrange(1, self.action_size)
        # if np.random.rand() <= self.epsilon:  # another method to explore/exploit
        #     if np.random.rand() <= 1:
        #         board_size = int(self.action_size**0.5)
        #         while True:
        #             target = random.randrange(1, self.action_size)
        #             x, y = int(target/board_size), (target-(int(target/board_size)*board_size))
        #             if state[0][2][x][y] == 1:
        #                 return target
        #     else:
        #         return random.randrange(1, self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values)

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
        if self.epsilon > self.epsilon_min and not self.iteration % 10:  # gradually change from explore to exploit
            self.epsilon *= self.epsilon_decay
        self.iteration += 1


if __name__ == "__main__":
    board_size = 9
    env = GoEnv(player_color='black',
                opponent='pachi:uct:_2400',
                observation_type='image3c',
                illegal_move_mode='lose',
                board_size=board_size)
    state_size = (3, board_size, board_size)
    action_size = board_size**2
    print("{} actions, {}-dim state".format(action_size, state_size))
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    illegal = 0
    emax = 5000
    for e in range(emax):
        state = env.reset()
        # state = np.rollaxis(state, 0, 3)
        state = np.array([state])
        success = False
        for time in range(200):
            # env.render()
            action = agent.act(state)
            # actions = np.argmax(action)+1
            next_state, reward, done, _, isillegal = env.step(action)
            illegal += isillegal
            # reward = reward if not done else -10
            # next_state = np.rollaxis(next_state, 0, 3)
            next_state = np.array([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                env.render()
                print("episode: {}/{}, action: ({},{}), e: {:.2}, illegal: {}"
                      .format(e+1, emax,
                              board_size - int(action/board_size),
                              chr(action-(int(action/board_size)*board_size)+1+64),
                              agent.epsilon,
                              illegal))  # score == time
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
