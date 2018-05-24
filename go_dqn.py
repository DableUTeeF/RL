from __future__ import print_function
from go.board import Board
from keras import backend as K
import random
import numpy as np
from goes import GoEnv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, Reshape
from keras.optimizers import Adam, SGD
import pachi_py
K.set_image_data_format('channels_first')

# todo: 1). Point out where is legal move on explore stage
# todo: 2). Should I use single Conv2D with large kernel
# todo: 3). What to do with 3rd plane(available move) that contains some illegal move
# todo: 4). I haven't use trained model as opponent yet


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
        model.add(Conv2D(64, 3, input_shape=self.state_size, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 1, activation='sigmoid', padding='same'))  # output nodes = #action
        model.add(Reshape((self.action_size*64,)))
        model.add(Dense(self.action_size+2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=1e-2, decay=1e-5, momentum=0.9))

        print(model.summary())
        return model

    def act(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        i = 0
        if np.random.rand() <= self.epsilon:
            if np.random.rand() <= 1:
                board_size = int(self.action_size**0.5)
                while True:
                    target = random.randrange(self.action_size)
                    x, y = int(target/board_size), (target-(int(target/board_size)*board_size))
                    if state[0][2][x][y] == 1:
                        return target
                    elif np.sum(state[0][2]) == 0:
                        print('No valid move left')
                        return self.action_size

                    i += 1
            else:
                return random.randrange(1, self.action_size)
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
    board_size = 7
    bb = Board(board_size)
    bb._turn = bb.BLACK
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
        # i = 0
        while True:
            # env.render()
            # bb._turn = bb.BLACK
            # for row in range(board_size):
            #     for col in range(board_size):
            #         if state[0][0][row][col] == 1:
            #             bb._array[row][col] = bb.BLACK
            #         elif state[0][1][row][col] == 1:
            #             bb._array[row][col] = bb.WHITE
            # for row in range(board_size):
            #     for col in range(board_size):
            #         if bb.count_liberties(row+1, col+1) == 0:
            #             state[0][2][row][col] = 0
            for move in range(action_size):
                try:
                    env.state.act(move)
                except pachi_py.IllegalMove:
                    x, y = int(move/board_size), (move-(int(move/board_size)*board_size))
                    state[0][2][x][y] = 0
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
                bb = Board(board_size)
                env.render()
                print("episode: {}/{}, action: ({},{}), e: {:.2}, illegal: {}"
                      .format(e+1, emax,
                              board_size - int(action/board_size),
                              chr(action-(int(action/board_size)*board_size)+1+64),
                              agent.epsilon,
                              illegal))  # score == time
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
