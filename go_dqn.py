# from __future__ import print_function
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
import subprocess
from datetime import datetime

# todo: 1). Point out where is legal move on explore stage
# todo: 2). Should I use single Conv2D with large kernel
# todo: 3). What to do with 3rd plane(available move) that contains some illegal move
# todo: 4). I haven't use trained model as opponent yet
# todo: 5). Add AI model to opponent move predict(but I train with black, will it works)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.8  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.25
        self.epsilon_decay = 0.995
        self.board_size = int(self.action_size**0.5)
        self.model = self._build_model()
        self.iteration = 0
        self.f = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, 3, input_shape=self.state_size, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 1, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, 3, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, 1, padding='same'))  # input dimension = #states
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 1, padding='same'))  # output nodes = #action
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((self.board_size**2*64,)))
        model.add(Dense(self.action_size+1, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-2, decay=1e-6))
        # model.add(Dense(self.action_size+1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy',
        #               optimizer=Adam(lr=1e-2, decay=1e-6))

        print(model.summary())
        return model

    def act(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        board_size = int(self.action_size ** 0.5)
        if np.sum(state[0][2]) == 0:
            return board_size ** 2 + 1
        if np.random.rand() <= self.epsilon:
            for i in range(self.action_size):
                target = random.randrange(self.action_size-1)
                x, y = int(target/board_size), (target-(int(target/board_size)*board_size))
                if state[0][2][x][y] == 1:
                    if not (self.f == 0 and x == y == int(board_size/2)):
                        self.f = 1
                        return target
                elif np.sum(state[0][2]) == 0:
                    return self.action_size

        act_values = self.model.predict(state)
        srt = np.argsort(act_values[0])

        for i in range(self.action_size+1):
            target = srt[-1*i]
            x, y = int(target / board_size), (target - (int(target / board_size) * board_size))
            if target == self.action_size:
                continue
            if state[0][2][x][y] == 1:
                if not (self.f == 0 and x == y == int(board_size / 2)) and target != self.action_size:
                    self.f = 1
                    return target
        return self.action_size

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
        if self.epsilon > self.epsilon_min and not self.iteration % 1000:  # gradually change from explore to exploit
            self.epsilon *= self.epsilon_decay
        self.iteration += 1


if __name__ == "__main__":
    board_size = 7
    bb = Board(board_size)
    bb._turn = bb.BLACK
    env = GoEnv(player_color='black',
                # opponent='pachi:uct:_2400',
                opponent='random',
                observation_type='image3c',
                illegal_move_mode='lose',
                board_size=board_size)
    state_size = (3, board_size, board_size)
    action_size = board_size**2
    print("{} actions, {}-dim state".format(action_size, state_size))
    agent = DQNAgent(state_size, action_size)
    try:
        agent.model.load_weights('old-{}.h5'.format(board_size))
    except:
        pass
    batch_size = 32
    illegal = 0
    e = 0
    dlwin = [0, 0]
    pachiwin = [0, 0]
    with open('log-{}.txt'.format(board_size), 'a') as wr:
        wr.write(datetime.now().__str__() + '\n\n')
    while True:
        if e % 100 == 0:
            env.opponent = 'pachi:uct:_2400'
        else:
            env.dl_model = agent.model
            env.dl_model.load_weights('old-{}.h5'.format(board_size))
            env.opponent = 'dl'
        e += 1

        state = env.reset()
        agent.f = 0
        # state = np.rollaxis(state, 0, 3)
        state = np.array([state])
        success = False
        for i in range(2000):
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
            subprocess.check_call('clear', shell=True)
            print("episode: {}, dl_win: {}/{}, pachi_win: {}/{}, e: {:.2}, illegal: {}"
                  .format(e,
                          int(dlwin[0]), int(dlwin[1]),
                          int(pachiwin[0]), int(pachiwin[1]),
                          agent.epsilon,
                          # board_size - int(action/board_size),
                          # chr(action-(int(action/board_size)*board_size)+1+64),
                          illegal))
            env.render()
            with open('log-{}.txt'.format(board_size), 'a') as wr:
                wr.write(repr(env.state) + '\n\n')
            actions = np.argmax(action)+1
            next_state, reward, done, _, isillegal = env.step(action)
            illegal += isillegal
            # reward = reward if not done else -10
            # next_state = np.rollaxis(next_state, 0, 3)
            next_state = np.array([next_state])
            agent.remember(state, action-1, reward, next_state, done)
            state = next_state
            if reward == -1:
                reward = 0
            elif reward == 1:
                if env.opponent == 'dl':
                    dlwin[0] += reward
                else:
                    pachiwin[0] += reward
            if done:
                if reward == 0:
                    if env.opponent == 'dl':
                        dlwin[1] += 1
                    else:
                        pachiwin[1] += 1
                bb = Board(board_size)
                # print("episode: {}, action: ({},{}), win: {}/{}, e: {:.2}, illegal: {}"
                #       .format(e,
                #               board_size - int(action/board_size),
                #               chr(action-(int(action/board_size)*board_size)+1+64),
                #               int(win), int(lose),
                #               agent.epsilon,
                #               illegal))
                # env.render()
                break

        if len(agent.memory) > batch_size:
            agent.model.save_weights('old-{}.h5'.format(board_size))
            agent.replay(batch_size)
