import random
from collections import deque

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQN:
    def __init__(self, action_space, model: Sequential):
        # Learning Parameters
        self.exploration_rate = EXPLORATION_MAX

        # Memory
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Model <TODO try others
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        # Append current environment to memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Get next move
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def cartpole():
    return

def cartpole_dqn():

    # initialize environment
    env = gym.make(ENV_NAME, render_mode='human')
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    # configure model
    model = Sequential([
        tf.keras.layers.Input(shape=(1,)),
        Dense(24, activation="relu"),
        Dense(24, activation="relu"),
        Dense(action_space, activation="linear"),
    ])
    model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

    # construct solver object
    dqn_solver = DQN(action_space, model)

    after_training = "after_training.mp4"
    video = VideoRecorder(env, after_training)

    for run in range(10):
        # initialize/reset state
        state = env.reset()

        step = terminal = truncated = 0
        while not (terminal or truncated):
            action = dqn_solver.act(state)
            state_next, reward, truncated, terminal, info = env.step(action)
            reward = reward if not (terminal or truncated) else -reward
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            dqn_solver.experience_replay()
            step += 1

        print("Run: " + str(run) + ", exploration: " +
              str(dqn_solver.exploration_rate) + ", score: " + str(step))

    #video.close()
    env.close()

#TODO SETUP METRICS --- HOW TO EVALUATE?

if __name__ == "__main__":
    cartpole_dqn()
