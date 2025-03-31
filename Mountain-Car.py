import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import gym
import random

from dqn import dqn
from utils import train
from replay_buffer import replay_buffer

tf.random.set_seed(1234)

def epi_greedy(epsilon, state, q_network, action_size):
    """Epsilon-greedy policy for action selection."""
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)  
    else:
        q_values = q_network(np.array([state])) 
        return np.argmax(q_values.numpy())  

def main():
    optimizer = optimizers.Adam(learning_rate=0.02)
    score_arr = np.array([])
    t_ = np.array([])
    
    env = gym.make('MountainCar-v0', render_mode="human")  
    env.reset(seed=1234)

    q = dqn(env.action_space.n)
    q_target = dqn(env.action_space.n)
    q.build(input_shape=(None, env.observation_space.shape[0]))
    q_target.build(input_shape=(None, env.observation_space.shape[0]))

    memory = replay_buffer()
    copy_round = 20
    score = 0.0

    for src, dest in zip(q.variables, q_target.variables):
        dest.assign(src)

    for n_epi in range(1, 3000):
        s, _ = env.reset(seed=1234)
        for t in range(1200):
            if n_epi > 1500:
                try:
                    env.render()
                except Exception as e:
                    print(f"Render Error: {e}")
                    pass  

            a = epi_greedy(max(0.1, 0.4 - 0.01 * (n_epi / 200)), s, q, env.action_space.n)
            s_prime, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated  

            pos, vel = s_prime
            r = 3 * abs(pos + 0.5)

            done_mask = int(not done)
            memory.add_to_buffer(s, r, a, s_prime, done_mask)
            s = s_prime
            score += r

            if done:
                break

        if memory.size() > 5000:
            train(memory, q, q_target, 0.99, 64, optimizer)

        if n_epi % 60 == 0:
            for src, dest in zip(q.variables, q_target.variables):
                dest.assign(src)

        if n_epi % copy_round == 0:
            print(n_epi, score / copy_round)
            t_ = np.append(t_, n_epi)
            score_arr = np.append(score_arr, score / copy_round / 3)
            score = 0.0

    env.close()
    plt.plot(t_, score_arr)
    plt.show()

if __name__ == '__main__':
    main()

