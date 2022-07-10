'''
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 Train_CartPole.py
xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip=127.0.0.1 --port=8888  --allow-root
'''
import os
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import torch
from DQN_farm.DQN_CartPole import DQN
from ExperienceReplay.Transition import Transition


def main():
    SEED = 7122
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    ENV = 'CartPole-v0'
    ER_TYPE = ['EER_t', 'EER_r', 'ER', 'PER']
    Total_Episode = 100

    for ER_type in ER_TYPE:
        weight_dir = './Evaluation/%s/%s' % (ENV, ER_type)
        if not os.path.isdir(weight_dir):
            os.makedirs(weight_dir)
        i = -1
        print(ER_type)
        env = gym.make(ENV)
        env = env.unwrapped
        env = wrappers.Monitor(env, './Evaluation/%s/video_%s' % (ENV, ER_type), video_callable=lambda episode_id: True, force=True)
        if ER_type == 'EER_t':
            dqn = DQN(env.observation_space.shape[0], env.action_space.n, 'EER', sample_function='tournament')
        elif ER_type == 'EER_r':
            dqn = DQN(env.observation_space.shape[0], env.action_space.n, 'EER', sample_function='roulette_wheel')
        else:
            dqn = DQN(env.observation_space.shape[0], env.action_space.n, ER_type)

        Max_R = dqn.hp.Max_R
        R_mean = np.zeros(Total_Episode)
        R_std = np.zeros(Total_Episode)

        # environment step
        env_step = 0
        while env_step < dqn.hp.capacity:
            # reset env
            s_t = env.reset()
            # reset counter
            r, t, done = 0, 0, False
            i += 1
            while not done:
                env_step += 1
                env.render()
                # interact
                a_t = np.random.randint(0, env.action_space.n)
                s_next, r_t, done, info = env.step(a_t)
                r += r_t
                dqn.store_experience(
                    Transition(s_t, a_t, r_t, s_next, (1 if done else 0)),
                    life=0, index=i, done=done, time=t, r=r
                )
                t += 1
                # update state
                s_t = s_next
                print('env_step: %d' % env_step, end='\r')

        EPSILON, EPSILON_update, EPSILON_min = 1, 1000, 0.1
        for i_episode in range(Total_Episode):
            # reset env
            s_t = env.reset()
            # reset counter
            r, t, done = 0, 0, False
            i += 1
            while not done:
                env.render()
                # interact
                a_t = dqn.get_action(s_t, EPSILON)
                s_next, r_t, done, info = env.step(a_t)
                r += r_t
                dqn.store_experience(
                    Transition(s_t, a_t, r_t, s_next, (1 if done else 0)),
                    life=0, index=i, done=done, time=t, r=r)
                t += 1
                dqn.learn()
                EPSILON = max(EPSILON - 1/EPSILON_update, EPSILON_min)
                if t > Max_R and not done:
                    while True:
                        _, _, done, _ = env.step(0)
                        if done:
                            break
                # update state
                s_t = s_next
            r_behavior = r
            dqn.save('./Evaluation/%s/%s/%s_%s_%d' % (ENV, ER_type, ENV, ER_type, i_episode))
            # evaluation step
            R = np.zeros(10)
            for eval_step in range(10):
                s_t = env.reset()
                t = 0
                while True:
                    t += 1
                    env.render()
                    a_t = dqn.get_action_target(s_t)
                    s_next, r_t, done, info = env.step(a_t)
                    s_t = s_next
                    print('evaluation step: %d, R=%d' % (eval_step, t), end='\r')
                    if t > Max_R and not done:
                        while True:
                            _, _, done, _ = env.step(0)
                            if done:
                                break
                    if done:
                        R[eval_step] = t if t < Max_R else Max_R
                        print('evaluation step: %d, R=      ' % eval_step, end='\r')
                        break

            # record step
            R_mean[i_episode] = np.mean(R)
            R_std[i_episode] = np.std(R)
            print('i: %4d | R: (%7s, %7s), EPSILON: %0.4f' % (i_episode, str('%.3f' % R_mean[i_episode]), str('%.3f' % R_std[i_episode]), EPSILON), R, r_behavior)
        # plot
        if ER_type == 'ER':
            plt.plot(R_mean, color='black', zorder=1)
            plt.fill_between([i for i in range(Total_Episode)], R_mean - R_std, R_mean + R_std, color='#DDDDDD', alpha=0.5, zorder=0)
        elif ER_type == 'PER':
            plt.plot(R_mean, color='red', zorder=1)
            plt.fill_between([i for i in range(Total_Episode)], R_mean - R_std, R_mean + R_std, color='#FFCCCC', alpha=0.5, zorder=0)
        elif ER_type == 'EER_t':
            plt.plot(R_mean, color='green', zorder=1)
            plt.fill_between([i for i in range(Total_Episode)], R_mean - R_std, R_mean + R_std, color='#99FF99', alpha=0.5, zorder=0)
        elif ER_type == 'EER_r':
            plt.plot(R_mean, color='blue', zorder=1)
            plt.fill_between([i for i in range(Total_Episode)], R_mean - R_std, R_mean + R_std, color='#CCEEFF', alpha=0.5, zorder=0)

        axes = plt.gca()
        axes.set_ylim([-5, 250])

        np.save('./Evaluation/%s/%s_R_mean.npy' % (ENV, ER_type), np.array(R_mean))
        np.save('./Evaluation/%s/%s_R_std.npy' % (ENV, ER_type), np.array(R_std))

    img_dir = './Evaluation/%s/image' % ENV
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    plt.savefig('%s/reward.png' % img_dir)
    return


if __name__ == "__main__":
    main()
