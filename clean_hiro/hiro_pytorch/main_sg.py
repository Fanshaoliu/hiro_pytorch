import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ['CUDA_VISIBLE_DEVICE'] = "4"
CUDA_VISIBLE_DEVICES=4
import torch
torch.cuda.set_device(4)

import argparse
import numpy as np
import datetime
import copy
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.hiro_utils import Subgoal
from hiro.utils import Logger, _is_update, record_experience_to_csv, listdirs
from hiro.models import HiroAgent, TD3Agent
import safety_gym
import yaml
import gym
from time import time


def run_evaluation(args, env, agent):
    agent.load(args.load_episode)

    rewards, success_rate = agent.evaluate_policy(env, args.eval_episodes, args.render, args.save_video, args.sleep)

    print('mean:{mean:.2f}, \
            std:{std:.2f}, \
            median:{median:.2f}, \
            success:{success:.2f}'.format(
        mean=np.mean(rewards),
        std=np.std(rewards),
        median=np.median(rewards),
        success=success_rate))

def run_evaluation_sg(args, env, agent, eval_epochs=10):
    agent.load(args.load_episode)

    obs = env.reset()
    # print("obs: \n", obs)
    done = False
    ep_ret = 0
    ep_cost = 0

    num_step = 0
    global_step = 0
    last_action = env.action_space.sample()

    results = []

    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f \t Episode num_step: %.3f'%(ep_ret, ep_cost, num_step))
            # results.append([i, ep_ret, ep_cost, num_step])
            results.append([ep_ret, ep_cost, num_step])
            ep_ret, ep_cost = 0, 0
            obs = env.reset()
            num_step = 0
        assert env.observation_space.contains(obs)
        # act = env.action_space.sample()

        a, r, n_s, done = agent.step(obs, env, num_step, global_step, explore=False)

        obs = n_s

        # act = 0.5 * act + 0.5 * last_action
        # last_action = a

        assert env.action_space.contains(a)
        # obs, reward, done, info = env.step(a)
        # print("info: \n", info)

        num_step += 1
        global_step += 1
        # print('reward', reward)
        ep_ret += r
        # ep_cost += info.get('cost', 0)

        agent.end_step()

        env.render()

    return results

def run_tra_collect(args, env, agent, eval_epochs=10):
    agent.load(args.load_episode)

    ts = []  # tragectory_saver
    fts = []  # final tragectory_saver

    obs = env.reset()
    # print("obs: \n", obs)
    done = False
    ep_ret = 0
    ep_cost = 0

    num_step = 0
    global_step = 0
    last_action = env.action_space.sample()

    results = []

    n = 0

    while n < eval_epochs:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f \t Episode num_step: %.3f'%(ep_ret, ep_cost, num_step))
            # results.append([i, ep_ret, ep_cost, num_step])
            n += 1
            results.append([ep_ret, ep_cost, num_step])
            ep_ret, ep_cost = 0, 0
            obs = env.reset()
            num_step = 0
        assert env.observation_space.contains(obs)
        # act = env.action_space.sample()

        a, r, n_s, done, info = agent.step(obs, env, num_step, global_step, explore=False)

        ts.append([obs, a, r, info['cost'], n_s])

        obs = n_s

        # act = 0.5 * act + 0.5 * last_action
        # last_action = a

        assert env.action_space.contains(a)
        # obs, reward, done, info = env.step(a)
        # print("info: \n", info)

        num_step += 1
        global_step += 1
        # print('reward', reward)
        ep_ret += r
        # ep_cost += info.get('cost', 0)

        agent.end_step()

        env.render()

    np.save("offline_data/", args.env + "_tragectory.npy", np.array(fts))

    return results

class Trainer():
    def __init__(self, args, env, agent, experiment_name):
        self.args = args
        self.env = env
        self.agent = agent
        log_path = os.path.join(args.log_path, experiment_name)
        self.logger = Logger(log_path=log_path)

    def train(self):
        global_step = 0
        start_time = time()
        for e in np.arange(self.args.num_episode) + 1:
            
            obs = self.env.reset()
            # fg = obs['desired_goal']
            # print("self.env.goal_pos: \n", self.env.goal_pos)
            fg = self.env.goal_pos[:2]

            # s = obs['observation']
            s = obs

            done = False

            step = 0
            episode_reward = 0
            # episode_cost

            self.agent.set_final_goal(fg)

            while not done:
                # Take action
                a, r, n_s, done, info = self.agent.step(s, self.env, step, global_step, explore=True)

                # Append
                self.agent.append(step, s, a, n_s, r, done)

                # Train
                losses, td_errors = self.agent.train(global_step)

                # Log
                self.log(global_step, [losses, td_errors])

                # Updates
                s = n_s
                # episode_cost += c
                episode_reward += r
                step += 1
                global_step += 1
                self.agent.end_step()

            self.agent.end_episode(e, self.logger)

            if e % 10 == 0:
                end_time = time()
                # print("Epoch: ",e , "Reward: ", episode_reward, "Cost: ", episode_cost, "Time consuming: ", int(end_time-start_time))
                print("Epoch: %d" % (e), "Reward: %.3f" % (episode_reward),
                      "Time consuming: %d" % (int(end_time - start_time)))
                start_time = time()

            self.logger.write('reward/Reward', episode_reward, e)

            # Safe-gym donot have evaluate dring training
            # self.evaluate(e)

    def log(self, global_step, data):
        losses, td_errors = data[0], data[1]

        # Logs
        if global_step >= self.args.start_training_steps and _is_update(global_step, args.writer_freq):
            for k, v in losses.items():
                self.logger.write('loss/%s' % (k), v, global_step)

            for k, v in td_errors.items():
                self.logger.write('td_error/%s' % (k), v, global_step)

    def evaluate(self, e):
        # Print
        if _is_update(e, args.print_freq):
            agent = copy.deepcopy(self.agent)
            rewards, success_rate = agent.evaluate_policy(self.env)
            # rewards, success_rate = self.agent.evaluate_policy(self.env)
            self.logger.write('Success Rate', success_rate, e)

            print(
                'episode:{episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}, success:{success:.2f}'.format(
                    episode=e,
                    mean=np.mean(rewards),
                    std=np.std(rewards),
                    median=np.median(rewards),
                    success=success_rate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--sleep', type=float, default=-1)
    parser.add_argument('--eval_episodes', type=float, default=5, help='Unit = Episode')
    parser.add_argument('--env', default='Safexp-PointGoal0-v0', type=str)
    parser.add_argument('--td3', action='store_true')

    # Training
    parser.add_argument('--num_episode', default=25000, type=int)
    parser.add_argument('--steps_per_epoch', default=30000, type=int)

    parser.add_argument('--start_training_steps', default=2500, type=int, help='Unit = Global Step')
    parser.add_argument('--writer_freq', default=25, type=int, help='Unit = Global Step')
    # Training (Model Saving)
    parser.add_argument('--subgoal_dim', default=15, type=int)
    parser.add_argument('--load_episode', default=7000, type=int)
    parser.add_argument('--model_save_freq', default=2000, type=int, help='Unit = Episodes')
    parser.add_argument('--print_freq', default=250, type=int, help='Unit = Episode')
    parser.add_argument('--exp_name', default=None, type=str)
    # Model
    parser.add_argument('--model_path', default='model', type=str)
    parser.add_argument('--log_path', default='log', type=str)
    parser.add_argument('--policy_freq_low', default=2, type=int)
    parser.add_argument('--policy_freq_high', default=2, type=int)
    # Replay Buffer
    parser.add_argument('--buffer_size', default=200000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--buffer_freq', default=10, type=int)
    parser.add_argument('--train_freq', default=10, type=int)
    parser.add_argument('--reward_scaling', default=0.1, type=float)
    args = parser.parse_args()

    # Select or Generate a name for this experiment
    if args.exp_name:
        experiment_name = args.exp_name
    else:
        if args.eval:
            # choose most updated experiment for evaluation
            dirs_str = listdirs(args.model_path)
            dirs = np.array(list(map(int, dirs_str)))
            experiment_name = dirs_str[np.argmax(dirs)]

            experiment_name = "Safexp-PointGoal0-v0-para1"
        else:
            experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(experiment_name)

    # Environment and its attributes
    # 1. env load for antmaze
    # env = EnvWithGoal(create_maze_env(args.env), args.env)
    # 2. env load for safegym
    env = gym.make(args.env)

    goal_dim = 2
    # state_dim = env.state_dim
    # action_dim = env.action_dim

    print("env observation_space", env.observation_space.shape[0])  # 28
    print("env action_space", env.action_space.shape[0])  # 2

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    env._max_episode_steps = args.steps_per_epoch
    print("max_episode_steps: ", env._max_episode_steps)

    scale = env.action_space.high * np.ones(action_dim)

    # Spawn an agent
    if args.td3:
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            scale=scale,
            model_save_freq=args.model_save_freq,
            model_path=os.path.join(args.model_path, experiment_name),
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            start_training_steps=args.start_training_steps
        )
    else:
        agent = HiroAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=args.subgoal_dim,
            scale_low=scale,
            start_training_steps=args.start_training_steps,
            model_path=os.path.join(args.model_path, experiment_name),
            model_save_freq=args.model_save_freq,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            buffer_freq=args.buffer_freq,
            train_freq=args.train_freq,
            reward_scaling=args.reward_scaling,
            policy_freq_high=args.policy_freq_high,
            policy_freq_low=args.policy_freq_low
        )

    # Run training or evaluation
    if args.train:
        # save para
        para_dict = {}
        for k in list(vars(args).keys()):
            para_dict[str(k)] = str(vars(args)[k])

        if not os.path.exists("model/" + experiment_name):
            os.makedirs("model/" + experiment_name)
        with open("model/" + experiment_name + "/para.yml", 'w') as f:
            yaml.dump(para_dict, f)

        # Record this experiment with arguments to a CSV file
        record_experience_to_csv(args, experiment_name)
        # Start training
        trainer = Trainer(args, env, agent, experiment_name)
        trainer.train()
    if args.eval:
        # run_evaluation(args, env, agent)
        args.load_episode = args.load_episode
        results = run_evaluation_sg(args, env, agent, 5)
