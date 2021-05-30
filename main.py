import os 
import argparse
import numpy as np
import datetime
import copy
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.hiro_utils import Subgoal 
from hiro.utils import Logger, _is_update, record_experience_to_csv, listdirs
from hiro.models import HiroAgent, TD3Agent

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


class Trainer():
    def __init__(self, args, env, agent, experiment_name):
        self.args = args
        self.env = env
        self.agent = agent 
        log_path = os.path.join(args.log_path, experiment_name)
        self.logger = Logger(log_path=log_path)

    def train(self):
        '''
        shape of variables
            agent: point
                obs: (28,)
                s  : (28,)
                a  : (2,)
                r  : ()
                fg : (2,)

        Network architecture
        TODO: 实现TD3中的min(Q1, Q2)  # 20210512 07:59

            self.actor and self.actor_target
                usage:
                    a = self.actor(states, goals)
                    n_actions = self.actor_target(n_states, n_goals)

                input_dim: state_dim + goal_dim (28+2)
                ouput_dim: action_dim (2)
                hidden_size: 64, relu
                output_action_func: tanh

            self.critic1 and self.critic2 and self.critic1_target and self.critic2_target
                usage:
                    target_Q1 = self.critic1_target(n_states, n_goals, n_actions)
                    target_Q2 = self.critic2_target(n_states, n_goals, n_actions)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q_detached = (rewards + not_done * self.gamma * target_Q).detach()

                    current_Q1 = self.critic1(states, goals, actions)
                    current_Q2 = self.critic2(states, goals, actions)

                input_dim: state_dim + goal_dim + action_dim (28+2+2)
                ouput_dim: reward_dim (2)
                hidden_size: 64, relu
                output_action_func: no
        '''
        global_step = 0
        start_time = time()
        self.env.render()
        for e in np.arange(self.args.num_episode)+1:
            self.env.render()
            obs = self.env.reset()

            fg = obs['desired_goal']
            s = obs['observation']
            done = False

            step = 0
            episode_reward = 0

            self.agent.set_final_goal(fg)

            while not done:
                # Take action
                '''
                para
                s:            状态, 由env.obs()得到
                self.env:     实例化的环境类
                ste:          p单次实验的步数
                global_step:  总步数

                logic
                1. low control
                global_step小于开始训练的步数时，low_action随机采样得到，大于时，通过self._choose_action(s, self.sg)得到
                self._choose_action就是网络

                执行动作：
                obs, r, done, info = env.step(a)  # info is about cost
                n_s = env.obs()

                2. high control
                同样的有一个探索逻辑，self._choose_subgoal(step, s, self.sg, n_s, n_pos)选择subgoal

                self._choose_subgoal:
                    if step % self.buffer_freq == 0:
                        sg = self.high_con.policy(s, self.fg)
                        其实就是网络了
                    else:
                        sg = self.subgoal_transition(s, sg, n_s, n_pos)
                        其实就是 s[:sg.shape[0]] + sg - n_pos[:sg.shape[0]]
                '''
                a, r, n_s, done, c = self.agent.step(s, self.env, step, global_step, explore=True)
                print(c)

                # Append
                '''
                append对于低级与高级buffer的实现与奖励都是不同的
                1. 低级策略的buffer和普通的dqn一样
                2. 高级策略的buffer是隔一段时间收集一次
                '''
                self.agent.append(step, s, a, n_s, r, done)

                # Train
                '''
                global_step大于训练步后
                1. 低级策略每步都训练
                2. 高级策略根据一个频率训练
                '''
                losses, td_errors = self.agent.train(global_step)

                # Log
                self.log(global_step, [losses, td_errors])
                
                # Updates
                s = n_s
                episode_reward += r
                step += 1
                global_step += 1

                '''
                end_step主要是调整奖励与sub_goal
                self.episode_subreward += self.sr  # self.sr = self.low_reward(s, self.sg, n_s)
                self.sg = self.n_sg                # self.n_sg = self._choose_subgoal(step, s, self.sg, n_s, n_pos)
                '''
                self.agent.end_step()
                
            self.agent.end_episode(e, self.logger)

            # print(step)  # 500

            if e%10==0:
                end_time = time()
                print("Epoch: ",e , "Reward: ", episode_reward, "Time consuming: ", int(end_time-start_time))
                start_time = time()

            self.logger.write('reward/Reward', episode_reward, e)
            self.evaluate(e)

    def log(self, global_step, data):
        losses, td_errors = data[0], data[1]

        # Logs
        if global_step >= self.args.start_training_steps and _is_update(global_step, args.writer_freq):
            for k, v in losses.items():
                self.logger.write('loss/%s'%(k), v, global_step)
            
            for k, v in td_errors.items():
                self.logger.write('td_error/%s'%(k), v, global_step)
    
    def evaluate(self, e):
        # Print
        if _is_update(e, args.print_freq):
            agent = copy.deepcopy(self.agent)
            rewards, success_rate = agent.evaluate_policy(self.env)
            #rewards, success_rate = self.agent.evaluate_policy(self.env)
            self.logger.write('Success Rate', success_rate, e)
            
            print('episode:{episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}, success:{success:.2f}'.format(
                    episode=e, 
                    mean=np.mean(rewards), 
                    std=np.std(rewards), 
                    median=np.median(rewards), 
                    success=success_rate))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--sleep', type=float, default=-1)
    parser.add_argument('--eval_episodes', type=float, default=5, help='Unit = Episode')
    parser.add_argument('--env', default='AntMaze', type=str)
    # parser.add_argument('--env', default='AntPush', type=str)
    parser.add_argument('--td3', action='store_true', default=False)

    # Training
    parser.add_argument('--num_episode', default=25000, type=int)
    parser.add_argument('--start_training_steps', default=2500, type=int, help='Unit = Global Step')
    parser.add_argument('--writer_freq', default=25, type=int, help='Unit = Global Step')
    # Training (Model Saving)
    parser.add_argument('--subgoal_dim', default=15, type=int)
    parser.add_argument('--load_episode', default=-1, type=int)
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
    parser.add_argument('--batch_size', default=256, type=int)
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
            # dirs_str = listdirs(args.model_path)
            # dirs = np.array(list(map(int, dirs_str)))
            # experiment_name = dirs_str[np.argmax(dirs)]
            experiment_name = "antmaze_256"
        else:
            experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(experiment_name)

    # Environment and its attributes
    env = EnvWithGoal(create_maze_env(args.env), args.env)
    goal_dim = 2
    state_dim = env.state_dim
    action_dim = env.action_dim
    print(state_dim, action_dim)
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
        # Record this experiment with arguments to a CSV file
        record_experience_to_csv(args, experiment_name)
        # Start training
        trainer = Trainer(args, env, agent, experiment_name)
        trainer.train()
    if args.eval:
        agent.load(24000)
        run_evaluation(args, env, agent)