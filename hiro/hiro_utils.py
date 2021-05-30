import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((buffer_size, state_dim))
        self.goal = np.zeros((buffer_size, goal_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.n_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))
        self.cost = np.zeros((buffer_size, 1))

        self.device = device

    def append(self, state, goal, action, n_state, reward, done, cost):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.cost[self.ptr] = cost

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
        )

class LowReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        super(LowReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.n_goal = np.zeros((buffer_size, goal_dim))

    def append(self, state, goal, action, n_state, n_goal, reward, done):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.n_goal[self.ptr] = n_goal
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.n_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    def save(self, path="buffer_dict"):
        '''
        import numpy as np

        # Save
        dict = {'a':1,'b':2,'c':3}
        np.save('my_file.npy', dict) # 注意带上后缀名

        # Load
        load_dict = np.load('my_file.npy').item()
        print(load_dict['a'])
        '''
        self.LowReplayBuffer_dict = {"state": self.state, "goal":self.goal, "action":self.action, "n_state":self.n_state, "n_goal":self.n_goal,"reward":self.reward,"not_done":self.not_done}
        np.save("/LowReplayBuffer_dict.npy", self.LowReplayBuffer_dict)

    def load(self, path="buffer_dict"):
        self.LowReplayBuffer_dict = np.load(path+"/LowReplayBuffer_dict.npy").item()
        self.state = self.LowReplayBuffer_dict["state"]
        self.goal = self.LowReplayBuffer_dict["goal"]
        self.action = self.LowReplayBuffer_dict["action"]
        self.n_state = self.LowReplayBuffer_dict["n_state"]
        self.n_goal = self.LowReplayBuffer_dict["n_goal"]
        self.reward = self.LowReplayBuffer_dict["reward"]
        self.not_done = self.LowReplayBuffer_dict["not_done"]


class HighReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, subgoal_dim, action_dim, buffer_size, batch_size, freq):
        super(HighReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.action = np.zeros((buffer_size, subgoal_dim))
        self.state_arr = np.zeros((buffer_size, freq, state_dim))
        self.action_arr = np.zeros((buffer_size, freq, action_dim))

    def append(self, state, goal, action, n_state, reward, done, state_arr, action_arr, cost):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.state_arr[self.ptr,:,:] = state_arr
        self.action_arr[self.ptr,:,:] = action_arr
        self.cost[self.ptr] = cost

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.state_arr[ind]).to(self.device),
            torch.FloatTensor(self.action_arr[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
        )

    def save(self, path="buffer_dict"):
        '''
        import numpy as np

        # Save
        dict = {'a':1,'b':2,'c':3}
        np.save('my_file.npy', dict) # 注意带上后缀名

        # Load
        load_dict = np.load('my_file.npy').item()
        print(load_dict['a'])
        '''
        self.HighReplayBuffer_dict = {"state": self.state, "goal": self.goal, "action": self.action, "n_state": self.n_state,
                                      "reward": self.reward, "not_done": self.not_done}
        np.save("/HighReplayBuffer_dict.npy", self.HighReplayBuffer_dict)

    def load(self, path="buffer_dict"):
        self.HighReplayBuffer_dict = np.load(path + "/HighReplayBuffer_dict.npy").item()
        self.state = self.HighReplayBuffer_dict["state"]
        self.goal = self.HighReplayBuffer_dict["goal"]
        self.action = self.HighReplayBuffer_dict["action"]
        self.n_state = self.HighReplayBuffer_dict["n_state"]
        self.reward = self.HighReplayBuffer_dict["reward"]
        self.not_done = self.HighReplayBuffer_dict["not_done"]

class SubgoalActionSpace(object):
    def __init__(self, dim):
        limits = np.array([-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3])
        self.shape = (dim,1)
        self.low = limits[:dim]
        self.high = -self.low

    def sample(self):
        return (self.high - self.low) * np.random.sample(self.high.shape) + self.low

class Subgoal(object):
    def __init__(self, dim=15):
        self.action_space = SubgoalActionSpace(dim)
        self.action_dim = self.action_space.shape[0]
