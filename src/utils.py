import dataclasses
import torch
import torchinfo
import numpy as np
import numpy.typing as npt
import warnings
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque
from dataclasses import dataclass
from torch import nn
from math import exp
from datetime import datetime
from pathlib import Path
from typing import Optional
from accelerate import Accelerator
from matplotlib.colors import LinearSegmentedColormap

from model import DQN

@dataclass
class TrainConfig:
    pq_dir: str
    rs_dir: str
    length: float
    width: float
    row: int
    col: int
    vm: float
    va: float
    gamma: float
    d: float
    epsilon: float
    episode: int
    steps: int
    capacity: int
    batch_size: int
    lr: float
    seed: int
    results_path: str


def print_model_summary(model, *, batch_size, shape, depth=2, batch_size_torchinfo=1):
    """
    Args:
        - model: the model to summarize
        - batch_size: the batch size to use for the summary
        - shape: the shape of the input tensor
        - depth: the depth of the summary
        - batch_size_torchinfo: the batch size to use for torchinfo
    """
    summary = torchinfo.summary(
        model,
        [(batch_size_torchinfo, *shape)],  # Input shape
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # no text output
    )
    log(summary)
    if batch_size is None or batch_size == batch_size_torchinfo:
        return
    output_bytes_large = summary.total_output_bytes / batch_size_torchinfo * batch_size
    total_bytes = summary.total_input + output_bytes_large + summary.total_param_bytes
    log(
        f"\n--- With batch size {batch_size} ---\n"
        f"Forward/backward pass size: {output_bytes_large / 1e9:0.2f} GB\n"
        f"Estimated Total Size: {total_bytes / 1e9:0.2f} GB\n"
        + "=" * len(str(summary).splitlines()[-1])
        + "\n"
    )

def apply_action_mask(q_values:npt.NDArray, valid_actions:list, grid_size:int)->npt.NDArray:
    """
    应用动作掩码

    Args:
        - q_values:模型输出的Q值
        - valid_actions:合法的动作
        - grid_size:网格大小
    """
    mask = np.full(q_values.shape, -np.inf)
    for action in valid_actions:
        i, j, k = action
        index = 2 * (i * grid_size + j) + k
        mask[index] = 0
    return q_values + mask

def apply_action_mask_tensor(q_values: torch.Tensor, states:npt.NDArray, grid_size:int, d:float):
    """
    应用动作掩码

    Args:
        - q_values:模型输出的Q值
        - valid_actions:合法的动作
        - grid_size:网格大小
    """
    mask = torch.full_like(q_values, float('-inf'))
    for idx, state_array in enumerate(states):
        state = State(grid_size, d)
        state.prob_matrix = state_array
        valid_actions = state.get_valid_actions()
        for action in valid_actions:
            i, j, k = action
            index = 2 * (i * grid_size + j) + k
            mask[idx, index] = 0
    return q_values + mask

class Action:
    """
    动作, 动作是一个三元组,前两位表示执行位置,第三位表示增加还是减少

    Args:
        - i:矩阵的行
        - j:矩阵的列
        - k:行为,表示当前位置概率增加还是减少, k = 1 表示 +d, k = 0 表示 -d
    """
    def __init__(self, i:int, j:int, k:int):
        self.i = i
        self.j = j
        self.k = k

    @classmethod
    def from_tuple(cls, t:tuple):
        act = cls(t[0], t[1], t[2])
        return act

    def __repr__(self):
        return f"Action(i={self.i}, j={self.j}, k={self.k})"

class State:
    """
    状态, 这是一个与网格大小相同的矩阵,每个元素表示相对应的网格是空洞的概率

    Args:
        - grid_size:网格大小
        - d:概率增加或减少的幅度

    Method:
        - get_prob(i, j):获取位置 (i, j) 的概率值
        - set_prob(i, j, value):设置位置 (i, j) 的概率值，并确保值在 [0, 1] 范围内
        - update_prob(i, j, delta):更新位置 (i, j) 的概率值
        - apply_action(action):根据动作更新状态
        - get_matrix:返回当前概率矩阵
    """
    def __init__(self, grid_size: int, d: float=0.1):
        self.grid_size = grid_size
        self.d = d
        self.prob_matrix = np.random.rand(grid_size, grid_size) # 初始化所有位置的概率

    @classmethod
    def from_matrix(cls, matrix:npt.NDArray, d:float):
        """从给定的矩阵创建状态"""
        grid_size = matrix.shape[0]
        state = cls(grid_size, d)
        state.prob_matrix = matrix
        return state

    def get_prob(self, i:int, j:int):
        """获取位置 (i, j) 的概率值"""
        return self.prob_matrix[i, j]

    def set_prob(self, i:int, j:int, value:float):
        """设置位置 (i, j) 的概率值，并确保值在 [0, 1] 范围内"""
        self.prob_matrix[i, j] = np.clip(value, 0, 1)

    def update_prob(self, i:int, j:int, delta:float):
        """更新位置 (i, j) 的概率值"""
        self.set_prob(i, j, self.get_prob(i, j) + delta)

    def apply_action(self, action:Action):
        """根据动作更新状态"""
        delta = self.d if action.k == 1 else -self.d
        self.update_prob(action.i, action.j, delta)
        return self

    def get_matrix(self):
        """获取整个概率矩阵"""
        return self.prob_matrix
    
    def get_valid_actions(self)->list[tuple[int, int, int]]:
        """返回所有合法的动作"""
        valid_actions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.get_prob(i, j) < 1.0:
                    valid_actions.append((i, j, 1))
                if self.get_prob(i, j) > 0.0:
                    valid_actions.append((i, j, 0))
        return valid_actions
    
    def show(self):
        colors = [(0.36, 0.25, 0.20), (1, 1, 1)]  # 从棕色到白色
        n_bins = 100  
        cmap_name = 'brown_white'

        # 创建颜色映射
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        plt.imshow(self.prob_matrix, cmap=cm, interpolation='nearest')
        plt.colorbar()

        plt.title('Probability Matrix')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        plt.savefig('probability_matrix.png')

    def __repr__(self):
        return str(self.prob_matrix) 
    
class Environment:
    """
    环境,与状态和动作进行交互

    Args:
        - pq:P-Q观测数据矩阵
        - rs:R-S数据观测矩阵
        - length:长度
        - width:宽度
        - row:网格行数
        - col:网格列数
        - vm:介质波速
        - va:空洞波速
        
    Method:
        - get_reward(state):返回给定状态对应的奖励
        - predict(state):根据当前状态预测理论上的PQ,RS观测值
    """
    def __init__(self, pq:npt.NDArray, rs:npt.NDArray, length:float, width:float, row:int, col:int, vm:float, va:float):
        self.pq = pq
        self.rs = rs
        self.length = length
        self.width = width
        self.row = row
        self.col = col
        self.vm = vm
        self.va = va

    def predict(self, state:State)->tuple[npt.NDArray, npt.NDArray]:
        PQ = self.get_t1(state.prob_matrix)
        RS = self.get_t2(state.prob_matrix)
        return PQ, RS

    def get_reward(self, state:State)->float:
        PQ, RS = self.predict(state)
        return -(np.mean((self.pq - PQ) ** 2) + np.mean((self.rs - RS) ** 2)) 
    
    # 计算A_{ijmn,1}
    def get_A1(self, i, j, m, n):
        """
        计算探测线P_i-Q_j在网格(m,n)内的长度
        """
        if i == j:
            if m == i or m == i-1:
                return self.width / (2 * self.col)
            else:
                return 0
        elif i < j:
            t_start = max(self.length / self.row * (m - 1), self.length / self.col * (n - 1) / (self.row / (j - i)) + self.length / self.row * (i - 1))
            t_end = min(self.length / self.row * m, self.length / self.col * n / (self.row / (j - i)) + self.length / self.row * (i - 1))
            L = (t_end - t_start) * np.sqrt(1 + (self.width * self.row / (self.length * (j - i)))**2)
            return L * (1 if t_end > t_start else 0)
        elif i > j:
            t_start = max(self.length / self.row * (m - 1), self.length / self.col * n / (self.row / (j - i)) + self.length / self.row * (i - 1))
            t_end = min(self.length / self.row * m, self.length / self.col * (n - 1) / (self.row / (j - i)) + self.length / self.row * (i - 1))
            L = (t_end - t_start) * np.sqrt(1 + (self.width * self.row / (self.length * (j - i)))**2)
            return L * (1 if t_end > t_start else 0)

    # 计算A_{ijmn,2}
    def get_A2(self, i, j, m, n):
        """
        计算探测线R_i-S_j在网格(m,n)内的长度
        """
        if i == j:
            if n == i or n == i-1:
                return self.width / (2 * self.col)
            else:
                return 0
        
        elif i < j:
            t_start = max(self.length / self.row * (m - 1), self.length * (n - i) / (j - i))
            t_end = min(self.length / self.row * m, self.length * (n - i + 1) / (j - i))
            L = (t_end - t_start) * np.sqrt(1 + (self.width * (j - i) / (self.length * self.col))**2)
            return L * (1 if t_end > t_start else 0)
        elif i > j:
            t_start = max(self.length / self.row * (m - 1), self.length * (n - i + 1) / (j - i))
            t_end = min(self.length / self.row * m, self.length * (n - i) / (j - i))
            L = (t_end - t_start) * np.sqrt(1 + (self.length * (j - i) / (self.length * self.col))**2)
            return L * (1 if t_end > t_start else 0)
        
    # 计算t_{ij,1}
    def get_t1(self, X:npt.NDArray):
        """
        计算理论上PQ的观测数据

        Args:
            - X:当前概率矩阵
        """
        t1 = np.zeros((self.row+1,self.col+1))
        for i in range(1, self.row+2):
            for j in range(1, self.col+2):
                if (i, j) == (1, 1):
                    A_ijmn1 = self.width / (2 * self.col)
                    t1[i-1, j-1] = sum((A_ijmn1 * X[i-1, n-1] / self.va + (self.width / self.col - A_ijmn1 * X[i-1, n-1]) / self.vm) for n in range(self.row))
                elif (i, j) == (self.row+1, self.col+1):
                    A_ijmn1 = self.width / (2 * self.col)
                    t1[i-1, j-1] = sum((A_ijmn1 * X[5, n-1] / self.va + (self.width / self.col - A_ijmn1 * X[5, n-1]) / self.vm) for n in range(self.col))                  
                else:
                    temp_sum = 0
                    for m in range(1, self.row):
                        for n in range(1, self.col):
                            A_ijmn = self.get_A1(i, j, m, n)
                            temp_sum += A_ijmn * (1 - X[m-1, n-1]) / self.vm + A_ijmn * X[m-1, n-1] / self.va
                    t1[i-1, j-1] = temp_sum
        return t1

    # 计算t_{ij,2}
    def get_t2(self, X:npt.NDArray):
        """
        计算理论上RS的观测数据

        Args:
            - X:当前概率矩阵
        """
        t2 = np.zeros((self.row+1, self.col+1))
        for i in range(1, self.row+2):
            for j in range(1, self.col+2):
                if (i, j) == (1, 1):
                    A_ijmn2 = self.length / (2 * self.row)
                    t2[i-1, j-1] = sum((A_ijmn2 * X[m-1, 0] / self.va + (self.length / self.row - A_ijmn2 * X[m-1, 0]) / self.vm) for m in range(1,self.row+1))
                elif (i, j) == (self.row+1, self.col+1):
                    A_ijmn2 = self.length / (2 * self.row)
                    t2[i-1, j-1] = sum((A_ijmn2 * X[m-1, 5] / self.va + (self.length / self.row - A_ijmn2 * X[m-1, 5]) / self.vm) for m in range(1,self.col+1))
                else:
                    temp_sum = 0
                    for m in range(1, 7):
                        for n in range(1, 7):
                            A_ijmn = self.get_A2(i, j, m, n)
                            temp_sum += A_ijmn * (1 - X[m-1, n-1]) / self.vm + A_ijmn * X[m-1, n-1] / self.va
                    t2[i-1, j-1] = temp_sum
        return t2
    
class ReplayBuffer:
    """
    经验池,用于储存与环境交互得到的经验

    Args:
        - capacity:经验池容量

    Method:
        - add(state, action, reward, next_state):把经验(s,a,r,s')压入经验池
        - sample(batch_size):从经验池中采样一个batch的数据
        - size:返回当前经验池中的经验数量
    """
    def __init__(self, capacity:int, grid_size:int, d:float, env:Environment):
        self.buffer = deque(maxlen=capacity)
        self.grid_size = grid_size
        print("Creating ReplayBuffer")
        for i in tqdm(range(capacity)):
            s = State(grid_size, d)
            act = Action.from_tuple(random.choice(s.get_valid_actions()))
            prev, succ = s, s.apply_action(act)
            self.add(prev, act, env.get_reward(succ), succ)
    
    def add(self, state:State, action:Action, reward:float, next_state:State):
        """将经验添加到缓冲区中"""
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size:int)->tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """从缓冲区中随机采样一个小批量"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array([state.get_matrix() for state in states],dtype=np.float32)
        next_states = np.array([state.get_matrix() for state in next_states],dtype=np.float32)
        actions = np.array([2 * (action.i * self.grid_size + action.j) + action.k for action in actions],dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        #print(f"states.shape={states.shape}")
        #print(f"actions.shape={actions.shape}")
        #print(f"rewards.shape={rewards.shape}")
        #print(f"next_states.shape={next_states.shape}")
        return states, actions, rewards, next_states
    
    def size(self)->int:
        """返回当前缓冲区中存储的经验数量"""
        return len(self.buffer)

class Policy:
    """
    1-\epsilon策略,以1-\epsilon的概率执行贪心策略

    Args:
        - epsilon:执行随机策略的概率
        - model:DQN模型

    Method:
        - choose_action(state):对于给定状态,返回一个合法的动作
    """
    def __init__(self, epsilon:float):
        if (epsilon <= 0.1 and epsilon > 0) or epsilon >= 1 or epsilon < 0:
            raise ValueError("Invalid epsilon")
        
        self.epsilon = epsilon
        self.max_eps = epsilon

    def choose_action(self, state:State, model:DQN)->Action:
        state_vector = state.get_matrix()
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cuda')
        q_values = model(state_tensor).squeeze(0)
        q_values = q_values.to('cpu')
        q_values = q_values.detach().numpy()
        q_values = apply_action_mask(q_values, state.get_valid_actions(), state.grid_size)
        
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(np.where(q_values != -np.inf)[0])
        else:
            action_index = np.argmax(q_values)
        
        i = action_index // 2 // state.grid_size
        j = (action_index // 2) % state.grid_size
        k = action_index % 2
        return Action(i, j, k)
    
    def decrease(self, step):
        self.epsilon = 0.1 + (self.max_eps - 0.1) * exp(- 0.1 * step)



def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def handle_results_path(res_path: str, default_root: str = "./results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path)
    log(f"Results will be saved to '{results_path}'")
    return results_path

@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module
    

def init_config_from_args(cls, args):
    """
    Initialize a dataclass from a Namespace.
    """
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})

_accelerator: Optional[Accelerator] = None


def init_logger(accelerator: Accelerator):
    global _accelerator
    if _accelerator is not None:
        raise ValueError("Accelerator already set")
    _accelerator = accelerator


def log(message):
    global _accelerator
    if _accelerator is None:
        warnings.warn("Accelerator not set, using print instead.")
        print_fn = print
    else:
        print_fn = _accelerator.print
    print_fn(message)
