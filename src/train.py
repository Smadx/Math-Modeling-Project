import argparse

import torch
import yaml
import os
import dataclasses
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils

from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from model import DQN

def main():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--pq-dir", type=str, default="../data/pq.csv")
    parser.add_argument("--rs-dir", type=str, default="../data/rs.csv")
    parser.add_argument("--length", type=float, default=240)
    parser.add_argument("--width", type=float, default=240)
    parser.add_argument("--row", type=int, default=6)
    parser.add_argument("--col", type=int, default=6)
    parser.add_argument("--vm", type=float, default=2880)
    parser.add_argument("--va", type=float, default=330)

    # Q-learning
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--d", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--capacity", type=int, default=10_000)

    # Training
    parser.add_argument("--episode", type=int, default=60)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--results-path", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)

    utils.init_logger(accelerator)
    cfg = utils.init_config_from_args(utils.TrainConfig, args)

    # 初始化模型
    exp_model = DQN(cfg.row)
    target_model = DQN(cfg.row)
    target_model.load_state_dict(exp_model.state_dict())
    utils.print_model_summary(exp_model, batch_size=cfg.batch_size,shape=(1, 6, 6))

    # 读取数据
    pq = pd.read_csv(cfg.pq_dir)
    rs = pd.read_csv(cfg.rs_dir)

    pq = pq.values
    rs = rs.values

    # 创建环境
    env = utils.Environment(pq, rs, cfg.length, cfg.width, cfg.row, cfg.col, cfg.vm, cfg.va)

    # 创建策略
    policy = utils.Policy(cfg.epsilon)

    # 创建经验池
    buffer = utils.ReplayBuffer(cfg.capacity, cfg.row, cfg.d, env)

    utils.log(f"Train on {accelerator.device}")
    Trainer(
        exp_model,
        target_model,
        env,
        policy,
        buffer,
        accelerator,
        make_opt=lambda params: torch.optim.Adam(params, lr=cfg.lr),
        replace_target_iter=10,
        config=cfg,
        results_path=utils.handle_results_path(cfg.results_path),
    ).train()

class Trainer:
    """
    训练器

    Args:
        - exp_model: 经验模型
        - target_model: 目标模型
        - env: 环境
        - policy: 策略
        - buffer:经验池
        - accelerator: Accelerate组件
        - make_opt: 优化器
        - replace_target_iter: 目标模型替换频率
        - config: 配置文件
        - results_path: 结果路径
    """
    def __init__(self, 
        exp_model: utils.DQN,
        target_model: utils.DQN,
        env: utils.Environment,
        policy: utils.Policy,
        buffer: utils.ReplayBuffer,
        accelerator:Accelerator,
        make_opt,
        replace_target_iter:int,
        config,
        results_path:Path
    ):
        super().__init__()
        self.model = accelerator.prepare(exp_model)
        self.target = target_model
        self.env = env
        self.policy= policy
        self.buffer = buffer
        self.accelerator = accelerator
        self.opt = accelerator.prepare(make_opt(self.model.parameters()))
        self.replace_target_iter = replace_target_iter
        self.critierion = torch.nn.MSELoss()
        self.cfg = config
        self.train_episodes = self.cfg.episode
        self.step_range = self.cfg.steps
        self.results_path = results_path
        self.device = self.accelerator.device
        print('Train on', self.device)
        self.model = self.model.to(self.device)
        self.target = self.target.to(self.device)
        self.target.eval()
        self.checkpoint_path = self.results_path / f"model.pt"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        outer_tqdm = tqdm(range(self.train_episodes), desc='Episode', position=0)
        for epi in outer_tqdm:
            s = utils.State(self.cfg.row, self.cfg.d)
            episode_losses = []
            for t in tqdm(range(self.step_range), desc='Step', position=1, leave=False):
                act = self.policy.choose_action(s, self.model)
                prev, succ = s, s.apply_action(act)
                self.buffer.add(prev, act, self.env.get_reward(succ), succ)
                S, A, R, SUCC = self.buffer.sample(self.cfg.batch_size) # (s, a, r, s')
                S_tensor = torch.tensor(S, dtype=torch.float32, device=self.device).unsqueeze(1) # [B, 1, H, W]
                A_tensor = torch.tensor(A, dtype=torch.long, device=self.device).unsqueeze(1)  # [B, 1]
                R_tensor = torch.tensor(R, dtype=torch.float32, device=self.device) # [B]
                SUCC_tensor = torch.tensor(SUCC, dtype=torch.float32, device=self.device).unsqueeze(1) # [B, 1, H, W]
                q_values = self.target(SUCC_tensor) # [B, Action_space]
                masked_q_values = utils.apply_action_mask_tensor(q_values, SUCC, self.cfg.row, self.cfg.d) # [B, Action_space]
                max_q_values, _ = torch.max(masked_q_values, dim=1) # [B, Action_space]
                Y = R_tensor + torch.tensor(self.cfg.gamma, dtype=torch.float32, device=self.device) * max_q_values # [B]
                q_eval = self.model(S_tensor) # [B, Action_space]
                q_eval = q_eval.gather(1, A_tensor).squeeze(1) # [B]
                loss = self.critierion(Y, q_eval)
                self.opt.zero_grad()
                self.accelerator.backward(loss)
                self.opt.step()
                episode_losses.append(loss.item())
                if t % self.replace_target_iter == 0:
                    self.target.load_state_dict(self.model.state_dict())
            avg_loss = sum(episode_losses) / len(episode_losses)
            loss_list.append(avg_loss)
            outer_tqdm.set_postfix(loss=avg_loss)
        
        self.save()


    def save(self):
        """
        Save model to checkpoint_path
        """
        self.model.eval()
        checkpoint_path = Path(self.checkpoint_path)
        checkpoint_dir = checkpoint_path.parent

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint = {
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        utils.log(f"Saved model to {checkpoint_path}")

if __name__ == "__main__":
    main()