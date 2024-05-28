import argparse

import torch
import yaml
import utils
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path

from tqdm import tqdm

from model import DQN

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inference-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--results-path", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = utils.TrainConfig(**yaml.safe_load(f))

    accelerator = Accelerator(split_batches=True)

    # 装载模型
    checkpoint_path = Path(args.results_path) / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
    model = DQN(cfg.row)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 创建策略
    policy = utils.Policy(0)

    model.to(accelerator.device)
    utils.log(f"Predict on {accelerator.device}")

    M = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0.8, 0, 0, 0],
        [0, 0.1, 0.8, 0.8, 0.1, 0],
        [0, 0.8, 0.8, 0.1, 0, 0],
        [0, 0.8, 0.8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=np.float32)
    s = utils.State.from_matrix(M, cfg.d)
    for i in tqdm(range(args.inference_steps)):
        a = policy.choose_action(s, model)
        s = s.apply_action(a)

    s.show()


if __name__ == "__main__":
    main()