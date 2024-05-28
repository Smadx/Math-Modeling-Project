# Math-Modeling-Project
Project of ustc course Math Modeling 24Spr. ~Renjie Chen

## Task
![](../data/task.png) 

## Environment
使用下面的命令来创建环境: 
```shell
conda env create -f environment.yml
```
文件目录应该为:
```log
--Math-Modeling-Project
  |--data
  |--src
    |--results
```

## Method
我们使用DQN(Deep Q Network)来解决这个问题。DQN是一种基于Q-learning的深度强化学习算法，它通过神经网络来拟合Q函数，从而实现对状态-动作空间的学习。

关于原理可以参考:[Q-leaning](https://zhuanlan.zhihu.com/p/365814943), [DQN](https://zhuanlan.zhihu.com/p/630554489),[Exoerience Replay](https://zhuanlan.zhihu.com/p/145102068)

## Usage
按默认参数启动训练脚本
```shell
bash train.sh
```
按默认参数启动推理脚本
```shell
bash predict.sh
```
