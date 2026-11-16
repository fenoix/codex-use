# 强化学习学习交接

## 给接续的 Codex

请用中文，从下方“下一步”继续教学，不必重讲全部背景。先举例、跑实验，再解释公式；一次讨论一个问题。学完并确认理解后再打钩，保存实验结果。用户已授权直接提交改动；此前使用 Git 的 work 分支并推送，不直接合并 main。

本文可独立使用，第一课完整代码已附在下面，不依赖其他文档。

## 学习目标与路线

用户对生成式推荐、语义 ID 和大模型强化学习感兴趣，但决定先打基础。已有 H200 服务器，初期小实验使用 CPU 即可。

- [ ] 1. 基础：状态、动作、奖励、策略、回合、累计回报、MDP。
- [ ] 2. 多臂老虎机：探索与利用、ε-greedy。
- [ ] 3. 表格型 Q-learning：Q 值、贝尔曼思想、更新公式。
- [ ] 4. DQN：神经网络估计 Q 值、经验回放、目标网络。
- [ ] 5. 策略梯度、Actor-Critic、PPO。
- [ ] 6. 大模型：PPO 式 RLHF、GRPO、DPO，联系生成式推荐。

目前只开始第一课，尚未确认概念理解；不要把整个第一阶段打钩。

## 之前聊过的背景（只做过介绍）

- DQN 学动作价值，经典形式适合离散动作，属于 off-policy。
- PPO 直接优化策略，通常配合价值模型，以裁剪等方式限制更新，通常按 on-policy 方式训练。
- GRPO 对同一提示采样多个回答，利用组内相对评分，通常不需要价值模型。
- DPO 用优选/劣选回答对优化偏好，标准形式通常离线训练。
- on-policy 使用当前或近期策略的数据；off-policy 可利用旧策略或其他策略的数据。“在线/离线”与它们不是同一组概念。
- 生成式推荐可用 RQ-KMeans 或 RQ-VAE 把物品向量变成多层码本编号，即语义 ID。物品编码、监督推荐训练和强化学习优化是不同环节，生成式推荐不一定需要强化学习。

## 第一课：随机机器人走网格

```text
S . . .
. # . .
. # . .
. . . G
```

坐标为 `(行, 列)`，从 0 开始。起点 `(0,0)`，终点 `(3,3)`，墙壁 `(1,1)`、`(2,1)`。

| 概念 | 本例对应 |
| --- | --- |
| 智能体 | 机器人 |
| 环境 | 地图、移动和奖励规则 |
| 状态 | 当前位置 |
| 动作 | 上、下、左、右 |
| 奖励 | 进入终点 +10，其他每步 -1 |
| 策略 | 每一步以各 25% 的概率选择方向 |

撞墙或越界留在原地，仍扣 1 分。到终点是终止；50 步没到终点是实验截断。一次交互是“状态 → 动作 → 新状态和奖励”，一个回合包含多次交互。

长期回报：6 步到终点，总奖励为 `5×(-1)+10=5`；10 步到达则为 `9×(-1)+10=1`。不能只看眼前一步的奖励。一般折扣回报为 `r₁+γr₂+γ²r₃+…`，本实验统计未折扣总奖励。

### 已完成的实验

- [x] 定义地图与规则。
- [x] 运行并展示随机策略。
- [ ] 确认用户理解概念与回报。

实际运行：seed=42，1000 回合，每回合最多 50 步，成功 542 次，平均步数 38.63，平均总奖励 -32.67。seed=7 的示例轨迹在 50 步处截断，总奖励 -50。

这里没有训练更新，因此随机走更多回合不会自动学会路线。

### 下一步：从这个问题继续

上一轮问了但用户尚未回答：

> 机器人在 `(0,1)`，如果向下走，会到哪里，奖励是多少？

教师参考：下面是墙，留在 `(0,1)`，奖励 -1。先等用户回答并解释，再问：

1. 6 步和 10 步到终点，哪个总奖励更高，为什么？
2. 随机走 1000 回合后会自动记住路线吗？缺少什么环节？

确认后再更新进度，继续探索与利用，之后实现 Q-learning。

## 第一课完整代码

把下面代码保存为 `01_grid_world.py`，执行 `python3 01_grid_world.py`。仅依赖 Python 标准库。

```python
"""第一课：固定随机种子的随机策略；仅使用 Python 标准库。"""

import random

SIZE = 4
START = (0, 0)
GOAL = (3, 3)
WALLS = {(1, 1), (2, 1)}
ACTIONS = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}


def step(state, action):
    """环境规则：撞墙/越界留在原地；到终点 +10，否则 -1。"""
    if state == GOAL:
        raise ValueError("回合已经终止，请重新从起点开始")
    dr, dc = ACTIONS[action]
    candidate = (state[0] + dr, state[1] + dc)
    valid = all(0 <= coordinate < SIZE for coordinate in candidate)
    next_state = candidate if valid and candidate not in WALLS else state
    terminated = next_state == GOAL
    return next_state, 10 if terminated else -1, terminated


def episode(rng, max_steps=50, trace=False):
    state = START
    total_reward = 0
    for count in range(1, max_steps + 1):
        # 策略：每一步等概率选择四个动作，不根据经验更新。
        action = rng.choice(tuple(ACTIONS))
        next_state, reward, terminated = step(state, action)
        total_reward += reward
        if trace:
            print(f"{count:2d}: {state} --{action}--> {next_state}, 奖励={reward:+d}")
        state = next_state
        if terminated:
            return count, total_reward, True
    # 达到步数上限是截断，不是环境中的终点。
    return max_steps, total_reward, False


def main():
    print("S . . .\n. # . .\n. # . .\n. . . G\n")
    print("示例轨迹（seed=7）：")
    steps, reward, success = episode(random.Random(7), trace=True)
    print(f"步数={steps}, 总奖励={reward}, 结束原因={'到达终点' if success else '步数上限截断'}")
    rng = random.Random(42)
    results = [episode(rng) for _ in range(1000)]
    print("\n1000 回合随机策略（seed=42，每回合最多 50 步）：")
    print(f"成功回合={sum(result[2] for result in results)}/1000")
    print(f"平均步数={sum(result[0] for result in results) / len(results):.2f}")
    print(f"平均总奖励={sum(result[1] for result in results) / len(results):.2f}")
    print("注意：这里没有学习，运行更多回合不会自动改善策略。")


if __name__ == "__main__":
    main()

```
