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
