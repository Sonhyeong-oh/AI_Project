# 인공지능 과제 1 Grid Crossing
# Policy Iteration

import kymnasium as kym
import gymnasium as gym
import numpy as np
import pickle

N = 26 # 맵의 한 변 길이
AGENT_RIGHT, AGENT_DOWN, AGENT_LEFT, AGENT_UP = 1000, 1001, 1002, 1003 # 에이전트 움직임
GOAL, LAVA, WALL = 810, 900, 250 # 타일 속성
DIR = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)} # 움직임 좌표
ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD = 0, 1, 2 # 회전
GAMMA, EVAL_THRESHOLD = 0.99, 1e-5 # 할인율 & 정책 개선 평가 기준

# 보상 & 패널티
TURN_COST, MOVE_COST, WALL_PENALTY = -0.1, -1.0, -5.0
GOAL_REWARD, LAVA_REWARD = 100.0, -100.0

# (행, 열, 방향)을 단일 상태 인덱스로 변환
def encode_state(row, col, ori):
    return row * N * 4 + col * 4 + ori

# 상태 인덱스를 (행, 열, 방향)으로 변환
def decode_state(state):
    return state // (N * 4), (state % (N * 4)) // 4, state % 4

# 관측값에서 플레이어 위치와 방향 추출
def parse_observation(obs):

    idx = np.where((obs == AGENT_RIGHT) | (obs == AGENT_DOWN) | (obs == AGENT_LEFT) | (obs == AGENT_UP))
    if len(idx[0]) > 0:
        pr, pc = int(idx[0][0]), int(idx[1][0])
        ori = {AGENT_UP: 0, AGENT_RIGHT: 1, AGENT_DOWN: 2, AGENT_LEFT: 3}[int(obs[pr, pc])]
    else:
        pr, pc, ori = 0, 0, 1
    return pr, pc, ori

class Agent(kym.Agent):
    def __init__(self, PI):
        self.PI = PI.astype(np.int8)

    def save(self, path):
        with open(path, mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, mode="rb") as f:
            return pickle.load(f)

    def act(self, observation, info):
        pr, pc, ori = parse_observation(observation)
        return int(self.PI[encode_state(pr, pc, ori)])

if __name__ == "__main__":
    import os

    # pkl 파일이 이미 있으면 바로 테스트
    if os.path.exists("agent.pkl"):
        print("Load from agent.pkl...")
        agent = Agent.load("agent.pkl")

    # 없으면 학습 후 테스트
    else:
        print("Training new agent")
        # 환경 초기화
        env = gym.make(id="kymnasium/GridAdventure-Crossing-26x26-v0", render_mode="rgb_array", bgm=False)
        obs, _ = env.reset(seed=42)
        env_map = obs.copy()
        env.close()

        # 전이 모델 생성
        state_count = N * N * 4
        next_states = np.zeros((state_count, 3), dtype=np.int32)
        rewards = np.zeros((state_count, 3), dtype=np.float32)
        dones = np.zeros((state_count, 3), dtype=np.bool_)
        terminal_mask = np.zeros(state_count, dtype=np.bool_)

        for state in range(state_count):
            row, col, ori = decode_state(state)
            tile = env_map[row, col]
            if tile in [WALL, GOAL, LAVA]:
                terminal_mask[state] = True
                next_states[state, :] = state
                dones[state, :] = True
                continue

            for action in range(3):
                if action == ACTION_LEFT:
                    next_row, next_col, next_ori = row, col, (ori - 1) % 4
                    reward, done = TURN_COST, False
                elif action == ACTION_RIGHT:
                    next_row, next_col, next_ori = row, col, (ori + 1) % 4
                    reward, done = TURN_COST, False
                else:
                    dr, dc = DIR[ori]
                    next_row, next_col, next_ori = row + dr, col + dc, ori
                    if not (0 <= next_row < N and 0 <= next_col < N):
                        next_row, next_col, reward, done = row, col, WALL_PENALTY, False
                    else:
                        next_tile = env_map[next_row, next_col]
                        if next_tile == WALL:
                            next_row, next_col, reward, done = row, col, WALL_PENALTY, False
                        elif next_tile == GOAL:
                            reward, done = GOAL_REWARD, True
                        elif next_tile == LAVA:
                            reward, done = LAVA_REWARD, True
                        else:
                            reward, done = MOVE_COST, False

                next_states[state, action] = encode_state(next_row, next_col, next_ori)
                rewards[state, action] = reward
                dones[state, action] = done

        # Policy Iteration
        policy = np.full(state_count, ACTION_FORWARD, dtype=np.int8)
        values = np.zeros(state_count, dtype=np.float32)

        for iteration in range(1, 101):
            # 정책 평가
            for _ in range(1000):
                delta, new_values = 0.0, values.copy()
                for state in range(state_count):
                    if terminal_mask[state]:
                        new_values[state] = 0.0
                        continue
                    action = int(policy[state])
                    value = rewards[state, action] + (0 if dones[state, action] else GAMMA * values[next_states[state, action]])
                    delta = max(delta, abs(value - values[state]))
                    new_values[state] = value
                values = new_values
                if delta < EVAL_THRESHOLD:
                    break

            # 정책 개선
            policy_stable = True
            for state in range(state_count):
                if terminal_mask[state]:
                    continue
                old_action = int(policy[state])
                best_action, best_value = old_action, -np.inf
                for action in range(3):
                    value = rewards[state, action] + (0 if dones[state, action] else GAMMA * values[next_states[state, action]])
                    if value > best_value:
                        best_value, best_action = value, action
                policy[state] = best_action
                if best_action != old_action:
                    policy_stable = False

            if policy_stable:
                break

        # 에이전트 저장
        agent = Agent(PI=policy)
        agent.save("agent.pkl")
        print("Agent saved to agent.pkl")

    # 에이전트 테스트
    eval_env = gym.make(id="kymnasium/GridAdventure-Crossing-26x26-v0", 
                        render_mode="human", 
                        bgm=True)
    obs, info = eval_env.reset(seed=42)
    done, step = False, 0

    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        step += 1
        done = terminated or truncated

    print(f"Finish. {step} steps")
    eval_env.close()

