import numpy as np

prisoners_dilemma = np.array([[5, 0],
                              [10, 1]])
matching_coins = np.array([[1, -1],
                           [-1, 1]])
rock_paper_scissors = np.array([[0, -1, 1],
                                [1, 0, -1],
                                [-1, 1, 0]])
p1_rewards = [prisoners_dilemma, matching_coins, rock_paper_scissors]
p2_rewards = [p1_rewards[0].transpose(), -p1_rewards[1], -p1_rewards[2]]


def normalize(policy):
    if (policy < 0).any():
        raise ValueError('Policy has negative value')
    tot = policy.sum()
    return policy/tot


def play_game(policy1, reward_matrix1, policy2, reward_matrix2):
    assert len(reward_matrix1) == len(reward_matrix2) == len(policy1) == len(policy2), \
        'Size of policy arrays and rewards matrices do not match'
    n = len(reward_matrix1)
    p1_action = np.random.choice(n, p=policy1)
    p2_action = np.random.choice(n, p=policy2)
    p1_reward = reward_matrix1[p1_action, p2_action]
    p2_reward = reward_matrix2[p1_action, p2_action]
    return (p1_action, p1_reward), (p2_action, p2_reward)


def update_policy(policy, action, reward, learning_rate, mode=1):
    if mode == 1:
        policy += -learning_rate * reward * policy
    else:
        raise NotImplementedError('Havent implemented expected value (E)')
        policy += -learning_rate * reward * policy + learning_rate*(E-policy[action])
    policy[action] += learning_rate * reward
    return normalize(policy)


def iterate(policy1, reward_matrix1, policy2, reward_matrix2, alpha=0.005, max_iterations=100000, mode=1):
    p1_history = []
    p2_history = []
    done_iterating = False
    for t in range(max_iterations):
        p1_history.append([*policy1])
        p2_history.append([*policy2])
        (a1, r1), (a2, r2) = play_game(policy1, reward_matrix1, policy2, reward_matrix2)
        policy1 = update_policy(policy1, a1, r1, learning_rate=alpha, mode=mode)
        policy2 = update_policy(policy2, a2, r2, learning_rate=alpha, mode=mode) # Doesn't capture final policy in history?
        if done_iterating:
            break
    return p1_history, p2_history


if __name__ == '__main__':
    matrix_1 = p1_rewards[0]
    matrix_2 = p2_rewards[0]

    policy_1 = normalize(np.random.random(2))
    policy_2 = normalize(np.random.random(2))

    p1_history, p2_history = iterate(policy_1, matrix_1, policy_2, matrix_2)
