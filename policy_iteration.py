import numpy as np

p_d = np.array([[5, 0],
                [10, 1]])
m_c = np.array([[1, -1],
                [-1, 1]])
r_p_s = np.array([[0, -1, 1],
                  [1, 0, -1],
                  [-1, 1, 0]])
p1_rewards = [p_d, m_c, r_p_s]
p2_rewards = [p_d.transpose(), -m_c, -r_p_s]


def normalize(*args):
    t = sum(args)
    return np.array([arg/t for arg in args])


def play_game(reward_matrix1, reward_matrix2, policy1, policy2):
    assert len(reward_matrix1) == len(reward_matrix2) == len(policy1) == len(policy2)
    n = len(reward_matrix1)
    p1_action = np.random.choice(n, p=policy1)
    p2_action = np.random.choice(n, p=policy2)
    p1_reward = reward_matrix1[p1_action, p2_action]
    p2_reward = reward_matrix2[p1_action, p2_action]
    return (p1_action, p1_reward), (p2_action, p2_reward)


def update_policy(p, a, r, ALPHA, mode=1):
    if mode == 1:
        p += -ALPHA * r * p
    else:
        p += -ALPHA * r * p + ALPHA*(E-p[a])    # NOT IMPLEMENTED
    p[a] += ALPHA * r
    return normalize(*p)


def iterate(p1, p2, matrix1, matrix2, ALPHA=0.5, T=50000, alg=1):
    p1_history = []
    p2_history = []
    for t in range(T):
        print('Iteration {}'.format(t+1))
        p1_history.append([*p1])
        p2_history.append([*p2])
        (a1, r1), (a2, r2) = play_game(matrix1, matrix2, p1, p2)
        p1 = update_policy(p1, a1, r1, ALPHA, mode=alg)
        p2 = update_policy(p2, a2, r2, ALPHA, mode=alg)
    return p1_history, p2_history


if __name__ == '__main__':
    p1_r = p1_rewards[0]
    p2_r = p2_rewards[0]

    policy_1 = normalize(*np.random.random(2))
    policy_2 = normalize(*np.random.random(2))

    p1_history, p2_history = iterate(policy_1, policy_2, p1_r, p2_r)

