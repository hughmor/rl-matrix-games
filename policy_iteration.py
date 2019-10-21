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
    return np.ma.array([arg/t for arg in args])


def play_game(reward_matrix1, reward_matrix2, policy1, policy2):
    assert len(reward_matrix1) == len(reward_matrix2) == len(policy1) == len(policy2)
    n = len(reward_matrix1)
    p1_action = np.random.choice(n, p=policy1)
    p2_action = np.random.choice(n, p=policy2)
    p1_reward = reward_matrix1[p1_action, p2_action]
    p2_reward = reward_matrix2[p1_action, p2_action]
    return (p1_action, p1_reward), (p2_action, p2_reward)


def update_policy(p, a, r, mode=1):
    p.mask = [False for _ in p]
    if mode == 1:
        p[a] += ALPHA * r * (1-p[a])
        p.mask[a] = True
        p += -ALPHA * r * p
        p.mask = False
    else:
        p[a] += ALPHA * r * (1-p[a]) + ALPHA*(E-p[a])    # NOT IMPLEMENTED
        p.mask[a] = True
        p += -ALPHA * r * p + ALPHA*(E-p[a])    # NOT IMPLEMENTED
        p.mask = False
    return normalize(p)