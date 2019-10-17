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


T = 50000
ALPHA = 0.5
games = {'prisoners': 0,
         'coins': 1,
         'rps': 2
         }

game = games['prisoners']  # Choose Game Here
alg = 1  # Choose Algorithm Here (1 or 2)


def main():
    p1_history = []
    p2_history = []

    policy_1 = normalize(*np.random.random(len(p1_rewards[game])))
    policy_2 = normalize(*np.random.random(len(p2_rewards[game])))

    for t in range(T):
        p1_history.append([*policy_1])
        p2_history.append([*policy_2])
        (a1, r1), (a2, r2) = play_game(p1_rewards[game], p2_rewards[game], policy_1, policy_2)
        policy_1 = update_policy(policy_1, a1, r1, mode=alg)
        policy_2 = update_policy(policy_2, a2, r2, mode=alg)


if __name__ == '__main__':
    main()  # do policy iteration
