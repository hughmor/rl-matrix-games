import numpy as np
import matplotlib.pyplot as plt

prisoners_dilemma = np.array([[5, 0],
                              [10, 1]])
matching_coins = np.array([[1, -1],
                           [-1, 1]])
rock_paper_scissors = np.array([[0, -1, 1],
                                [1, 0, -1],
                                [-1, 1, 0]])
p1_rewards = [prisoners_dilemma, matching_coins, rock_paper_scissors]
p2_rewards = [p1_rewards[0].transpose(), -p1_rewards[1], -p1_rewards[2]]
action_labels = [['lie', 'confess'], ['head', 'tail'], ['rock', 'paper', 'scissors']]


def normalize(policy):
    if (policy < 0).any():
        raise ValueError('Policy has negative value')
    tot = policy.sum()
    return policy/tot


def play_game(policies, rewards):
    n = len(rewards[0])
    p1_action = np.random.choice(n, p=policies[0])
    p2_action = np.random.choice(n, p=policies[1])
    p1_reward = rewards[0][p1_action, p2_action]
    p2_reward = rewards[1][p1_action, p2_action]

    return (p1_action, p2_action), (p1_reward, p2_reward)


def update_policy(policy, action, reward, learning_rate, update_alg='standard'):
    if update_alg == 'standard':
        for i in range(len(policy)):
            if i is not action:
                policy[i] += -learning_rate * reward * policy[i]
            else:
                policy[i] += learning_rate * reward * (1-policy[i])
    elif update_alg == 'modified':
        raise NotImplementedError('Havent implemented expected value (E)')
        policy += -learning_rate * reward * policy + learning_rate*(E-policy[action])
    else:
        raise NotImplementedError('Algorithm not implemented.')        
    return normalize(policy)


def iterate(policy1, reward_matrix1, policy2, reward_matrix2, alpha=0.005, max_iterations=10000, update_alg='standard'):
    p_history = [[],[]]
    policies = [policy1, policy2]
    rewards = [reward_matrix1, reward_matrix2]
    Q = [[0 for _ in range(len(policy1))]] * 2
    N = [[0 for _ in range(len(policy1))]] * 2     
    for _ in range(max_iterations):
        action, reward = play_game(policies, rewards)
        for i in range(2):
            p_history[i].append(policies[i])
            policies[i] = update_policy(policies[i], action[i], reward[i], learning_rate=alpha, update_alg=update_alg)
            N[i][action[i]] += 1
            Q[i][action[i]] += 1/N[i][action[i]]*(reward[i]-Q[i][action[i]])
    return p_history[0], p_history[1], Q[0], Q[1]


def make_plots(player1_history, player2_history, action_labels):
    # Plots of convergance
    episodes = list(range(len(player1_history)))
    players_list = [zip(*player1_history), zip(*player2_history)]
    figure = plt.figure()
    subplots = []
    subplots.append(figure.add_subplot(121))
    subplots.append(figure.add_subplot(122, sharey=subplots[0]))
    for subplot, player, i in zip(subplots, players_list, range(2)):
        for policy, action in zip(player, action_labels):
            subplot.plot(episodes, policy, label=action)
        subplot.set_xlabel('Episode')
        subplot.set_title('Player ' + str(i+1))
        subplot.legend()
    figure.text(0.06, 0.5, 'Action Probability', ha='center', va='center', rotation='vertical')
    plt.show()

rewards_1 = p1_rewards[0]
rewards_2 = p2_rewards[0]
labels = action_labels[0]
N = 10000

## start with random policies
policy_1 = normalize(np.random.random(2))
policy_2 = normalize(np.random.random(2))

p1_history, p2_history, Q1, Q2 = iterate(policy_1, rewards_1, policy_2, rewards_2, max_iterations=N)
p1_lie, p1_confess = zip(*p1_history)
p2_lie, p2_confess = zip(*p2_history)
print('Converged to probabilities of lying:\n\tPlayer 1:{}\n\tPlayer 2:{}\n'.format(p1_lie[-1], p2_lie[-1]))
print('Converged to probabilities of confessing:\n\tPlayer 1:{}\n\tPlayer 2:{}\n'.format(p1_confess[-1], p2_confess[-1]))
make_plots(p1_history, p2_history, labels)