import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


prisoners_dilemma = np.array([[5, 0],
                              [10, 1]])
matching_coins = np.array([[1, -1],
                           [-1, 1]])
rock_paper_scissors = np.array([[0, -1, 1],
                                [1, 0, -1],
                                [-1, 1, 0]])
p1_rewards = [prisoners_dilemma, matching_coins, rock_paper_scissors]
p2_rewards = [p1_rewards[0].transpose(), -p1_rewards[1], -p1_rewards[2]]
action_labels = [['lie', 'confess'],
                 ['head', 'tail'],
                 ['rock', 'paper', 'scissors']]
games = ['prisoners dilemma', 'matching pennies', 'rock paper scissors']


def normalize(policy):
    if (policy < 0).any():
        raise ValueError('Policy has negative value')
    total = policy.sum()
    return policy/total


def play_game(policies, rewards):
    n = len(rewards[0])
    p1_action = np.random.choice(n, p=policies[0])
    p2_action = np.random.choice(n, p=policies[1])
    p1_reward = rewards[0][p1_action, p2_action]
    p2_reward = rewards[1][p1_action, p2_action]
    return (p1_action, p2_action), (p1_reward, p2_reward)


def update_policy(policy, action_taken, reward, learning_rate=0.005, expected_values=None, update_alg='standard', N=None):
    if expected_values is None:
        expected_values = [0 for _ in range(len(policy))]
    #if update_alg == 'modified':
    #    learning_rate = min(learning_rate, 5000*learning_rate/N)
    for action in range(len(policy)):
        if action is not action_taken:
            if update_alg == 'standard':
                policy[action] += -learning_rate * reward * policy[action]
            elif update_alg == 'modified':
                policy[action] += -learning_rate * reward * policy[action] + learning_rate * (expected_values[action] - policy[action])
                expected_values[action] += learning_rate * (policy[action]-expected_values[action])
            else:
                raise ValueError('Invalid value supplied to update_alg')
        else:
            if update_alg == 'standard':
                policy[action] += learning_rate * reward * (1-policy[action])
            elif update_alg == 'modified':
                policy[action] += learning_rate * reward * (1-policy[action]) + learning_rate * (expected_values[action] - policy[action])
                expected_values[action] += learning_rate * (policy[action]-expected_values[action])
            else:
                raise ValueError('Invalid value supplied to update_alg')
    return normalize(policy), expected_values


def iterate(policy1, reward_matrix1, policy2, reward_matrix2, alpha=0.005, max_iterations=10000, update_alg='standard'):
    p_history = [[policy1],[policy2]]
    policies = [policy1, policy2]
    rewards = [reward_matrix1, reward_matrix2]
    expected_values = [[0 for _ in range(len(policy1))]] * 2
    Q = [[0 for _ in range(len(policy1))]] * 2
    N = [[0 for _ in range(len(policy1))]] * 2     
    for _ in range(max_iterations):
        action, reward = play_game(policies, rewards)
        for i in range(2):
            N[i][action[i]] += 1
            Q[i][action[i]] += 1/N[i][action[i]]*(reward[i]-Q[i][action[i]])
            policies[i], expected_values[i] = update_policy(policies[i], action[i], reward[i], learning_rate=alpha, expected_values=expected_values[i], update_alg=update_alg, N=N[i][action[i]])
            p_history[i].append(policies[i])
    return p_history[0], p_history[1], Q[0], Q[1]


def make_plots(player1_history, player2_history, action_labels):
    episodes = [n/1000.0 for n in range(len(player1_history))]
    players_list = [zip(*player1_history), zip(*player2_history)]
    figure = plt.figure()
    subplots = []
    subplots.append(figure.add_subplot(121))
    subplots.append(figure.add_subplot(122, sharey=subplots[0]))
    for subplot, player, i in zip(subplots, players_list, range(2)):
        for policy, action in zip(player, action_labels):
            subplot.plot(episodes, policy, label=action)
        subplot.set_xlabel('Episode ($*10^3$)')
        subplot.set_ylim([0,1])
        subplot.set_title('Player ' + str(i+1))
        subplot.legend()
    figure.text(0.06, 0.5, 'Action Probability', ha='center', va='center', rotation='vertical')
    plt.show()
    
def make_trajectory_plots(player1_history, player2_history, action_labels):
    if len(action_labels) == 2:
        figure = plt.figure()
        raise NotImplementedError
    elif len(action_labels) == 3:
        p1_hist = list(zip(*player1_history))
        p2_hist = list(zip(*player2_history))
        figure = plt.figure()
        ax3d = figure.add_subplot('111',projection='3d')
        ax3d.plot(p1_hist[0],p1_hist[1],p1_hist[2],'C1',lw=0.85)
        ax3d.scatter(p1_hist[0][-1],p1_hist[1][-1],p1_hist[2][-1],s=5,c='C1')
        ax3d.plot(p2_hist[0],p2_hist[1],p2_hist[2],'C2',lw=0.85)
        ax3d.scatter(p2_hist[0][-1],p2_hist[1][-1],p2_hist[2][-1],s=5,c='C2')
        ax3d.set_title('Policy Trajectory'.format())
        ax3d.set_xlabel('$P_{rock}$')
        ax3d.set_ylabel('$P_{paper}$')
        ax3d.set_zlabel('$P_{scissors}$')
    else:
        raise NotImplementedError
    
def statistics(p1_history, p2_history, labels):
    make_plots(p1_history, p2_history, labels)
    p1_lie, p1_confess = zip(*p1_history)
    p2_lie, p2_confess = zip(*p2_history)
    print('Converged to probabilities of action 1:\n\tPlayer 1:{}\n\tPlayer 2:{}\n'.format(p1_lie[-1], p2_lie[-1]))
    print('Converged to probabilities of action 2:\n\tPlayer 1:{}\n\tPlayer 2:{}\n'.format(p1_confess[-1], p2_confess[-1]))    


if __name__ == '__main__':
    matrix_1 = p1_rewards[0]
    matrix_2 = p2_rewards[0]

    policy_1 = normalize(np.random.random(2))
    policy_2 = normalize(np.random.random(2))

    p1_history, p2_history = iterate(policy_1, matrix_1, policy_2, matrix_2)
