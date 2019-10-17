import numpy as np

prisoners_dilemma = np.array([[5, 0],
                              [10, 1]])

matching_coins = np.array([[1, -1],
                           [-1, 1]])

rock_paper_scissors = np.array([[0, -1, 1],
                                [1, 0, -1],
                                [-1, 1, 0]])

alpha = 0.5
rewards = [prisoners_dilemma, matching_coins, rock_paper_scissors]


def normalize(p1, p2):
    t = p1 + p2
    return p1/t, p2/t


if __name__ == '__main__':
    NotImplementedError()  # do policy iteration
