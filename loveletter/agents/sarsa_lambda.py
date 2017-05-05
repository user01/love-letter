import numpy as np

class AgentSarsaLambda(Agent):
    '''Agent which leverages Sarsa Lambda Learning'''

    def __init__(self, seed = 451,
                q_path = 'loveletter/models/sarsa_lambda_Q.npz'):
        with np.load(q_path) as data:
            Q= data['Q']
        

    def _move(self, game):
        assert game.active()

        S = nn_to_ss(game.state())

        if (np.random.random() > epsilon):
            A = np.argmax(Q[(...) + tuple(S)])
        else:
            A = np.random.randint(15)

    def nn_to_ss(state):
        lh = np.argmax(state[0:8])
        rh = np.argmax(state[8:16])
        cards = np.multiply(state[16:24], [5, 2, 2, 2, 2, 1, 1, 1]).astype(int)

        return np.concatenate(([lh, rh], cards))