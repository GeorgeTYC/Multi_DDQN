import numpy as np
from utils.Enviroment import Enviroment
from utils.Action import Actions

def gather_stats(agent, env: Enviroment, e):
    """ Compute average rewards over 10 episodes
    """
    score = []
    for k in range(3):
        old_state = env.reset(e)
        cumul_r, done = 0, False
        actions = Actions()
        while not done:
            a = agent.policy_action(old_state.reshape_state())
            actions.set_action(action=a, order=actions.get_order())
            old_state, r, done = env.step(actions=actions, state=old_state)
            cumul_r += r
        score.append(cumul_r)
    return np.mean(np.array(score)), np.std(np.array(score))