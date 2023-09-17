from random import random, randrange
import numpy as np
from copy import deepcopy
import tensorflow as tf

from .agent import Agent
from tqdm import tqdm
from .MemoryBuffer import MemoryBuffer
from .stats import gather_stats
from .Action import Actions
from .Enviroment import Enviroment

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, state_dim, action_dim, args):

        self.with_per = args.with_per
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.learning_rate = 2.5e-4
        self.gamma = 0.95
        self.epsilon = 1
        # TODO: 做修改
        self.epsilon_decay = 0.998
        self.buffer_size = 20000
        self.batch_size = args.batch_size
        self.training_step = 20

        self.tau = 1e-2
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, self.action_dim, self.learning_rate, self.tau)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        # index = np.argmax(self.agent.predict(s)[0])
        if random() <= self.epsilon:
            index = randrange(self.action_dim)
        else:
            index = np.argmax(self.agent.predict(s)[0])
        action = np.zeros((self.action_dim, 1), dtype=bool)
        action[index] = True
        return action

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        # q = self.agent.predict(s)
        # next_q = self.agent.predict(new_s)
        # q_targ = self.agent.target_predict(new_s)

        action_dim = a[0].shape[0]
        q = np.zeros((s.shape[0], action_dim))
        next_q = np.zeros((s.shape[0], action_dim))
        q_targ = np.zeros((s.shape[0], action_dim))
        for i in range(s.shape[0]):
            q[i] = self.agent.predict(s[i])
            next_q[i] = self.agent.predict(new_s[i])
            q_targ[i] = self.agent.target_predict(new_s[i])

        for i in range(s.shape[0]):
            old_q = q[i, a[i].reshape(-1)]
            if d[i]:
                q[i, a[i].reshape(-1)] = r[i]
            else:
                next_best_action = np.argmax(next_q[i, :])
                q[i, a[i].reshape(-1)] = r[i] + self.gamma * q_targ[i, next_best_action]
            if (self.with_per):
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))

        # Train on batch
        for i in range(s.shape[0]):
            self.agent.fit(s[i], q[i])

        # self.agent.fit(s, q)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay

    def train(self, env: Enviroment, args, summary_writer):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        step = 0
        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset(e)
            actions = Actions()
            while not done:
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state.reshape_state())
                order = actions.get_order()
                actions.set_action(action=a, order=actions.get_order())
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done = env.step(actions=actions, state=old_state)
                # Memorize for experience replay
                self.memorize(deepcopy(old_state).reshape_state(), a, r, done, deepcopy(new_state).reshape_state())
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                step += 1
                # Train DDQN and transfer weights to target network
                # 每隔20个step进行一次训练
                if (self.buffer.size() > self.batch_size):
                    if step >= self.training_step:
                        self.train_agent(self.batch_size)
                        self.agent.transfer_weights()
                        step = 0

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, deepcopy(env),e)
                results.append([e, mean, stdev, self.epsilon])

            # Export results for Tensorboard
            # score = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='score', simple_value=cumul_reward)])
            # tf.summary.scalar("my_metric",score, e)
            # summary_writer.add_summary(score, global_step=e)
            tf.summary.scalar('score', cumul_reward, step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """

        # td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state)



    def save_weights(self, path):
        path += '_LR_{}'.format(self.learning_rate)
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)
