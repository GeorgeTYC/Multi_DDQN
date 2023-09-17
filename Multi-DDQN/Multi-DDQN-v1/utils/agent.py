from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

import numpy as np



class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, learning_rate, tau):
        '''

        :param state_dim: 代理能够观察到的环境状态的纬度
        :param action_dim: 动作纬度
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau

        # 训练参数
        self.learning_rate = learning_rate
        self.alpha = 0.005
        self.gamma = 0.95


        # 初始化网络
        self.model = self.network()
        self.model.compile(Adam(self.learning_rate), 'mse')
        # 构建目标Q网络
        self.target_model = self.network()
        self.target_model.compile(Adam(self.learning_rate), 'mse')
        self.target_model.set_weights(self.model.get_weights())


    def network(self):
        """construct network"""
        inp = Input(self.state_dim)
        x = Dense(256, activation='relu')(inp)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.action_dim, activation='linear')(x)

        return Model(inp, x)

    def transfer_weights(self):
        '''
        用于在模型和目标模型之间传输权重
        :return:
        '''
        W = self.model.get_weights()
        target_W = self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1-self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def reshape(self, x):
        return np.expand_dims(x,axis=0)

    def fit(self, inp, target):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), self.reshape(target), epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(self.reshape(inp))

    def save(self, path):
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)