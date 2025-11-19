import tensorflow as tf
import numpy as np
from collections import deque


class QNetwork:
    """
    DRQN with a dueling output head.

    - LSTM encoder over history of states
    - Two fully connected streams:
        * Value stream V(s)
        * Advantage stream A(s, a)
      Combined as Q(s, a) = V(s) + (A(s,a) - mean_a A(s,a))
    """

    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10, step_size=1,
                 name='QNetwork'):

        # 关闭 eager，以便兼容 TF1 风格的占位符与 Session
        tf.compat.v1.disable_eager_execution()

        with tf.compat.v1.variable_scope(name):
            # 输入：形状为 [batch, step_size, state_size] 的序列
            self.inputs_ = tf.compat.v1.placeholder(
                tf.float32, [None, step_size, state_size], name='inputs_')
            self.actions_ = tf.compat.v1.placeholder(
                tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            self.targetQs_ = tf.compat.v1.placeholder(
                tf.float32, [None], name='target')

            # 使用 v1 rnn_cell API 构造 LSTM
            lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size)
            self.lstm_out, self.state = tf.compat.v1.nn.dynamic_rnn(
                lstm_cell, self.inputs_, dtype=tf.float32)

            # 取序列最后一个时间步的输出
            self.reduced_out = self.lstm_out[:, -1, :]
            self.reduced_out = tf.reshape(
                self.reduced_out, shape=[-1, hidden_size])

            # 共享前馈层
            self.w2 = tf.Variable(tf.random.uniform([hidden_size, hidden_size]))
            self.b2 = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
            self.h2 = tf.nn.relu(tf.matmul(self.reduced_out, self.w2) + self.b2)

            # ---------------------
            # Dueling 结构
            # ---------------------
            # Value stream V(s)
            self.w_value = tf.Variable(tf.random.uniform([hidden_size, 1]))
            self.b_value = tf.Variable(tf.constant(0.1, shape=[1]))
            self.value = tf.matmul(self.h2, self.w_value) + self.b_value  # [batch, 1]

            # Advantage stream A(s, a)
            self.w_adv = tf.Variable(tf.random.uniform([hidden_size, action_size]))
            self.b_adv = tf.Variable(tf.constant(0.1, shape=[action_size]))
            self.advantage = tf.matmul(self.h2, self.w_adv) + self.b_adv  # [batch, action_size]

            # Combine into Q values
            adv_mean = tf.reduce_mean(self.advantage, axis=1, keepdims=True)
            self.output = self.value + (self.advantage - adv_mean)

            # Q 值以及损失
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.compat.v1.train.AdamOptimizer(
                learning_rate).minimize(self.loss)


class Memory():
    """
    简单经验回放（暂未加权重），方便后续扩展为优先经验回放（PER）。
    """

    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, step_size):
        idx = np.random.choice(
            np.arange(len(self.buffer) - step_size),
            size=batch_size, replace=False)

        res = []
        for i in idx:
            temp_buffer = []
            for j in range(step_size):
                temp_buffer.append(self.buffer[i + j])
            res.append(temp_buffer)
        return res
