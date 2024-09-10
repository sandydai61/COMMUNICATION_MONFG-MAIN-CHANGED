#复现一下这个算法
import numpy as np
from utils import *


class CoopActionAgent:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, id, u, du, alpha_q, alpha_theta, alpha_decay, num_actions, num_objectives, opt=False):
        self.id = id
        self.u = u  #一个目标函数
        self.du = du
        self.num_actions = num_actions  #可能的动作数量
        self.num_objectives = num_objectives #需要优化的目标数量
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.alpha_decay = alpha_decay
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)      #tmp = np.random.lognormal(average, squr) #bianlinag shizhengshu jiuzhemequ
                                                #self.policy = exp(tmp);
        self.op_policy = np.full(num_actions, 1 / num_actions)
        self.best_response_theta = np.zeros(num_actions)
        self.best_response_policy = softmax(self.best_response_theta)
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_actions, num_objectives))
        self.communicating = False

    def update(self, communicator, message, actions, reward):
        """
        This method will update the Q-table, strategy and internal parameters of the agent.
        :param communicator: The id of the communicating agent.
        :param message: The message that was sent. Unused by this agent.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained in this episode.
        :return: /
        """
        self.update_q_table(actions, reward)
        #计算期望Q值
        if self.id == 0:
            expected_q = self.op_policy @ self.q_table
        else:
            # We have to transpose axis 0 and 1 to interpret this as the column player.
            expected_q = self.op_policy @ self.q_table.transpose((1, 0, 2))
        theta, policy = self.update_policy(self.best_response_policy, self.best_response_theta, expected_q) #更新策略参数和策略
        self.theta = theta
        self.policy = policy
        self.best_response_theta = theta
        self.best_response_policy = policy
        self.update_parameters()

    def update_q_table(self, actions, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.q_table[actions[0], actions[1]] += self.alpha_q * (reward - self.q_table[actions[0], actions[1]])

    def update_policy(self, policy, theta, expected_q): #对两个参数求导
        """
        This method will update the given theta parameters and policy.
        :policy: The policy we want to update
        :theta: The current parameters for this policy.
        :expected_q: The Q-values for this policy.
        :return: Updated theta parameters and policy.
        """

        policy = np.copy(policy)  # This avoids some weird numpy bugs where the policy/theta is referenced by pointer.
        theta = np.copy(theta)
        expected_u = policy @ expected_q
        # We apply the chain rule to calculate the gradient.
        grad_u = self.du(expected_u)  # The gradient of u
        grad_pg = softmax_grad(policy).T @ expected_q  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        theta += self.alpha_theta * grad_theta
        policy = softmax(theta)
        return theta, policy

    def update_parameters(self):  #?为什么 decay = 1?
        """
        This method will update the internal parameters of the agent.
        :return: /
        这段代码的主要作用是 逐步减小学习率。通过将 alpha_q 和 alpha_theta 乘以一个衰减因子 alpha_decay，代理在训练过程中的学习率逐渐减小。这种设计有助于在训练早期快速学习，并在后期稳定收敛。开始时，较大的学习率能够帮助代理快速适应环境，而在后期，较小的学习率则能够使模型在接近最优策略时避免剧烈波动，最终达到更好的稳定性。
        """
        self.alpha_q *= self.alpha_decay
        self.alpha_theta *= self.alpha_decay

    def get_message(self):
        """
        This method will determine what action this agent will publish.
        :return: The current learned policy.
        """
        self.communicating = True
        return np.random.choice(range(self.num_actions), p=self.policy) #self.policy 是当前策略，它是一个概率分布，表示每个动作在当前策略下被选择的概率。

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if self.communicating:
            self.communicating = False
            return self.select_committed_action(message)
        else:
            return self.select_counter_action(message)

    def select_counter_action(self, action):  #？看不懂    在对抗中做出最佳反应
        """
        This method will update the policy based on the obtained message and select a response using the new policy.
        :param op_policy: The strategy committed to by the opponent.
        :return: The selected action.
        """
        op_policy = np.zeros(self.num_actions) #对手的策略
        op_policy[action] = 1
        self.op_policy = op_policy
        if self.id == 0:
            expected_q = self.op_policy @ self.q_table  #根据对手策略计算的期望Q值。通过将对手的策略 op_policy 与代理的 Q 表进行矩阵乘法得到的。这表示本代理是“行玩家”，对手是“列玩家”。
        else:
            # We have to transpose axis 0 and 1 to interpret this as the column player.
            expected_q = self.op_policy @ self.q_table.transpose((1, 0, 2))  #如果本代理的 ID 不为 0，则需要将 Q 表的第一个维度和第二个维度转置后再进行计算。这表示本代理是“列玩家”，对手是“行玩家”。
        best_response_theta, best_response_policy = self.update_policy(self.policy, self.theta, expected_q)
        self.best_response_theta = best_response_theta
        self.best_response_policy = best_response_policy  #代理将其策略调整为对当前对手动作的最佳响应。
        return np.random.choice(range(self.num_actions), p=self.best_response_policy)

    @staticmethod
    def select_committed_action(action):
        """
        This method simply plays the action that was committed.
        :return: The committed action.
        """
        return action
