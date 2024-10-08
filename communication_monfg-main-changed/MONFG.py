import argparse
import time

#from comp_action_agent import CompActionAgent
from coop_action_agent import CoopActionAgent
from notus_coop_policy_agent import CoopPolicyAgent
from data import save_metadata, save_data
from games import *
from no_com_agent import NoComAgent
from optional_com_agent import OptionalComAgent
from utils import *


def select_actions(agents, message):  
    """
    This function selects an action from each agent's policy.
    :param agents: The list of agents.
    :param message: The message from the leader.
    :return: A list of selected actions.
    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action(message))
    return selected


def calc_payoffs(agents, actions, payoff_matrix):  #写一个这个函数，输入是三个函数的输入，输出是三个函数的输出
    """
    This function will calculate the payoffs of the agents.
    :param agents: The list of agents.
    :param actions: The action that each agent chose.
    :param payoff_matrix: The payoff matrix.
    :return: A list of received payoffs.
    """
    payoffs = []
    for _ in agents:
        payoffs.append(payoff_matrix[actions[0]][actions[1]])  # Append the payoffs from the actions.
    return payoffs


def calc_returns(payoffs, agents, rollouts):
    """
    This function will calculate the scalarised expected returns for each agent.
    :param payoffs: The payoffs obtained by the agents.
    :param agents: The agents in this experiment
    :param rollouts: The amount of rollouts that were performed.
    :return: A list of scalarised expected returns.
    """
    returns = []
    for idx, payoff_hist in enumerate(payoffs):
        payoff_sum = np.sum(payoff_hist, axis=0)
        avg_payoff = payoff_sum / rollouts
        ser = agents[idx].u(avg_payoff)
        returns.append(ser)
    return returns


def calc_action_probs(actions, num_actions, rollouts):
    """
    This function will calculate the empirical action probabilities.
    :param actions: The actions performed by each agent over the rollout period.
    :param num_actions: The number of possible actions.
    :param rollouts: The number of rollouts that were performed.
    :return: The action probabilities for each agent.
    """
    all_probs = []

    for action_hist in actions:
        probs = np.zeros(num_actions)

        for action in action_hist:
            probs[action] += 1

        probs = probs / rollouts
        all_probs.append(probs)

    return all_probs


def calc_com_probs(messages, rollouts):
    """
    This function will calculate the empirical communication probabilities.
    :param messages: The messages that were sent.
    :param rollouts: The number of rollouts that were performed.
    :return: The communication probabilities for each agent.
    """
    com = sum(message is not None for message in messages)
    no_com = (rollouts - com)
    return [com / rollouts, no_com / rollouts]


def update(agents, communicator, message, actions, payoffs):
    """
    This function gets called after every episode so that agents can update their internal mechanisms.
    :param agents: A list of agents.
    :param communicator: The id of the communicating agent.
    :param message: The message that was sent.
    :param actions: A list of each action that was chosen, indexed by agent.
    :param payoffs: A list of each payoff that was received, indexed by agent.
    :return:
    """
    for idx, agent in enumerate(agents):
        agent.update(communicator, message, actions, payoffs[idx])


def reset(experiment, num_agents, num_actions, num_objectives, alpha_lq, alpha_ltheta, alpha_fq, alpha_ftheta, alpha_mq,
          alpha_mtheta, alpha_decay, opt=False):
    """
    This function will create new agents that can be used in a new trial.
    :param experiment: The type of experiments we are running.
    :param num_agents: The number of agents to create.
    :param num_actions: The number of actions each agent can take.
    :param num_objectives: The number of objectives they have.
    :param alpha_lq: The learning rate for the leader's Q values.
    :param alpha_ltheta: The learning rate for leader's theta.
    :param alpha_fq: The learning rate for the follower's Q values.
    :param alpha_ftheta: The learning rate for the follower's theta.
    :param alpha_mq: The learning rate for Q-values for communication in the optional communication experiments.
    :param alpha_mtheta: The learning rate for learning a messaging strategy in the optional communication experiments.
    :param alpha_decay: The learning rate decay.
    :param opt: A boolean that decides on optimistic initialization of the Q-tables.
    :return:
    """
    agents = []
    for ag in range(num_agents):
        u, du = get_u_and_du(ag + 1)  # The utility function and derivative of the utility function for this agent.
        if experiment == 'no_com':
            new_agent = NoComAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives, opt)
        elif experiment == 'comp_action':
            new_agent = CompActionAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_fq, alpha_ftheta, alpha_decay,
                                        num_actions, num_objectives, opt)
        elif experiment == 'coop_action':
            new_agent = CoopActionAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives,
                                        opt)
        elif experiment == 'coop_policy':
            new_agent = CoopPolicyAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives,
                                        opt)
        elif experiment == 'opt_comp_action':
            no_com_agent = NoComAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives, opt)
            com_agent = CompActionAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_fq, alpha_ftheta, alpha_decay,
                                        num_actions, num_objectives, opt)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, du, alpha_mq, alpha_mtheta, alpha_decay,
                                         num_objectives, opt)
        elif experiment == 'opt_coop_action':
            no_com_agent = NoComAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives, opt)
            com_agent = CoopActionAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives,
                                        opt)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, du, alpha_mq, alpha_mtheta, alpha_decay,
                                         num_objectives, opt)
        elif experiment == 'opt_coop_policy':
            no_com_agent = NoComAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives, opt)
            com_agent = CoopPolicyAgent(ag, u, du, alpha_lq, alpha_ltheta, alpha_decay, num_actions, num_objectives,
                                        opt)
            new_agent = OptionalComAgent(no_com_agent, com_agent, ag, u, du, alpha_mq, alpha_mtheta, alpha_decay,
                                         num_objectives, opt)
        else:
            raise Exception('Something went wrong!')
        agents.append(new_agent)
    return agents


def run_experiment(experiment, runs, episodes, rollouts, payoff_matrix, opt_init, seed=1):
    """
    This function will run the requested experiment.
    :param experiment: The type of experiment we are running.  实验的类型
    :param runs: The number of different runs.                  实验的次数
    :param episodes: The number of episodes in each run.        运行中的集数
    :param rollouts: The rollout period for the policies.       每次策略执行的滚动期数
    :param payoff_matrix: The payoff matrix for the game.        收益矩阵
    :param opt_init: A boolean that decides on optimistic initialization of the Q-tables.    表示是否进行乐观初始化
    :param seed: An optional seed for random number generation.   随机种子
    :return: A log of payoffs, a log for action probabilities for both agents and a log of the state distribution.
    """
    if seed is not None:
        np.random.seed(seed=seed)

    # Setting hyperparameters.
    num_agents = 2
    num_actions = payoff_matrix.shape[0]  #动作的数量
    num_objectives = 2
    alpha_lq = 0.01
    alpha_ltheta = 0.01
    alpha_fq = 0.1
    alpha_ftheta = 0.1
    alpha_mq = 0.01
    alpha_mtheta = 0.01
    alpha_decay = 1

    # Setting up lists containing the results.
    returns_log = [[] for _ in range(num_agents)]   #记录回报日志
    action_probs_log = [[] for _ in range(num_agents)]  #记录动作概率日志
    com_probs_log = [[] for _ in range(num_agents)]     #记录通信概率日志
    state_dist_log = np.zeros((num_actions, num_actions))  #记录每次运行状态的分布日志
    metadata = {
        'experiment': experiment,
        'runs': runs,
        'episodes': episodes,
        'rollouts': rollouts,
        'payoff_matrix': payoff_matrix.tolist(),
        'opt_init': opt_init,
        'seed': seed,
        'num_agents': num_agents,
        'alpha_lq': alpha_lq,
        'alpha_ltheta': alpha_ltheta,
        'alpha_fq': alpha_fq,
        'alpha_ftheta': alpha_ftheta,
        'alpha_mq': alpha_mq,
        'alpha_mtheta': alpha_mtheta,
        'alpha_decay': alpha_decay
    }

    start = time.time()

    for run in range(runs):      #实验运行循环 每次实验100次   100 * 5000
        print("Starting run: ", run)
        agents = reset(experiment, num_agents, num_actions, num_objectives, alpha_lq, alpha_ltheta, alpha_fq,
                       alpha_ftheta, alpha_mq, alpha_mtheta, alpha_decay, opt_init)  #调用 reset 函数重置和初始化代理。

        for episode in range(episodes):   #集数循环 每集5000次迭代
            # We keep the actions and payoffs of this episode so that we can later calculate the SER.
            ep_actions = [[] for _ in range(num_agents)]
            ep_payoffs = [[] for _ in range(num_agents)]
            ep_messages = []

            # Select the communicator in round-robin fashion.
            communicator = episode % len(agents)   #选择通信代理
            communicating_agent = agents[communicator]

            for rollout in range(rollouts):  # Required to evaluate the SER and action probabilities.
                message = communicating_agent.get_message()    #滚动期循环
                actions = select_actions(agents, message)
                payoffs = calc_payoffs(agents, actions, payoff_matrix)  #改成目标函数，写一个大函数 #payoff_matrix为什么这么多东西

                # Log the results of this roll
                for idx in range(num_agents):
                    ep_actions[idx].append(actions[idx])
                    ep_payoffs[idx].append(payoffs[idx])
                ep_messages.append(message)
 
            # Update the agent after the episode
            # We use the last action and payoff to update the agent. It doesn't really matter which rollout we select
            # to update our agent as the agent doesn't learn any new information during the rollout.
            last_actions = np.array(ep_actions)[:, -1]    #更新代理
            last_payoffs = np.array(ep_payoffs)[:, -1]
            last_message = ep_messages[-1]
            update(agents, communicator, last_message, last_actions, last_payoffs)  # Update the agents.

            # Get the necessary results from this episode.  #计算结果
            action_probs = calc_action_probs(ep_actions, num_actions, rollouts)
            returns = calc_returns(ep_payoffs, agents, rollouts)
            com_probs = calc_com_probs(ep_messages, rollouts)

            # Append the logs.  #修改日志
            for idx in range(num_agents):
                returns_log[idx].append([run, episode, returns[idx]])
                prob_log = [run, episode] + action_probs[idx].tolist()
                action_probs_log[idx].append(prob_log)
            com_log = [run, episode] + com_probs
            com_probs_log[episode % num_agents].append(com_log)

            # If we are in the last 10% of episodes we build up a state distribution log.
            # This code is specific to two player games.
            if episode >= 0.9 * episodes:      #状态分布日志  在最后10%的集中，记录状态分布
                state_dist = np.zeros((num_actions, num_actions))
                for a1, a2 in zip(ep_actions[0], ep_actions[1]):
                    state_dist[a1, a2] += 1
                state_dist /= rollouts
                state_dist_log += state_dist  

    end = time.time()   #计算和返回结果
    elapsed_mins = (end - start) / 60.0
    print("Minutes elapsed: " + str(elapsed_mins))

    return returns_log, action_probs_log, com_probs_log, state_dist_log, metadata


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

#解析命令行参数
    parser.add_argument('--game', type=str, default='game1', help="which MONFG game to play")
    parser.add_argument('--experiment', type=str, default='opt_coop_policy', help='The experiment to run.')
    parser.add_argument('--runs', type=int, default=100, help="number of trials")
    parser.add_argument('--episodes', type=int, default=5000, help="number of episodes")
    parser.add_argument('--rollouts', type=int, default=100, help="Rollout period for the policies")
    parser.add_argument('--opt_init', action='store_true', help="optimistic initialization")

    args = parser.parse_args()

    # Extracting the arguments.
    #提取参数
    game = args.game
    experiment = args.experiment
    runs = args.runs
    episodes = args.episodes
    rollouts = args.rollouts
    opt_init = args.opt_init

    # Starting the experiments.
    #获取收益矩阵
    payoff_matrix = get_payoff_matrix(game)
    #运行试验
    data = run_experiment(experiment, runs, episodes, rollouts, payoff_matrix, opt_init)
    returns_log, action_probs_log, com_probs_log, state_dist_log, metadata = data

    # Writing the data to disk.
    #保存数据
    path = create_game_path('C:/2MyFile/1Study/2lunwen', experiment, game, opt_init)
    #print(path)
    mkdir_p(path)
    save_metadata(path, **metadata)
    save_data(path, experiment, game, returns_log, action_probs_log, com_probs_log, state_dist_log, runs, episodes)
