import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Rewards:
    def __init__(self, actions=10):
        self._q_means = np.random.normal(0, 1, size=actions)
        self._best_action = np.argmax(self._q_means)
        self._rewards = []
        self._is_optimal = []

    def get_reward(self, action):
        mean = self._q_means[action]
        reward = np.random.normal(mean, 1)
        self._track_reward(action, reward)
        return reward

    def _track_reward(self, action, reward):
        self._rewards.append(reward)
        self._is_optimal.append(action == self._best_action)

    @property
    def rewards(self):
        return self._rewards

    @property
    def optimal(self):
        return self._is_optimal


class ActionValue:
    def __init__(self, actions=10, alpha=0.1, init=0):
        self._qs = np.zeros(shape=(actions,)) + init
        self._selections = np.zeros(shape=(actions,))
        self._alpha = alpha

    def update(self, action, reward):
        q = self._qs[action]
        self._selections[action] += 1
        self._qs[action] += 1 / self._selections[action] * (reward-q)

    def greedy(self, epsilon=0):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self._qs.size)
        else:
            return np.argmax(self._qs)

    def softmax(self, temperature=1):
        xs = np.exp(self._qs/temperature)
        xs = xs / np.sum(xs)
        ys = np.random.multinomial(1, xs, 1)
        return np.argmax(ys)


def run_experiments(experiments, actions, plays, epsilon, softmax, temperature):
    average_rewards = []
    optimal_actions = []
    for _ in tqdm(range(experiments)):
        rewards = Rewards(actions=actions)
        av = ActionValue(actions=actions)

        for _ in range(plays):
            if softmax:
                action = av.softmax(temperature=temperature)
            else:
                action = av.greedy(epsilon=epsilon)
            r = rewards.get_reward(action)
            av.update(action, r)
        average_rewards.append(np.asarray(rewards.rewards))
        optimal_actions.append(np.asarray(rewards.optimal, dtype=np.int))
    average_rewards = np.stack(average_rewards, axis=0)
    average_rewards = np.mean(average_rewards, axis=0)
    optimal_actions = np.stack(optimal_actions, axis=0)
    optimal_actions = np.mean(optimal_actions, axis=0)
    return average_rewards, optimal_actions


def main():
    actions = 10
    plays = 1000
    experiments = 2000

    np.random.seed(0)

    experiment_outcomes = {}

    experiment_outcomes['greedy, e=0'] = run_experiments(
        experiments, actions, plays, epsilon=0, softmax=False, temperature=0)
    experiment_outcomes['greedy, e=0.1'] = run_experiments(
        experiments, actions, plays, epsilon=0.1, softmax=False, temperature=0)
    experiment_outcomes['greedy, e=0.01'] = run_experiments(
        experiments, actions, plays, epsilon=0.01, softmax=False, temperature=0)
    experiment_outcomes['softmax, t=1'] = run_experiments(
        experiments, actions, plays, epsilon=0, softmax=True, temperature=1)
    experiment_outcomes['softmax, t=4'] = run_experiments(
        experiments, actions, plays, epsilon=0, softmax=True, temperature=4)
    experiment_outcomes['softmax, t=10'] = run_experiments(
        experiments, actions, plays, epsilon=0, softmax=True, temperature=10)
    experiment_outcomes['softmax, t=100'] = run_experiments(
        experiments, actions, plays, epsilon=0, softmax=True, temperature=100)

    _, axs = plt.subplots(2)
    for title, (average_rewards, optimal_actions) in experiment_outcomes.items():
        axs[0].plot(average_rewards, label=title)
        axs[0].set_title('Average rewards')
        axs[1].plot(optimal_actions, label=title)
        axs[1].set_title('Optimal Actions %')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
