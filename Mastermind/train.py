import numpy as np
from Policy import Policy
from MasterMindEnv import MasterMindEnv
import torch.nn as nn
import torch as T
import config


def train(num_episodes=1000, gamma=0.99):
    policy = Policy(batch_size=1, n_guesses=config.maxLength)
    total_rewards = []
    steps = []
    saved_log_probs = []

    for e in range(num_episodes):
        target = np.random.choice(
            ['red', 'green', 'blue'], size=3)

        env = MasterMindEnv(policy, target)
        random_guess = np.random.choice(
            ['red', 'green', 'blue'], size=3)

        # print("TARGET:", target)
        random_feedback = env.feedback_from_guess(
            random_guess)

        random_guess_embedding = env.guess_to_embedding(random_guess)
        random_guess_embedding.extend([random_feedback])

        history = [random_guess_embedding] + \
            [[0]*10] * (config.maxLength-1)

        guessed_right = False

        for i in range(config.maxLength-1):
            action, log_prob = policy.act(history)
            saved_log_probs.append(log_prob)
            state, reward, done = env.step(history, action, i)
            history = state
            saved_log_probs.append(log_prob)
            if done:
                steps.append(i)
                guessed_right = True
                total_rewards.append(1)
                break

        if not guessed_right:
            total_rewards.append(-1)
        steps.append(config.maxLength)

        if e % 256 == 0:
            policy.optimizer.zero_grad()
            G = np.zeros_like(total_rewards, dtype=np.float64)
            for t in range(len(total_rewards)):
                G_sum = 0
                discount = 1
                for k in range(t, len(total_rewards)):
                    G_sum += total_rewards[k] * discount
                    discount *= gamma

                G[t] = G_sum
            mean = np.mean(G)
            std = np.std(G) if np.std(G) > 0 else 1
            G = (G - mean) / std
            G = T.tensor(G, dtype=T.float32)

            policy_loss = 0
            for g, log_prob in zip(G, saved_log_probs):
                policy_loss += -g * log_prob

            policy_loss.backward()
            policy.optimizer.step()

            percent_correct = total_rewards.count(1) / len(total_rewards)
            print('Episode {}\tAverage Score: {:.2f}\tAverage Steps: {:.2f}\tAccuracy: {:.2f}'.format(
                e, np.mean(total_rewards), np.mean(steps), percent_correct))
            total_rewards = []
            steps = []
            saved_log_probs = []

    return total_rewards


def main():
    train(num_episodes=100000)


main()
