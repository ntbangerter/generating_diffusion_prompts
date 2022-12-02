# Write a program to play the game MasterMind in OpenAI Gym
import config
import numpy as np
import torch as T


class MasterMindEnv():
    def __init__(self, policy, target):
        self.policy = policy
        self.target = target
        self.rewards = []
        self.actions = []
        # Change to three
        self.color_to_embedding = {
            'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}

        self.number_to_color = {0: 'red', 1: 'green', 2: 'blue'}

    def guess_to_embedding(self, guess):
        guess_embedding = []
        for color in guess:
            guess_embedding.extend(self.color_to_embedding[color])
        return guess_embedding

    def embedding_to_guess(self, embedding):
        n = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if n == embedding:
                        return [self.number_to_color[i], self.number_to_color[j], self.number_to_color[k]]
                    n += 1

    def feedback_from_guess(self, guess):
        # Red, blue, green
        correct_guess = 0
        correct_pos = 0
        target_copy = list(self.target.copy())

        for i in range(len(guess)):
            if(guess[i] == self.target[i]):
                correct_pos += 1
                target_copy[i] = -1

        for i in range(len(guess)):
            if(guess[i] in target_copy):
                correct_guess += 1
                target_copy.remove(guess[i])

        return correct_pos + 0.5*correct_guess

    def step(self, history, guess, i):
        feedback = self.feedback_from_guess(self.embedding_to_guess(guess))
        next_guess = self.guess_to_embedding(
            self.embedding_to_guess(guess))
        next_guess.extend([feedback])
        history[i+1] = next_guess
        done = feedback == 3
        return history, feedback, done
