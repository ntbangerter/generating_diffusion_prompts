import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, batch_size = 32, n_guesses = 4):
        super(Network, self).__init__()
        
        self.batch_size = batch_size
        self.n_guesses = n_guesses

        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64*n_guesses, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 27)
        self.softmax = nn.Softmax(dim=1)

        self.style1 = nn.Linear(4, 16)
        self.style2 = nn.Linear(16, 16)
        self.style3 = nn.Linear(16*n_guesses, 128)
        self.style4 = nn.Linear(128, 128)

        self.subject1 = nn.Linear(4, 16)
        self.subject2 = nn.Linear(16, 16)
        self.subject3 = nn.Linear(16*n_guesses, 128)
        self.subject4 = nn.Linear(128, 128)

        self.setting1 = nn.Linear(4, 16)
        self.setting2 = nn.Linear(16, 16)
        self.setting3 = nn.Linear(16*n_guesses, 128)
        self.setting4 = nn.Linear(128, 128)

        self.final1 = nn.Linear(64*n_guesses + 128*3, 128)
        self.final2 = nn.Linear(128, 128)
        self.final3 = nn.Linear(128, 27)


    def get_guess_embedding(self, guess):
        embedding = self.fc1(guess)
        embedding = self.fc2(embedding)

        return embedding
    
    def get_crosswise_embeddings(self, guess):
        style = torch.cat((guess[:, :, :3], guess[:, :, -1:]), 2)
        style = self.style1(style)
        style = self.style2(style).reshape(self.batch_size, self.n_guesses*16)
        
        subject = torch.cat((guess[:, :, 3:6], guess[:, :, -1:]), 2)
        subject = self.subject1(subject)
        subject = self.subject2(subject).reshape(self.batch_size, self.n_guesses*16)
        
        setting = torch.cat((guess[:, :, 6:9], guess[:, :, -1:]), 2)
        setting = self.setting1(setting)
        setting = self.setting2(setting).reshape(self.batch_size, self.n_guesses*16)
        
        return style, subject, setting

    def forward(self, x):
        guess_embeddings = x.reshape(self.batch_size * self.n_guesses, 10)
        guess_embeddings = self.get_guess_embedding(guess_embeddings)
        guess_embeddings = guess_embeddings.reshape(self.batch_size, 64*self.n_guesses)

        styles, subjects, settings = self.get_crosswise_embeddings(x)

        styles = self.style3(styles)
        styles = self.style4(styles)

        subjects = self.subject3(subjects)
        subjects = self.subject4(subjects)

        settings = self.setting3(settings)
        settings = self.setting4(settings)

        out = torch.cat([guess_embeddings, styles, subjects, settings], dim=1)

        out = self.final1(out)
        out = self.final2(out)
        out = self.final3(out)
        out = self.softmax(out)

        return out
