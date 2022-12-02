from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import random
from network import Network

styles = ['photo', 'cartoon', 'painting']
subjects = ['an astronaut riding a horse', 'a teddy bear', 'a blue cube']
settings = ['on mars', 'in the ocean', 'in a field']

def get_prompt(style, subject, setting):
    prompt = "a {} of {} {}".format(style, subject, setting)
    return prompt

prompts = []
prompt_to_embedding = {}
for i, style in enumerate(styles):
    for j, subject in enumerate(subjects):
        for k, setting in enumerate(settings):
            prompt = get_prompt(style, subject, setting)
            prompts.append(prompt)
            
            style_embed = [0, 0, 0]
            subject_embed = [0, 0, 0]
            setting_embed = [0, 0, 0]
            style_embed[i] = 1
            subject_embed[j] = 1
            setting_embed[k] = 1
            
            prompt_to_embedding[prompt] = style_embed + subject_embed + setting_embed



losses = []
accuracy = []

batch_size = 128
n_guesses = 4
lr = 1e-5
print_every = 1000
save_every = 10000

net = Network(batch_size, n_guesses)
net = net.to('cuda')

epochs = 1000

def train():
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    step = 0
    batches = pickle.load(open('batches (copy).p', 'rb'))
    
    for epoch in tqdm(range(epochs)):
        random.shuffle(batches)
        for target, data in batches:
            step += 1

            target = target.to('cuda')
            data = data.to('cuda')

            optimizer.zero_grad()

            out = net(data)
            loss = criterion(out, target)
            loss.backward()

            losses.append(loss.item())
            optimizer.step()

            acc = (out.argmax(1) == target.argmax(1)).float().mean()
            accuracy.append(acc.item())
            
        plt.plot(losses)
        plt.show()

            if step % print_every == 0:
                plt.plot(losses)
                plt.show()

            if step % save_every == 0:
                torch.save(net.state_dict(), "weights{}.pt".format(step))
            
train()
