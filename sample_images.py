import random
from PIL import Image
from image_similarity import ImageSimilarity
import pickle
from tqdm import tqdm

similarity = ImageSimilarity()

styles = ['photo', 'cartoon', 'painting']
subjects = ['an astronaut riding a horse', 'a teddy bear', 'a blue cube']
settings = ['on mars', 'in the ocean', 'in a field']

def get_prompt(style, subject, setting):
    prompt = "a {} of {} {}".format(style, subject, setting)
    return prompt

def sample_image():
    prompt_i = random.randint(0, 26)
    prompt = prompts[prompt_i]
    i = random.randint(0, 200)
    guess_embedding = prompt_to_embedding[prompt].copy()
    
    target_embedding = [0 for _ in range(27)]
    target_embedding[prompt_i] = 1
    
    return Image.open("./data/{}_{}.jpg".format(prompt, i)), guess_embedding, target_embedding

def get_data(n_guesses = 4):
    target_image, _, target_embedding = sample_image()
    target_embedding = torch.Tensor(target_embedding)
    
    guess_embeddings = []
    
    for i in range(n_guesses):
        image, guess_embedding, _ = sample_image()
        
        sim = similarity.get_image_similarity_score(target_image, image)
        guess_embedding.append(sim)
        guess_embedding = torch.Tensor(guess_embedding)
        
        guess_embeddings.append(guess_embedding)
        
    guess_embeddings = torch.stack(guess_embeddings)
    
    return target_embedding, guess_embeddings

def get_batch(batch_size, n_guesses = 4):
    batch_target = []
    batch_guesses = []
    for _ in range(batch_size):
        target, guesses = get_data(n_guesses)
        batch_target.append(target)
        batch_guesses.append(guesses)
    
    batch_target = torch.stack(batch_target)
    batch_guesses = torch.stack(batch_guesses)
    
    return batch_target, batch_guesses


batches = []

for i in tqdm(range(10000)):
    if (i + 1) % 100:
        pickle.dump(batches, open('batches.p', 'wb'))
    batches.append(get_batch(128, 4))
