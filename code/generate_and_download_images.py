import time
import openai
import requests
from PIL import Image
import numpy as np
from io import BytesIO

openai.api_key = 'ADD_OPENAI_KEY_HERE'

def get_image(prompt):
    url = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256"
    )['data'][0]['url']

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    return img

styles = ['photo', 'cartoon', 'painting']
subjects = ['an astronaut riding a horse', 'a teddy bear', 'a blue cube']
settings = ['on mars', 'in the ocean', 'in a field']

def get_prompt(style, subject, setting):
    prompt = "a {} of {} {}".format(style, subject, setting)
    return prompt

start_at = 105
n_images = 400
n = start_at*27
total = n_images * len(styles) * len(subjects) * len(settings)

for i in range(start_at, n_images):
    for style in styles:
        for subject in subjects:
            for setting in settings:
                n += 1
                print("{} - {} / {} images".format(i, n, total))
                prompt = get_prompt(style, subject, setting)
                try:
                    image = get_image(prompt)
                    image.save("./data/{}_{}.jpg".format(prompt, i))
                except RateLimitError:
                    time.sleep(60)
                    image = get_image(prompt)
                    image.save("./data/{}_{}.jpg".format(prompt, i))
                time.sleep(3)
