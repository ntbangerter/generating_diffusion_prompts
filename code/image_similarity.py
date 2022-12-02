import clip
from sklearn.preprocessing import normalize

class ImageSimilarity:
    def __init__(self):
        self.image_embedder, self.image_embedder_preprocess = clip.load("ViT-B/32", device="cuda")
        
    def get_image_embedding(self, image):
        preprocessed_image = self.image_embedder_preprocess(image).unsqueeze(0).to("cuda")
        embedding = self.image_embedder.encode_image(preprocessed_image)
        return normalize(embedding.cpu().detach().numpy())[0]
    
    def get_image_similarity_score(self, image1, image2):
        embedding1 = self.get_image_embedding(image1)
        embedding2 = self.get_image_embedding(image2)
        return embedding1.dot(embedding2)
