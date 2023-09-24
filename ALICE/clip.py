from sentence_transformers import SentenceTransformer, util
from PIL import Image
from typing import List
import numpy as np

from alice.config import logger
from alice.utils.image import crop_image


class CLIP():

    def __init__(self):
        # Load the OpenAI CLIP Model
        self.model = SentenceTransformer('clip-ViT-B-32')

    def _preprocess_image(self, image: np.array):  
        image_size = 300
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Crop image, so we get more of the label, less of the surround which 
        # confuses the CLIP model
        image = crop_image(image)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.convert('L').resize((image_size, image_size), Image.LANCZOS)
        return pil_image
        

    def get_duplicates(self, images: List[np.array], threshold=0.85):
        # To Pil images
        # pre_processed = [self._preprocess_image(img) for img in images]
        images = [self._preprocess_image(img) for img in images]
        encoded_image = self.model.encode(images)
        processed_images = util.paraphrase_mining_embeddings(encoded_image)
        duplicates = [image for image in processed_images if image[0] >= threshold]
        return duplicates

    def cluster(self, images: List[np.array]):
        duplicates = self.get_duplicates(images)
        clusters = []
        for duplicate in duplicates:
            ids = set(duplicate[1:])
            existing = [i for i, cluster in enumerate(clusters) if ids.intersection(cluster)]        
            if not existing:
                clusters.append(ids)
            elif len(existing) == 1:
                i = existing[0]
                # If both IDS already exist (already added as the other end of the duplicate pair) -> ignore
                if not ids.difference(clusters[i]): continue
                if len(clusters[i]) < 4:
                    clusters[i].update(ids)
                else:
                    logger.info('Existing cluster already has maximum number of items')
            else:
                # We just ignore duplcates found in multiple clusters
                # Best duplcates are found first & if an image belongs
                # in another cluster, it will be in another duplcate
                logger.info('Duplicates found in mutliple clusters - ignoring')
        return clusters