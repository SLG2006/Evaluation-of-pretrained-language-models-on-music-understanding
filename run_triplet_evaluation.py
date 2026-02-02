from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load a pretrained language model
model = SentenceTransformer("all-mpnet-base-v2")

# Load triplets
triplets = pd.read_csv("audioset_triplets_genre.csv")

correct = 0
total = len(triplets)

for _, row in tqdm(triplets.iterrows(), total=total):
    anchor = row["anchor"]
    positive = row["positive"]
    negative = row["negative"]

    emb_anchor = model.encode(anchor)
    emb_positive = model.encode(positive)
    emb_negative = model.encode(negative)

    sim_pos = cosine_similarity([emb_anchor], [emb_positive])[0][0]
    sim_neg = cosine_similarity([emb_anchor], [emb_negative])[0][0]

    if sim_pos > sim_neg:
        correct += 1

accuracy = correct / total
print(f"Triplet accuracy: {accuracy:.4f}")