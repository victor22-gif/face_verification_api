import numpy as np


def train_pca(faces, k=50):
    mean = faces.mean(axis=0)
    centered = faces - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]
    return mean, components


def embed(face, mean, components):
    return components @ (face - mean)


def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
