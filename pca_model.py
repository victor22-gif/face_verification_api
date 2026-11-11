import numpy as np
from split_data import split_faces
from load_data import load_faces


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


faces, labels = load_faces()
train_faces, test_faces, train_labels, test_labels = split_faces(faces, labels)

mean, components = train_pca(train_faces, k=50)
print(f"mean face shape: {mean.shape}")
print(f"components shape: {components.shape}")

fingerprint = embed(train_faces[0], mean, components)
print(f"fingerprint shape: {fingerprint.shape}")

same = cosine_similarity(
    embed(train_faces[0], mean, components), embed(train_faces[1], mean, components)
)
different = cosine_similarity(
    embed(train_faces[0], mean, components), embed(train_faces[7], mean, components)
)
print(f"same person's score: {round(same, 3)}")
print(f"different person's score: {round(different, 3)}")
