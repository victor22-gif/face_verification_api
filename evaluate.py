import numpy as np
from load_data import load_faces
from split_data import split_faces
from pca_model import train_pca, embed, cosine_similarity

faces, labels = load_faces()
train_faces, train_labels, test_faces, test_labels = split_faces(faces, labels)


def accuracy_for_k(k):
    mean, components = train_pca(train_faces, k=k)
    train_prints = [embed(f, mean, components) for f in train_faces]
    correct = 0
    for face, true_label in zip(test_faces, test_labels):
        fp = embed(face, mean, components)
        scores = [cosine_similarity(fp, t) for t in train_prints]
        best = np.argmax(scores)
        if train_labels[best] == true_label:
            correct += 1
    return correct / len(test_faces)


for k in [10, 30, 50, 100]:
    print(f"k={k}, accuracy: {round(accuracy_for_k(k), 3)}")
