import numpy as np
from split_data import split_faces
from load_data import load_faces
from pca_model import train_pca, embed, cosine_similarity

faces, labels = load_faces()
train_faces, train_labels, test_faces, test_labels = split_faces(faces, labels)


def person_scores(k=50):
    mean, components = train_pca(train_faces, k=k)
    train_prints = [embed(f, mean, components) for f in train_faces]
    genuine = []  # test face vs its own person
    impostor = []  # test face vs other people
    for face, true_label in zip(test_faces, test_labels):
        fp = embed(face, mean, components)
        scores = np.array([cosine_similarity(fp, t) for t in train_prints])
        for person in np.unique(train_labels):
            best = scores[train_labels == person].max()
            if person == true_label:
                genuine.append(best)
            else:
                impostor.append(best)
    return np.array(genuine), np.array(impostor)


genuine, impostor = person_scores()
print("genuine scores: average", round(genuine.mean(), 3))
print("impostor scores: average", round(impostor.mean(), 3))

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    frr = (genuine < threshold).mean()
    far = (impostor >= threshold).mean()
    print("threshold", threshold, "FAR:", round(far, 3), "FRR:", round(frr, 3))
