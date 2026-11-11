import numpy as np
import pandas as pd
from split_data import split_faces
from load_data import load_faces
from pca_model import train_pca, embed

faces, labels = load_faces()
train_faces, train_labels, test_faces, test_labels = split_faces(faces, labels)

mean, components = train_pca(train_faces, k=50)
np.save("mean.npy", mean)
np.save("components.npy", components)

rows = []
for face, label in zip(train_faces, train_labels):
    fp = embed(face, mean, components)
    rows.append({"person_id": int(label), "fingerprint": fp.tolist()})

registry = pd.DataFrame(rows)
registry.to_pickle("registry.pkl")
print("saved registry with", len(registry), "rows")
print(registry.head())
