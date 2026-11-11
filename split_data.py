import numpy as np
from load_data import load_faces


def split_faces(faces, labels, train_count=7):
    train_idx = []
    test_idx = []

    for person in np.unique(labels):
        idx = np.where(labels == person)[0]
        train_idx.extend(idx[:train_count])
        test_idx.extend(idx[train_count:])

    train_faces = faces[train_idx]
    train_labels = labels[train_idx]
    test_faces = faces[test_idx]
    test_labels = labels[test_idx]
    return train_faces, train_labels, test_faces, test_labels


faces, labels = load_faces()

train_faces, train_labels, test_faces, test_labels = split_faces(faces, labels)

print(f"Train faces: {train_faces.shape}")
print(f"Test faces: {test_faces.shape}")
print(f"People in train: {len(np.unique(train_labels))}")
print(f"People in test: {len(np.unique(test_labels))}")
