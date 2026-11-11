from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt


def load_faces():
    data = fetch_olivetti_faces()
    faces = data.data
    labels = data.target

    return faces, labels


def show_person(faces, labels, person, filename):
    rows = faces[labels == person]
    plt.figure(figsize=(10, 2))
    for i in range(len(rows)):
        plt.subplot(1, len(rows), i + 1)
        plt.imshow(rows[i].reshape(64, 64), cmap="grey")
        plt.axis("off")

    plt.savefig(filename)


faces, labels = load_faces()
print(f"faces shape: {faces.shape}")
print(f"labels shape: {labels.shape}")
print(f"first 10 labels: {labels[:10]}")
print(f"pixel min and max: {faces.min()}, {faces.max()}")
show_person(faces, labels, 0, "person0.png")
