from PIL import Image
from split_data import split_faces
from load_data import load_faces

faces, labels = load_faces()
train_faces, train_labels, test_faces, test_labels = split_faces(faces, labels)


def save_face(face, filename):
    pixels = (face.reshape(64, 64) * 255).astype("uint8")
    Image.fromarray(pixels).save(filename)


save_face(test_faces[0], "person0_test.png")
save_face(test_faces[3], "person1_test.png")
