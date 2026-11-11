import io
import requests
from PIL import Image
from split_data import split_faces
from load_data import load_faces

URL = "http://127.0.0.1:8000/verify"


def send(face, person_id):
    pixels = (face.reshape(64, 64) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    buffer.seek(0)
    files = {"image": ("face.png", buffer, "image/png")}
    response = requests.post(URL, data={"person_id": person_id}, files=files)
    return response.json()


faces, labels = load_faces()
train_faces, train_labels, test_faces, test_labels = split_faces(faces, labels)

false_rejects = 0
false_accepts = 0

for face, true_label in zip(test_faces, test_labels):
    result = send(face, int(true_label))
    if not result["match"]:
        false_rejects += 1
    wrong_id = (int(true_label) + 1) % 40
    result = send(face, wrong_id)
    if result["match"]:
        false_accepts += 1

print("genuine attempts:", len(test_faces), "false rejects:", false_rejects)
print("impostor attempts:", len(test_faces), "false accepts:", false_accepts)
