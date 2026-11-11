# Face Verification API

A face verification API built with numpy, pandas and FastAPI.
Built as a learning project, using the Eigenfaces method (PCA).

## How it works
1. Faces (64x64 grayscale) are turned into 4096 numbers each.
2. PCA (numpy SVD) finds the 50 main directions faces vary in.
3. Each face becomes a 50-number "fingerprint".
4. Registered fingerprints are stored in a pandas DataFrame.
5. `/verify` compares an uploaded face to a claimed person's
   registered fingerprints using cosine similarity.
   Score >= 0.7 means a match.

## Results (Olivetti faces, 7 train / 3 test per person)
- Identification accuracy: 89%
- At threshold 0.7: about 3% false accepts, 10% false rejects

## Limitations
- Eigenfaces is sensitive to lighting and pose.
- Error rates are far too high for real banking use.
  Production systems use deep learning models.
- Trained on a small public dataset (40 people).

## Run it
pip install -r requirements.txt
python save_model.py
uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
