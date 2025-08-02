from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_sentences(sentences: list):
    return model.encode(sentences, convert_to_numpy=True)
