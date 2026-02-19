from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

index = None
documentos = []

def construir_indice(lista_textos):
    global index, documentos

    documentos = lista_textos
    embeddings = model.encode(lista_textos)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

def buscar_contexto(pregunta, k=3):
    if index is None:
        return []

    embedding = model.encode([pregunta])
    D, I = index.search(np.array(embedding), k)
    return [documentos[i] for i in I[0]]

