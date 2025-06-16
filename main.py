import os
import requests
import pdfplumber
import json
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)
metadata = []


max_lines = 20
overlap = 5
max_chars = 1000
char_overlap = 100
top_k = 3
meta_path = "filemeta.json"

file_mod_data = {}

if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
    with open(meta_path, "r", encoding="utf-8") as f:
        try:
            file_mod_data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: file_meta.json is invalid, resetting.")
            file_mod_data = {}

mypath = os.path.join(os.getcwd(), "files")
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

def chunk_paragraphs(text, max_chars=1000, overlap=100):
    paragraphs = text.split("\n\n")
    chunks = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < max_chars:
            buffer += "\n\n" + para
        else:
            chunks.append(buffer.strip())
            buffer = para[-overlap:] if overlap else ""
    if buffer:
        chunks.append(buffer.strip())
    return chunks


for file in files:
    full_path = os.path.join(mypath, file)
    last_modify = os.path.getmtime(full_path)
    if file in file_mod_data and file_mod_data[file]  == last_modify:
        continue


    print(f"📄 Processing: {file}")
    try:
        if file.lower().endswith(".pdf"):
            with pdfplumber.open(full_path) as p:
                text = "\n".join([page.extract_text() or "" for page in p.pages])
            chunks = [chunk.strip() for chunk in text.split("\n") if len(chunk.strip()) > 20]

        elif file.lower().endswith(".docx"):
            doc = Document(full_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            chunks = chunk_paragraphs(text, max_chars, char_overlap)

        elif file.lower().endswith(".txt"):
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_paragraphs(text, max_chars, char_overlap)

        elif file.lower().endswith(".py"):
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
            chunks = ["\n".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines - overlap)]

        else:
            continue  

        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        index.add(np.array(embeddings))
        file_mod_data[file] = last_modify
        for i, chunk in enumerate(chunks):
            metadata.append({"file": file, "chunk": chunk})

        faiss.write_index(index,"my_index.index")
        with open("metadata.json","w",encoding='utf-8') as j:
            json.dump(metadata,j,indent=2)


    except Exception as e:
        print(f" Error processing {file}: {e}")



with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(file_mod_data, f, indent=2)

while True:
    query = input("\n🔍 Ask a question (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break

    query_vector = model.encode(query, convert_to_numpy=True)
    D, I = index.search(np.array([query_vector]), top_k)

    context_chunks = [metadata[i]["chunk"] for i in I[0]]
    context_text = "\n\n".join(context_chunks)

    prompt = f"""You are an assistant. Use the following context to answer the question.

Context:
{context_text}

Question:
{query}

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt},
            stream=True
        )

        print("\nAnswer:\n")
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode("utf-8"))
                    full_response += json_data.get("response", "")
                except json.JSONDecodeError as e:
                    print(f" JSON error: {e}")

        print(full_response.strip())

    except Exception as e:
        print(f" Request error: {e}")
