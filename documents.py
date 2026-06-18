"""
documents.py
-------------
Loads documents as a multi-document knowledge base.
"""

from pathlib import Path

DATA_DIR = Path("data")

def load_documents():

    documents = []

    for file_path in sorted(DATA_DIR.glob("*.md")):
        documents.append(
            {
                "title": file_path.stem,   
                "text": file_path.read_text(encoding="utf-8")
            }
        )

    return documents


DOCUMENTS = load_documents()
# print(f"Loaded {len(DOCUMENTS)} documents from {DATA_DIR}/")
# print(DOCUMENTS)