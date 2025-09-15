from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# List of markdown files (add more as needed)
files = ["document.md"]  # later you can add more files here

# Define markdown headers we care about
headers = [
    ("#", "heading1"),
    ("##", "heading2"),
    ("###", "heading3"),
    ("####", "heading4"),
    ("#####", "heading5"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

all_docs = []

# Process each file
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    
    docs = splitter.split_text(markdown_text)

    # Enrich metadata with context path
    for doc in docs:
        heading_keys = ["heading1", "heading2", "heading3", "heading4", "heading5"]
        context_path = []
        for key in heading_keys:
            if key in doc.metadata and doc.metadata[key]:
                title = doc.metadata[key]
                if title.startswith("**") and title.endswith("**"):
                    title = title[2:-2].strip()
                context_path.append(title)
        doc.metadata["context_path"] = " > ".join(context_path)
        doc.metadata["source_file"] = file  # 🔑 keep track of which file this chunk came from
    
    all_docs.extend(docs)  # add this file's docs to global list

# Inspect first few chunks from all files
for i, doc in enumerate(all_docs[:5]):
    print(f"--- Chunk {i+1} ---")
    print("Source file:", doc.metadata.get("source_file"))
    print("Context path:", doc.metadata.get("context_path", "No context"))
    print("Content preview:", doc.page_content[:100], "...\n")

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Save to Chroma
db = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    collection_name="policy_docs_hf",
    persist_directory="./chroma_db"   # path to save DB
)

db.persist()
