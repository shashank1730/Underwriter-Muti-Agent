from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma




# Define markdown headers we care about
headers = [
    ("#", "heading1"),
    ("##", "heading2"),
    ("###", "heading3"),
    ("####", "heading4"),
    ("#####", "heading5"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

with open("document.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

docs = splitter.split_text(markdown_text)

# Inspect each chunk with metadata
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print("Metadata:", doc.metadata)



# Assuming `docs` is the list of Document objects from MarkdownHeaderTextSplitter
for doc in docs:
    # Collect all heading metadata in order
    heading_keys = ["heading1", "heading2", "heading3", "heading4", "heading5"]
    context_path = []
    for key in heading_keys:
        if key in doc.metadata and doc.metadata[key]:
            # Remove bold if present
            title = doc.metadata[key]
            if title.startswith("**") and title.endswith("**"):
                title = title[2:-2].strip()
            context_path.append(title)
    
    # Add a new metadata field with the full context path
    doc.metadata["context_path"] = " > ".join(context_path)

# Check the first few chunks
for i, doc in enumerate(docs[:5]):  # first 5 chunks
    print(f"--- Chunk {i+1} ---")
    print("Context path:", doc.metadata.get("context_path", "No context"))
    print("Content preview:", doc.page_content[:20], "...\n")  # first 200 chars


# Load a HuggingFace sentence-transformer model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="policy_docs_hf",
    persist_directory="./chroma_db"   # path to save the DB
)

db.persist() 