import argparse
import pypdf
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths and Chroma collection settings (relative to this script).
BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "documents" / "Alice's Adventures in Wonderland by Lewis Carroll.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "alice_in_wonderland"

# Ollama models — pull them first: `ollama pull nomic-embed-text` and `ollama pull llama3.2`
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "llama3.2"


def parse_args() -> argparse.Namespace:
    """Read the user question and optional flags from the command line."""
    parser = argparse.ArgumentParser(
        description="Terminal RAG app with LangChain, Chroma, and Ollama."
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="+",
        help="Question for the RAG system. Example: python rag_app.py Who is the White Rabbit?",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="How many relevant chunks to retrieve from Chroma.",
    )
    parser.add_argument(
        "--o",
        type=int,
        default=100,
        help="The overlap size of the chunks text splitter",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=1000,
        help="The size of the chunks text splitter",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the vector store from the PDF (required after changing embed model).",
    )
    return parser.parse_args()


def load_and_chunk_pdf(pdf_path: Path, args: argparse.Namespace) -> list[Document]:
    """
    Load the PDF page by page, then split into overlapping text chunks.
    Smaller chunks improve retrieval precision; overlap avoids cutting sentences in half.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"page": i, "source": str(pdf_path)},
        )
        for i, page in enumerate(reader.pages)
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.c,
        chunk_overlap=args.o,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(pages)


def ensure_vector_store(args: argparse.Namespace, embeddings: OllamaEmbeddings, rebuild: bool = False) -> Chroma:
    """
    Open or create the Chroma vector database.
    On first run (empty collection), embed all PDF chunks via Ollama and store them.
    """
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    current_count = vector_store._collection.count()
    if rebuild and current_count > 0:
        # Drop existing vectors so we can rebuild.
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        current_count = 0

    if current_count == 0:
        if not PDF_PATH.exists():
            raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

        chunks = load_and_chunk_pdf(PDF_PATH, args)
        vector_store.add_documents(chunks)

    return vector_store


def build_augmented_prompt(user_prompt: str, retrieved_docs) -> str:
    """
    RAG "augmentation": prepend retrieved book chunks to the user question
    so the LLM answers from the document.
    """
    context = "\n\n".join(
        f"[Source {idx} | page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}"
        for idx, doc in enumerate(retrieved_docs, start=1)
    )

    return (
        "You are a helpful assistant.\n"
        "Use the context below to answer the user question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"User question:\n{user_prompt}"
    )


def main() -> None:
    args = parse_args()
    user_prompt = " ".join(args.prompt).strip()

    # Local embeddings via Ollama.
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    vector_store = ensure_vector_store(args, embeddings, rebuild=args.rebuild)

    # Retrieve chunks whose vectors are closest to the question embedding.
    retrieved_docs = vector_store.similarity_search(user_prompt, k=args.k)
    augmented_prompt = build_augmented_prompt(user_prompt, retrieved_docs)

    # Local chat model via Ollama.
    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.2)
    response = llm.invoke(augmented_prompt)

    print("\n=== User Prompt ===")
    print(user_prompt)
    print("\n=== Retrieved Context Snippets ===")
    print("\n".join(
        f"{idx}. page={doc.metadata.get('page', 'unknown')} | {" ".join(doc.page_content.split())[:220]}..."
        for idx, doc in enumerate(retrieved_docs, start=1)
    ))
    print("\n=== LLM Response ===")
    print(response.content)


if __name__ == "__main__":
    main()
