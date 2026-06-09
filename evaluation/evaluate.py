from dotenv import load_dotenv
from langsmith import Client, evaluate
from test_dataset import test_cases
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import rag_app as rag
from environment import set_env
load_dotenv()
set_env()

DATASET_NAME = "rag-eval"

client = Client()
llm = rag.ChatOllama(model=rag.OLLAMA_CHAT_MODEL)
vectorstore = rag.Chroma(
    collection_name=rag.COLLECTION_NAME,
    persist_directory=str(rag.CHROMA_DIR),
    embedding_function=rag.OllamaEmbeddings(model=rag.OLLAMA_EMBED_MODEL)
)


def rag_pipeline(inputs: dict) -> dict:
    question = inputs["question"]
    docs = vectorstore.similarity_search(question, k=4)
    prompt = rag.build_augmented_prompt(question, docs)
    answer = llm.invoke(prompt)
    return {
        "answer": answer.content,
        "contexts": "\n\n".join(d.page_content for d in docs),
    }


from rouge_score import rouge_scorer as rouge
def answer_correctness(outputs: dict, reference_outputs: dict) -> dict:
    scorer = rouge.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(
        reference_outputs["ground_truth"].lower(),
        outputs["answer"].lower()
    )
    return {"key": "answer_correctness", "score": round(score["rougeL"].fmeasure, 4)}


def context_groundedness(outputs: dict) -> dict:
    answer = outputs["answer"].lower()
    contexts = outputs["contexts"].lower()
    # extract key nouns from answer and check if they appear in context
    words = [w for w in answer.split() if len(w) > 3]
    matches = sum(1 for w in words if w in contexts)
    score = round(matches / len(words), 4) if words else 0.0
    return {"key": "context_groundedness", "score": score}


def setup_dataset():
    existing = next((d for d in client.list_datasets() if d.name == DATASET_NAME), None)
    
    if existing:
        client.delete_dataset(dataset_id=existing.id)

    client.create_dataset(DATASET_NAME)
    client.create_examples(
        inputs=[{"question": tc["question"]} for tc in test_cases],
        outputs=[{"ground_truth": tc["ground_truth"]} for tc in test_cases],
        dataset_name=DATASET_NAME,
    )


if __name__ == "__main__":
    print("=== Starting RAG Evaluation ===")
    setup_dataset()
    results = evaluate(
        rag_pipeline,
        data=DATASET_NAME,
        evaluators=[answer_correctness, context_groundedness],
        experiment_prefix="rag-eval",
    )