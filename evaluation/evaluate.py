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
llm_judge = rag.ChatOllama(model="mistral")
 
vectorstore = rag.Chroma(
    collection_name=rag.COLLECTION_NAME,
    persist_directory=str(rag.CHROMA_DIR),
    embedding_function=rag.OllamaEmbeddings(model=rag.OLLAMA_EMBED_MODEL)
)


def rag_pipeline(inputs: dict) -> dict:
    question = inputs["question"]
    docs = vectorstore.similarity_search(question, k=6)
    prompt = rag.build_augmented_prompt(question, docs)
    answer = llm.invoke(prompt)
    return {
        "answer": answer.content,
        "contexts": "\n\n".join(d.page_content for d in docs),
    }


def answer_correctness(outputs: dict, reference_outputs: dict) -> dict:
    prompt = f"""Does this answer correctly address the answer correctness?
        Answer: {outputs['answer']}
        Ground truth: {reference_outputs['ground_truth']}
        Reply only with a word between [excellent, good, insufficient]"""
    judgement = llm_judge.invoke(prompt).content.strip().lower()
    return {"score": {"excellent": 2, "good": 1}.get(judgement, 0), "value": judgement}


def context_groundedness(outputs: dict, reference_outputs: dict) -> dict:
    prompt = f"""Does this answer correctly address the ground truth?
        Answer: {outputs['answer']}
        Ground truth: {reference_outputs['ground_truth']}
        Reply only with a word between [excellent, good, insufficient]"""
    judgement = llm_judge.invoke(prompt).content.strip().lower()
    return {"score": {"excellent": 2, "good": 1}.get(judgement, 0), "value": judgement}


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
    print("======================== Starting RAG Evaluation ========================")
    setup_dataset()
    results = evaluate(
        rag_pipeline,
        data=DATASET_NAME,
        evaluators=[answer_correctness, context_groundedness],
        experiment_prefix="rag-eval",
    )
    print("========================== Evaluation Finished ==========================")