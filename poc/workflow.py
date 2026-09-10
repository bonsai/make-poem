"""PoC workflow: rhyme-finder → LangGraph → LangChain → LangSmith."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class PoemState(TypedDict, total=False):
    seed: str
    context: dict[str, Any]
    candidates: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    trace: dict[str, Any]


CRITERIA = {
    "phonetic": "deterministic",
    "semantic": "deterministic",
    "context": "rule",
    "rhythm": "deterministic",
    "novelty": "deterministic",
    "style": "llm",
}


def rhyme_candidates(state: PoemState) -> PoemState:
    """Adapter boundary for rhyme-finder; fixture is used for the first PoC."""
    fixtures = {
        "日暮里": [
            {"word": "しっぽり", "reading": "しっぽり", "phonetic_score": 0.92},
            {"word": "にっこり", "reading": "にっこり", "phonetic_score": 0.88},
        ]
    }
    return {"candidates": fixtures.get(state["seed"], [])}


def score_candidates(state: PoemState) -> PoemState:
    context = state.get("context", {})
    mood = set(context.get("mood", []))
    scene = set(context.get("scene", []))
    ranked = []
    for candidate in state.get("candidates", []):
        word = candidate["word"]
        semantic = 0.5
        context_score = 0.5
        if word == "しっぽり":
            semantic = 0.95 if {"大人", "しっとり"} & mood else 0.65
            context_score = 0.95 if {"夜", "酒", "静けさ"} & scene else 0.55
        elif word == "にっこり":
            semantic = 0.95 if {"笑顔", "楽しさ"} & mood else 0.65
            context_score = 0.95 if {"出会い", "笑顔"} & scene else 0.55
        vector = {
            "phonetic": candidate["phonetic_score"],
            "semantic": semantic,
            "context": context_score,
            "rhythm": 0.90,
            "novelty": 0.80,
        }
        ranked.append({**candidate, "score_vector": vector,
                       "score": round(sum(vector.values()) / len(vector), 4)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return {"ranked": ranked}


def llm_delegation(state: PoemState) -> PoemState:
    """Only criteria configured as ``llm`` are delegated to LangChain."""
    delegated = [k for k, v in CRITERIA.items() if v == "llm"]
    return {"trace": {"delegated_criteria": delegated}}


def build_workflow():
    graph = StateGraph(PoemState)
    graph.add_node("rhyme_finder", rhyme_candidates)
    graph.add_node("score", score_candidates)
    graph.add_node("llm", llm_delegation)
    graph.add_edge(START, "rhyme_finder")
    graph.add_edge("rhyme_finder", "score")
    graph.add_edge("score", "llm")
    graph.add_edge("llm", END)
    return graph.compile()


if __name__ == "__main__":
    result = build_workflow().invoke({
        "seed": "日暮里",
        "context": {"scene": ["夜", "酒", "静けさ"], "mood": ["大人", "しっとり"]},
    })
    print(result["ranked"])
    print(result["trace"])
