"""PoC workflow: rhyme-finder → SEM (Owlready2) → PHO → ranking."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from owlready2 import Thing, get_ontology


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


def build_semantic_ontology():
    """Build the minimal SEM ontology used by the PoC."""
    onto = get_ontology("http://bonsai.local/make-poem-sem.owl")
    with onto:
        class SemanticWord(Thing):
            pass

        class has_mood(Thing >> str):
            pass

        class has_scene(Thing >> str):
            pass

        class has_theme(Thing >> str):
            pass

        shippori = SemanticWord("shippori")
        shippori.has_mood = ["大人", "しっとり"]
        shippori.has_scene = ["夜", "酒", "静けさ"]
        shippori.has_theme = ["夜の酒"]

        nikkori = SemanticWord("nikkori")
        nikkori.has_mood = ["笑顔", "楽しさ"]
        nikkori.has_scene = ["出会い", "笑顔"]
        nikkori.has_theme = ["楽しい時間"]
    return onto


def semantic_score(word: str, context: dict[str, Any], ontology) -> tuple[float, list[str]]:
    """Score a candidate against SEM concepts stored in Owlready2."""
    lookup = {"しっぽり": "shippori", "にっこり": "nikkori"}
    individual = ontology.search_one(iri=f"*#{lookup.get(word, word)}")
    if individual is None:
        return 0.5, []

    requested = set(context.get("mood", [])) | set(context.get("scene", []))
    known = set(getattr(individual, "has_mood", [])) | set(getattr(individual, "has_scene", []))
    matches = sorted(requested & known)
    score = 0.5 if not requested else min(1.0, 0.5 + 0.45 * len(matches) / len(requested))
    return score, matches


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
    ontology = build_semantic_ontology()
    ranked = []
    for candidate in state.get("candidates", []):
        semantic, semantic_matches = semantic_score(candidate["word"], context, ontology)
        vector = {
            "phonetic": candidate["phonetic_score"],
            "semantic": semantic,
            "context": semantic,
            "rhythm": 0.90,
            "novelty": 0.80,
        }
        ranked.append({
            **candidate,
            "score_vector": vector,
            "semantic_matches": semantic_matches,
            "score": round(sum(vector.values()) / len(vector), 4),
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {"ranked": ranked, "trace": {"semantics": "owlready2"}}


def llm_delegation(state: PoemState) -> PoemState:
    """Only criteria configured as ``llm`` are delegated to LangChain."""
    delegated = [key for key, value in CRITERIA.items() if value == "llm"]
    return {"trace": {**state.get("trace", {}), "delegated_criteria": delegated}}


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
    print("=== ranked ===")
    for candidate in result["ranked"]:
        print(candidate)
    print("=== trace ===")
    print(result["trace"])
