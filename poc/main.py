"""Minimal LangGraph PoC for Semantics × Phonetics lyric generation."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PoemState(TypedDict, total=False):
    seed: str
    context: dict
    candidates: list[dict]
    ranked: list[dict]
    poem: str


PHONETIC_FIXTURES = {
    "日暮里": [
        {"word": "しっぽり", "reading": "しっぽり", "phonetic_score": 0.92},
        {"word": "にっこり", "reading": "にっこり", "phonetic_score": 0.88},
    ]
}


def phonetics(state: PoemState) -> PoemState:
    """Generate candidates; replace with rhyme-finder integration later."""
    return {"candidates": PHONETIC_FIXTURES.get(state["seed"], [])}


def semantics(state: PoemState) -> PoemState:
    """Score meaning/context independently from phonetic similarity."""
    context = state.get("context", {})
    mood = set(context.get("mood", []))
    scene = set(context.get("scene", []))
    ranked = []

    for candidate in state.get("candidates", []):
        word = candidate["word"]
        semantic_score = 0.5
        context_score = 0.5
        if word == "しっぽり":
            semantic_score = 0.95 if {"大人", "しっとり"} & mood else 0.65
            context_score = 0.95 if {"夜", "酒", "静けさ"} & scene else 0.55
        elif word == "にっこり":
            semantic_score = 0.95 if {"笑顔", "楽しさ"} & mood else 0.65
            context_score = 0.95 if {"出会い", "笑顔"} & scene else 0.55

        score = (
            candidate["phonetic_score"] * 0.4
            + semantic_score * 0.3
            + context_score * 0.3
        )
        ranked.append({
            **candidate,
            "semantic_score": semantic_score,
            "context_score": context_score,
            "poetic_score": round(score, 4),
        })

    ranked.sort(key=lambda x: x["poetic_score"], reverse=True)
    return {"ranked": ranked}


def compose(state: PoemState) -> PoemState:
    top = state["ranked"][0]
    seed = state["seed"]
    return {
        "poem": (
            f"{seed}、{top['word']}。\n"
            "音が似ているだけじゃない、\n"
            "いまの景色に似合う言葉。"
        )
    }


def build_graph():
    graph = StateGraph(PoemState)
    graph.add_node("phonetics", phonetics)
    graph.add_node("semantics", semantics)
    graph.add_node("compose", compose)
    graph.add_edge(START, "phonetics")
    graph.add_edge("phonetics", "semantics")
    graph.add_edge("semantics", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "seed": "日暮里",
        "context": {
            "scene": ["夜", "酒", "静けさ"],
            "mood": ["大人", "しっとり"],
        },
    })
    print("=== ranked candidates ===")
    for candidate in result["ranked"]:
        print(candidate)
    print("\n=== poem ===")
    print(result["poem"])
