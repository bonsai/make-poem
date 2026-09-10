"""Minimal LangGraph PoC for Semantics × Phonetics lyric generation.

The graph deliberately keeps phonetic candidate generation separate from
semantic/contextual selection. LangSmith tracing is enabled automatically by
LangChain when LANGSMITH_TRACING=true and the usual LANGSMITH_* environment
variables are configured.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PoemState(TypedDict, total=False):
    seed: str
    context: dict
    candidates: list[dict]
    ranked: list[dict]
    poem: str


# Tiny fixture for the first PoC. Later this node will call rhyme-finder.
PHONETIC_FIXTURES = {
    "日暮里": [
        {"word": "しっぽり", "reading": "しっぽり", "phonetic_score": 0.92},
        {"word": "にっこり", "reading": "にっこり", "phonetic_score": 0.88},
    ]
}


def phonetics(state: PoemState) -> PoemState:
    candidates = PHONETIC_FIXTURES.get(state["seed"], [])
    return {"candidates": candidates}


def semantics(state: PoemState) -> PoemState:
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
    poem = f"{seed}、{top['word']}。\n音が似ているだけじゃない、\nいまの景色に似合う言葉。"
    return {"poem": poem}


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
