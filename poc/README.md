# Python PoC

`make-poem` の **Semantics × Phonetics** を、LangGraph で最小実装する PoC。

## Flow

```text
seed
  ↓
phonetics
  ↓
semantics + context
  ↓
poetic ranking
  ↓
compose
  ↓
poem
```

LangSmith は環境変数による LangChain tracing を利用する。API key がない状態でも PoC のローカル実行はできる。

## Run

```bash
cd poc
uv run python main.py
```

または通常の venv で `pip install -e .` 相当の環境を作って `python main.py`。

## LangSmith

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=make-poem-poc
```

PowerShell:

```powershell
$env:LANGSMITH_TRACING="true"
$env:LANGSMITH_API_KEY="..."
$env:LANGSMITH_PROJECT="make-poem-poc"
```

## Canonical example

```text
日暮里
 ├─ しっぽり
 └─ にっこり
```

`夜 + 酒 + 静けさ` の context では `しっぽり` を上位にする。

## Next

- `phonetics` → `bonsai/rhyme-finder` contract
- semantic/context → richer graph/context retrieval
- LangGraph → parallel candidate generation and ranking
- LangSmith → trace/evaluation dataset
- LLM composition → final lyric generation
