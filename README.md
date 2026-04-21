# adaptive-learning-kt

EdNet-KT1 データセットを用いた Knowledge Tracing 系モデルの比較実装。

## Setup

```bash
uv sync
```

## Run notebooks

Positron で `notebooks/*.qmd` を開き、セル単位で実行。

全体レンダーは:
```bash
uv run quarto render
```

## Structure

- `notebooks/` - EDA / 実験ノート (.qmd)
- `src/` - 再利用可能モジュール
- `tests/` - 単体テスト

