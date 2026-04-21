# adaptive-learning-kt

EdNet-KT1 データセットを用いた Knowledge Tracing 系モデルの比較実装。

## Setup

```bash
uv sync
```

Python 3.11 前提。依存追加は `uv add <pkg>`。

## Data pipeline

### 1. 生データをダウンロード

```bash
make download
# or:
uv run python -m src.data.download --dest data/raw
```

`data/raw/KT1/` 以下に約780k件のユーザー別 CSV が展開される。

### 2. pyKT 標準前処理を適用 (オプション)

[pyKT-toolkit](https://github.com/pykt-team/pykt-toolkit) を**参照用**として別ディレクトリに clone し、前処理のみ実行して CSV を取り込む。pyKT 本体はこのプロジェクトの依存には含めない。

```bash
make ref-pykt-clone           # ~/reference-pykt に clone
make ref-pykt-venv            # そこに Python 3.9 の独立 venv を作成
make ref-pykt-preprocess      # data/raw/KT1/ をリンクして pyKT の前処理実行
make ref-pykt-import-processed # 処理済み CSV を data/processed/ednet/ にコピー
```

`Makefile` の各ターゲットは冪等で、失敗時に途中から再実行可能。

## Run notebooks

Positron で `notebooks/*.qmd` を開き、セル単位で実行。

全体レンダー:
```bash
make test        # pytest
uv run quarto render
```

## Structure

- `notebooks/` - EDA / 実験ノート (.qmd、Claude Code は編集しない)
- `src/` - 再利用可能モジュール (モデル / データ / 学習 / 評価)
  - `src/data/` - ダウンロード・ローダ
  - `src/models/` - IRT / BKT / DKT / SAKT / SimpleKT
  - `src/training/` - PyTorch Lightning モジュール
  - `src/eval/` - AUC / Early & Late fusion
- `tests/` - pytest 単体テスト
- `data/raw/` - 生データ (gitignore)
- `data/processed/` - pyKT 出力の取り込み先 (gitignore)
- `mlruns/` - MLflow ログ (gitignore)
