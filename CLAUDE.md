# Adaptive Learning KT - Project Context

## 目的
EdNet-KT1データセットを用いたKnowledge Tracing系モデルの比較実装。
ポートフォリオ用プロジェクト (spin-glass/adaptive-learning-kt)。

## ディレクトリの役割
- `notebooks/*.qmd` - 対話的EDA・分析 (Positronで手動実行が原則)
- `src/` - 再利用可能なモジュール (Claude Codeが自由に編集可)
- `tests/` - pytestによる単体テスト
- `data/raw/` - 生データ (git管理外)
- `_output/` - Quartoレンダー結果 (gitignore)

## Claude Codeへのルール
- `notebooks/*.qmd` は明示的な指示がない限り編集しない
- `src/` のコードは型ヒント必須、pytestでテスト可能な形で書く
- 依存追加は `uv add <pkg>` を使う (pipやpoetry禁止)
- Python 3.11前提
- PyTorchコードは `device` パラメータを受け取れる形に

## 主要ライブラリ
- pyKT: KTモデルの横断比較ツールキット
- pyBKT: Bayesian Knowledge Tracingベースライン
- py-irt: IRT (Item Response Theory) ベースライン
- polars: 大規模データは pandas より polars を優先
- mlflow: 実験管理

## パイプライン
1. `notebooks/01-eda.qmd` - EdNet-KT1のEDA
2. `notebooks/02-baseline.qmd` - IRT + BKT
3. `notebooks/03-dkt.qmd` - DKT/SAKT/SimpleKT比較
4. `notebooks/04-policy.qmd` - 出題戦略シミュレーション

## スタイル
- コメントは日本語可、コード内識別子は英語
- docstringはNumPy形式
