# Team 本田（仮） 2025 LLM Competition

## ドキュメント

- [Notion](https://www.notion.so/Team-22a9dd6b4cc28015850dc9a4d2314393)

## 環境構築

それぞれのディレクトリの README.md をご覧ください。

## Team 間の IO インターフェース

それぞれ HuggingFace を経由することを想定しています。

- データ(HLE/DNA) <-> 学習: HuggingFace Datasets
  - データチームが push した Dataset を、学習チームが pull して学習に使用する
- 学習 <-> 推論: HuggingFace Models
  - 学習チームが push したモデルを、推論チームが pull して推論に使用する

## License

TBD
