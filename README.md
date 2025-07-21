# Team 本田（仮） 2025 LLM Competition

## ドキュメント

- [Notion](https://www.notion.so/Team-22a9dd6b4cc28015850dc9a4d2314393)

## 環境構築

```bash
$ make build
```

## 開発環境

以下のコマンドで、コンテナの中に入ることができる。コンテナの中で、uv や python3 等のコマンドを実行することができる。

```bash
$ make run
```

上記の `make run` か、もしくは `make up` でコンテナを裏側で立ち上げ続けることができる。VSCode 互換のエディタであれば、`Attach to Running Container...` で `/llm2025compet-app-1` を選ぶことで、コンテナ内で作業することができる。

## パッケージを追加する時

必ず、コンテナ内で uv を実行するようにする。

```bash
$ make run
$ uv add numpy
```

## Team 間の IO インターフェース

それぞれ HuggingFace を経由することを想定しています。

- データ(HLE/DNA) <-> 学習: HuggingFace Datasets
  - データチームが push した Dataset を、学習チームが pull して学習に使用する
- 学習 <-> 推論: HuggingFace Models
  - 学習チームが push したモデルを、推論チームが pull して推論に使用する

## License

TBD
