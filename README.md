# Team Neko 2025 LLM Competition

## ドキュメント

- [Notion](https://www.notion.so/Team-22a9dd6b4cc28015850dc9a4d2314393)

## ローカル環境構築

Nvidia GPU を持っていらっしゃらない方は、docker で開発するメリットはあまりないと思われます。docker の中で開発することもできます。

Nvidia GPU を持っている方は、host にインストールされている Nvidia ドライバのバージョンの関係で、docker の中でないとうまく動かないことがあるかもしれません。(未検証)

### Docker を使わずに開発する

```bash
$ curl -LsSf https://astral.sh/uv/0.8.0/install.sh | sh
$ echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
$ uv sync
```

### Docker を使って開発する

```bash
$ make build
$ # 以下のコマンドで、コンテナの中に入ることができる。コンテナの中で、uv や python3 等のコマンドを実行することができる。
$ make run
```

上記の `make run` か、もしくは `make up` でコンテナを裏側で立ち上げ続けることができる。VSCode 互換のエディタであれば、`Attach to Running Container...` で `/llm2025compet-app-1` を選ぶことで、コンテナ内で作業することができる。

## GPU ノードでの環境構築

こちらをご参照ください。

[Notion](https://www.notion.so/2379dd6b4cc2803aac5eddbf87c7a436)

## Team 間の IO インターフェース

それぞれ HuggingFace を経由することを想定しています。

- データ(HLE/DNA) <-> 学習: HuggingFace Datasets
  - データチームが push した Dataset を、学習チームが pull して学習に使用する
- 学習 <-> 推論: HuggingFace Models
  - 学習チームが push したモデルを、推論チームが pull して推論に使用する

## License

TBD
