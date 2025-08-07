import logging

import datasets
from datasets import DatasetDict, concatenate_datasets
import json

from configs import DataConfig

logger = logging.getLogger(__name__)

def get_datas_from_config(config: DataConfig, seed: int = 42):
    """
    設定オブジェクトに基づき、複数のデータセットをロードして結合する。
    最終的なカラム名は 'prompt' と 'completion' に統一される。

    Args:
        config (DataConfig): ロードするデータセットの情報を含む設定オブジェクト。

    Returns:
        DatasetDict: 全てのデータセットを結合し、カラム名を整形した単一のDatasetオブジェクト。
    """
    loaded_datasets = []
    
    print("データセットのロードを開始します...")
    for i, dataset_info in enumerate(config.datasets):
        print(f"  ({i+1}/{len(config.datasets)}) ロード中: {dataset_info.name} ({dataset_info.config})")
        # 1. スライシングを指定してデータセットをロード
        # from_idとto_idが両方指定されているかチェック
        # if dataset_info.from_id is not None and dataset_info.to_id is not None:
        #     # 指定されている場合、スライスしてロード
        #     split_spec = f"{dataset_info.config}[{dataset_info.from_id - 1}:{dataset_info.to_id}]"
        #     print(f"  ({i+1}/{len(config.datasets)}) ロード中: {dataset_info.name}, スライス: {split_spec}")
        # else:
        #     # いずれかがNoneの場合、分割全体をロード
        #     split_spec = dataset_info.config
        #     print(f"  ({i+1}/{len(config.datasets)}) ロード中: {dataset_info.name}, 分割: {split_spec} (全件)")

        # 1. データセットをロード
        dataset = datasets.load_dataset(dataset_info.name, name=dataset_info.config)
        
        print(f"  データセット '{dataset_info.name}' の分割: {list(dataset.keys())}")
        print(f"  データセット '{dataset_info.name}' のカラム: {list(dataset.column_names.values())}")
        print(f"  データセット '{dataset_info.name}' のサンプル数: {len(dataset['train'])} (train)")
        print(f"  データセット '{dataset_info.name}' の最初のサンプル: {dataset['train'][0]}")

        # 2. 指定されたカラム名を一時的に 'prompt' と 'completion' に統一
        #    これにより、後続の処理を共通化できる
        temp_rename_dict = {}
        # 元の question_field が存在する場合のみリネーム対象に追加
        if dataset_info.question_field in list(dataset.column_names.values())[0]:
            temp_rename_dict[dataset_info.question_field] = 'prompt'
            print(f"  question_field '{dataset_info.question_field}' を 'prompt' にリネームしました。")
        else:
            logger.warning(f"指定された question_field '{dataset_info.question_field}' がデータセットに存在しません。")
        # 元の answer_field が存在する場合のみリネーム対象に追加
        if dataset_info.answer_field in list(dataset.column_names.values())[0]:
            temp_rename_dict[dataset_info.answer_field] = 'completion'
            print(f"  answer_field '{dataset_info.answer_field}' を 'completion' にリネームしました。")
        else:
            logger.warning(f"指定された answer_field '{dataset_info.answer_field}' がデータセットに存在しません。")

        # リネームを実行
        dataset = dataset.rename_columns(temp_rename_dict)

        # 3. 'text' カラムに整形する関数を定義
        def format_to_text_column(example):
            """
            'prompt' と 'completion' の内容から指定のJSON形式の文字列を作成する。
            """
            messages = [
                {"role": "user", "content": example.get('prompt')},
                {"role": "assistant", "content": example.get('completion')}
            ]
            # json.dumpsでJSON文字列に変換（ensure_ascii=Falseで日本語を正しく扱う）
            return {"messages": messages}

        # 4. map関数を適用して全データセットの各分割に新しいフォーマットを適用
        #    同時に、整形に使った 'prompt', 'completion' やその他不要なカラムをすべて削除
        current_columns = list(list(dataset.column_names.values())[0])
        dataset = dataset.map(format_to_text_column, remove_columns=current_columns)
        
        # 一番目をprint
        print(f"  データセット '{dataset_info.name}' の最初のサンプル: {dataset['train'][0]}")
        print(f"  データセット '{dataset_info.name}' のカラム: {list(dataset.column_names.values())}")

        loaded_datasets.append(dataset)

    if not loaded_datasets:
        raise ValueError("ロードできるデータセットがありませんでした。")

    print("\n全データセットを結合中...")
    # すべてのDatasetDictからキーの集合を取得
    all_keys = set(k for dd in loaded_datasets for k in dd.keys())
    print(f"結合するキー: {all_keys}")
    print(list(list(loaded_datasets[0].column_names.values())[0]))

    # 辞書内包表記を使って各キーごとにデータセットを結合
    combined_dataset = DatasetDict({
        "train" : concatenate_datasets([data["train"] for data in loaded_datasets]),
    })
    
    

    print("結合が完了しました！")
    # print(combined_dataset['train'][0])  # 最初のサンプルを表示して確認
    
    print("\nデータセットをシャッフル中...")
    # DatasetDict全体をシャッフルする。引数で受け取ったseedを使用する。
    combined_dataset = combined_dataset.shuffle(seed=seed)
    print(f"シャッフルが完了しました！ (シード: {seed})")
    
    print("最終的なデータセットの情報:")
    print(combined_dataset)
    print(combined_dataset['train'])

    print("  最初のサンプル:", combined_dataset['train'][0])
    print("  最初のサンプルの文字数:", len(json.dumps(combined_dataset['train'][0]['messages'], ensure_ascii=False)))
    print("  文字数の合計:", sum(len(json.dumps(sample['messages'], ensure_ascii=False)) for sample in combined_dataset['train']))

    return combined_dataset