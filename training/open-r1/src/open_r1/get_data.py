import logging
import datasets
from datasets, import DatasetDict, concatenate_dataset
import json

from configs import DataConfig

logger = logging.getLogger(__name__)

def get_data_from_config(config: DataConfig, seed: int=42):
    """
    設定オブジェクト(yamlファイルからロード)に基づいて、複数のデータセットをロードして結合する関数
    最終的なカラム名は "question", "chosen", "rejected" に統一される
    注: HF上にあるデータセットのカラムは "question", "preferred_output", "non_preferred_output" を想定

    Args:
        config(DataConfig): ロードするデータセットの情報を含む設定オブジェクト
    return:
        DatasetDict: 全データセットを結合し、カラム名を統一させた単一のDatasetオブジェクト
    """
    loaded_dataset = [] # 読み込んだデータセットをここに追加する

    print("Start loading dataset...")
    for i, dataset_info in enumerate(config.datasets):
        print(f"  ({i+1}/{len(config.datasets)}) Loading dataset name: {dataset_info.name} ({dataset_info.config})")
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

        print(f"データセット [{dataset_info.name}] の分割: {list(dataset.keys())}")
        print(f"  データセット [{dataset_info.name}] のカラム: {list(dataset.column_names.values())}")
        print(f"  データセット [{dataset_info.name}] のサンプル数(Train): {len(dataset['train'])}")
        # print(f"  データセット '{dataset_info.name}' の最初のサンプル: {dataset['train'][0]}")

        # 2. 指定されたカラム名を一時的に "question", "preferred_output", "non_preferred_output" に統一(元々されていると信じてはいるが...)
        tmp_rename_dict = {}
        # 元の instructtion_field が存在する場合のみリネーム対象に追加
        # 存在しない場合は warning で知らせる
        if dataset_info.instruction_field in list(dataset.column_names.values())[0]:
            tmp_rename_dict[dataset_info.instruction_field] = "question"
            print(f"データセット [{dataset_info.name}] の instruction_field [{dataset_info.instruction_field}] を 'question' に変更しました")
        else:
            logger.warning(f"データセット [{dataset_info.name}] に指定された instruction_field: [{dataset_info.instruction_field}] がデータセットに存在しません")
        # 元の preferred_output が存在する場合のみリネーム対象に追加
        # 存在しない場合は warining で知らせる
        if dataset_info.chosen_field in list(dataset.column_names.values())[0]:
            tmp_rename_dict[dataset_info.chosen_field] = "preferred_output"
            print(f"データセット [{dataset_info.name}] の chosen_field [{dataset_info.chosen_field}] を 'preferred_output' に変更しました")
        else:
            logger.warining(f"データセット [{dataset_info.name}] に指定された chosen_field: [{dataset_info.chosen_field}] がデータセットに存在しません")
        # 元の non_preferred_output が存在する場合のみリネーム対象に追加
        # 存在しない場合は warining で知らせる
        if dataset_info.rejected_field in list(dataset.column_names.values())[0]:
            tmp_rename_dict[dataset_info.rejected_field] = "non_preferred_output"
            print(f"データセット [{dataset_info.name}] の rejected_field [{dataset_info.rejected_field}] を 'non_preferred_output' に変更しました")
        else:
            logger.warining(f"データセット [{dataset_info.name}] に指定された rejected_field: [{dataset_info.rejected_field}] がデータセットに存在しません")
        
        # リネームを実行
        dataset = dataset.rename_columns(tmp_rename_dict)

        # 3. 選好チューニング用の整形に関する関数を定義
        def format_pref(example):
            return {
                "prompt": example["question"],
                "chosen": example["preferred_output"],
                "rejected": example["non_preferred_output"]
            }
        
        # 4. map関数で全データセットに新しいフォーマットを適用
        dataset["train"] = dataset["train"].map(format_pref, remove_columns=dataset["train"].column_names)

        # 先頭データを表示
        print(f"データセット [{dataset_info.name}] のカラム: {list(dataset.column_names.values())}")

        loaded_dataset.append(dataset)
    
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
    print(combined_dataset["train"])

    return combined_dataset