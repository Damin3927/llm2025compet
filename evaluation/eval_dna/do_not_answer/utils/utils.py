import numpy as np
import pandas as pd
#from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import json
from typing import Dict, List, Any
import yaml
import os

def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    正解率 (Accuracy)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    normalize : bool, default=True
        True の場合は正解率 (0～1) を返し、False の場合は正解数を返す。
    sample_weight : array-like of shape (n_samples,), default=None
        各サンプルの重み。

    Returns
    -------
    score : float or int
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")
    matches = (y_true == y_pred)
    if sample_weight is None:
        if normalize:
            return np.mean(matches)
        else:
            return np.sum(matches)
    else:
        sw = np.asarray(sample_weight)
        if sw.shape != matches.shape:
            raise ValueError(f"sample_weight must have shape {matches.shape}, got {sw.shape}")
        if normalize:
            return sw[matches].sum() / sw.sum()
        else:
            return sw[matches].sum()


def confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None):
    """
    混同行列

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    labels : array-like, default=None
        ラベルの一覧。None の場合は np.unique(y_true, y_pred) のソート順。
    sample_weight : array-like of shape (n_samples,), default=None
    normalize : {'true', 'pred', 'all'}, default=None
        'true' : 各行 (真のラベルごと) を正規化
        'pred' : 各列 (予測ラベルごと) を正規化
        'all'  : 全体を正規化

    Returns
    -------
    C : ndarray of shape (n_labels, n_labels)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    n_labels = labels.shape[0]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    C = np.zeros((n_labels, n_labels), dtype=float if normalize else int)

    if sample_weight is None:
        for t, p in zip(y_true, y_pred):
            i = label_to_index[t]
            j = label_to_index[p]
            C[i, j] += 1
    else:
        sw = np.asarray(sample_weight)
        for t, p, w in zip(y_true, y_pred, sw):
            i = label_to_index[t]
            j = label_to_index[p]
            C[i, j] += w

    if normalize == "true":
        C = C / C.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        C = C / C.sum(axis=0, keepdims=True)
    elif normalize == "all":
        C = C / C.sum()
    return C


def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average=None,
    sample_weight=None,
    zero_division="warn"
):
    """
    Precision, recall, F-score, support

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    labels : array-like, default=None
    pos_label : str or int, default=1 (binary のときだけ使用)
    average : {'micro', 'macro', 'weighted', None}, default=None
    sample_weight : array-like of shape (n_samples,), default=None
    zero_division : "warn", 0 or 1, default="warn"

    Returns
    -------
    precision : array-like or float
    recall    : array-like or float
    fscore    : array-like or float
    support   : ndarray of int
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    # ラベル定義
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    # サポート (真ラベル数) と TP, FP, FN のカウント
    support = np.array([np.sum(y_true == lab) for lab in labels], dtype=int)
    tp = np.array([np.sum((y_true == lab) & (y_pred == lab)) for lab in labels], dtype=int)
    fp = np.array([np.sum((y_true != lab) & (y_pred == lab)) for lab in labels], dtype=int)
    fn = np.array([np.sum((y_true == lab) & (y_pred != lab)) for lab in labels], dtype=int)

    # クラスごとの precision, recall, fscore
    with np.errstate(divide="ignore", invalid="ignore"):
        precision_per = tp / (tp + fp)
        recall_per    = tp / (tp + fn)
        fscore_per = 2 * precision_per * recall_per / (precision_per + recall_per)

    # zero_division の処理
    def _handle_zero_div(arr, denom, name):
        mask = denom == 0
        if zero_division in (0, 1):
            arr[mask] = zero_division
        elif zero_division == "warn" and np.any(mask):
            # sklearn では Warning を発行しますが省略
            arr[mask] = 0.0
        return arr

    precision_per = _handle_zero_div(
        precision_per, tp + fp, "precision"
    )
    recall_per = _handle_zero_div(
        recall_per, tp + fn, "recall"
    )
    fscore_per = _handle_zero_div(
        fscore_per, precision_per + recall_per, "fscore"
    )

    # 平均化
    if average is None:
        return precision_per, recall_per, fscore_per, support

    # マイクロ平均：全合集計
    if average == "micro":
        tp_sum = tp.sum()
        if sample_weight is None:
            total_pred = y_pred.shape[0]
            total_true = y_true.shape[0]
        else:
            sw = np.asarray(sample_weight)
            total_pred = sw.sum()
            total_true = sw.sum()
        p = tp_sum / total_pred if total_pred > 0 else 0.0
        r = tp_sum / total_true  if total_true  > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f, support

    # マクロ平均：クラス平均
    if average == "macro":
        p = np.mean(precision_per)
        r = np.mean(recall_per)
        f = np.mean(fscore_per)
        return p, r, f, support

    # 重み付き平均：サポートで重み付け
    if average == "weighted":
        total = support.sum()
        if total == 0:
            return 0.0, 0.0, 0.0, support
        p = np.sum(precision_per * support) / total
        r = np.sum(recall_per * support)    / total
        f = np.sum(fscore_per * support)    / total
        return p, r, f, support

    raise ValueError("average must be one of {None, 'micro', 'macro', 'weighted'}")

# Read YAML file
base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path,'info.yaml'), 'r') as file:
    info_data = yaml.safe_load(file)

model_names = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]

def save_json(dictionary: Dict[str, Any], save_dir: str) -> None:
    # Serializing json
    json_object = json.dumps(dictionary, indent=4, ensure_ascii=False)

    # Writing to sample.json
    with open(save_dir, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)

def read_json(filepath: str) -> Dict[str, Any]:
    data = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data