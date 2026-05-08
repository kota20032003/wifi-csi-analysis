# smooth_fsm.py
#
# 状態遷移ルールで予測ラベルをなめらかにする。
# ルール:
#   0 -> {0,1}
#   1 -> {1,2}
#   2 -> {2,0}
# 許されない遷移が出たら「前の状態を維持」する。
#
# 入力:  timestamp_iso_us, true_label, pred_label を持つ CSV
# 出力:  pred_label を上書きした CSV

import pandas as pd

# ===== ここだけ適宜書き換える =====
INPUT_PATH = "pred_iq_noweight.csv"          # 入力ファイル
OUTPUT_PATH = "pred_iq_noweight_fsm.csv"     # 出力ファイル
# ===================================


def next_allowed_states(prev_state: int):
    """前の状態から許される次状態の集合を返す。"""
    if prev_state == 0:
        return {0, 1}
    elif prev_state == 1:
        return {1, 2}
    elif prev_state == 2:
        return {2, 0}
    else:
        # 想定外クラスはとりあえず自分自身のみ許可
        return {prev_state}


def smooth_with_fsm(labels):
    """状態遷移ルールに従って系列をなめらかにする。"""
    if len(labels) == 0:
        return labels

    smoothed = labels.copy()

    for i in range(1, len(labels)):
        prev = smoothed[i - 1]
        cur = labels[i]
        allowed = next_allowed_states(prev)

        if cur in allowed:
            smoothed[i] = cur
        else:
            # 許可されない遷移なら、前の状態を維持
            smoothed[i] = prev

    return smoothed


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    if "pred_label" not in df.columns:
        raise ValueError("CSV に 'pred_label' 列がありません。")

    preds = df["pred_label"].astype(int).values
    smoothed = smooth_with_fsm(preds)

    df["pred_label"] = smoothed  # 上書き

    print(f"Saving smoothed result to {OUTPUT_PATH} ...")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("Done.")


if __name__ == "__main__":
    main()
