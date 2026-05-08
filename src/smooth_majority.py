# smooth_majority.py
#
# 幅 w の窓で多数決スムージングを行う。
#
# 入力:  timestamp_iso_us, true_label, pred_label を持つ CSV
# 出力:  pred_label を上書きした CSV

import pandas as pd
from collections import Counter

# ===== ここだけ適宜書き換える =====
INPUT_PATH = "pred_iq_noweight.csv"              # 入力ファイル
OUTPUT_PATH = "pred_iq_noweight_mv_w5.csv"       # 出力ファイル
WINDOW_SIZE = 5                                   # 奇数推奨 (3,5,7,...)
# ===================================


def smooth_with_majority(labels, w: int):
    """多数決スムージング。窓幅 w は奇数推奨。"""
    n = len(labels)
    if n == 0:
        return labels
    if w <= 1:
        return labels

    r = w // 2
    smoothed = labels.copy()

    for i in range(n):
        left = max(0, i - r)
        right = min(n, i + r + 1)  # Python のスライスは右端非含
        window = labels[left:right]

        counts = Counter(window)
        # 最頻値の個数
        max_count = max(counts.values())
        # 個数が最大のクラスたち
        candidates = [cls for cls, c in counts.items() if c == max_count]

        if len(candidates) == 1:
            smoothed[i] = candidates[0]
        else:
            # 複数クラスが同数なら元のラベルを優先
            orig = labels[i]
            if orig in candidates:
                smoothed[i] = orig
            else:
                # それでも決まらないなら最小クラス番号
                smoothed[i] = min(candidates)

    return smoothed


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    if "pred_label" not in df.columns:
        raise ValueError("CSV に 'pred_label' 列がありません。")

    preds = df["pred_label"].astype(int).values
    print(f"Applying majority smoothing (window={WINDOW_SIZE}) ...")
    smoothed = smooth_with_majority(preds, WINDOW_SIZE)

    df["pred_label"] = smoothed  # 上書き

    print(f"Saving smoothed result to {OUTPUT_PATH} ...")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("Done.")


if __name__ == "__main__":
    main()
