# csi_experiment.py
#
# train.csv の csi_payload から「データを載せないサブキャリア（ガード帯＋DC）」を除外して
# 4パターンのロジスティック回帰を実行するコード
#
# 4パターン:
# 1) IQ (csi_payloadそのまま) クラス重みなし
# 2) IQ クラス重みあり
# 3) 振幅特徴量 クラス重みなし
# 4) 振幅特徴量 クラス重みあり
#
# 各パターンごとに予測結果 CSV も保存する:
#   pred_iq_noweight.csv
#   pred_iq_balanced.csv
#   pred_amp_noweight.csv
#   pred_amp_balanced.csv

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ====== 使うファイル名（同じフォルダに置く） ======
TRAIN_PATH = "train.csv"
TEST_NOLABEL_PATH = "test_nolabel.csv"
TEST_LABEL_PATH = "test_label.csv"


# ====== csi_payload 文字列 -> (I, Q)ペア列に変換 ======
def parse_iq_pairs(payload_str: str):
    """
    "84 -64 4 0 0 0 ..." のようなスペース区切りの文字列を
    [(I1, Q1), (I2, Q2), ..., (I64, Q64)] に変換する
    """
    vals = list(map(int, payload_str.split()))
    # 偶数番目を I, 奇数番目を Q としてペアにする
    pairs = list(zip(vals[0::2], vals[1::2]))
    return pairs


# ====== ガード帯＋DC サブキャリアのインデックス ======
# 0 始まりのサブキャリア番号（(I,Q)ペアの番号）。
#
# データを観察すると、
# - 先頭 6 ペア: だいたい "84 -64 4 0 0 0 0 0 0 0 0 0" のようになっていて、データを載せない部分
# - 真ん中あたりの 1 ペア: (0,0) 固定
# - 最後の 5 ペア: (0,0) 固定
#
# これらをまとめて「使わないサブキャリア」として除外する。
GUARD_DC_INDICES = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]


# ====== 特徴量変換: IQ そのまま ======
def make_features_iq(series_csi: pd.Series, removed_indices=None) -> np.ndarray:
    """
    csi_payload から (I,Q) をそのまま特徴量にする。
    removed_indices で指定されたサブキャリアは除外する。
    戻り値の shape は (サンプル数, 2 * 有効サブキャリア数)
    """
    if removed_indices is None:
        removed_indices = []

    X = []
    for s in series_csi:
        pairs = parse_iq_pairs(s)
        kept = [p for i, p in enumerate(pairs) if i not in removed_indices]
        # [I1, Q1, I2, Q2, ...] にフラット化
        flat = [v for (I, Q) in kept for v in (I, Q)]
        X.append(flat)
    return np.array(X, dtype=float)


# ====== 特徴量変換: 振幅 sqrt(I^2 + Q^2) ======
def make_features_amp(series_csi: pd.Series, removed_indices=None) -> np.ndarray:
    """
    csi_payload から sqrt(I^2 + Q^2) を特徴量にする。
    removed_indices で指定されたサブキャリアは除外する。
    戻り値の shape は (サンプル数, 有効サブキャリア数)
    """
    if removed_indices is None:
        removed_indices = []

    X = []
    for s in series_csi:
        pairs = parse_iq_pairs(s)
        kept = [p for i, p in enumerate(pairs) if i not in removed_indices]
        amps = [math.sqrt(I * I + Q * Q) for (I, Q) in kept]
        X.append(amps)
    return np.array(X, dtype=float)


# ====== 1パターン分の学習・評価・CSV保存 ======
def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    time_str,
    class_weight,
    description: str,
    out_csv: str,
):
    print("============================================================")
    print(description)
    print("  学習データ形状:", X_train.shape)
    print("  テストデータ形状:", X_test.shape)
    print("  class_weight :", class_weight)
    print("------------------------------------------------------------")

    clf = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight=class_weight,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 正解率
    acc = accuracy_score(y_test, y_pred)

    # マクロ平均の適合率・再現率・F値
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"  正解率 (accuracy)        : {acc:.4f}")
    print(f"  適合率 (precision, macro): {prec_macro:.4f}")
    print(f"  再現率 (recall, macro)   : {rec_macro:.4f}")
    print(f"  F値 (F1, macro)          : {f1_macro:.4f}")
    print()

    # クラスごとの指標
    labels = np.unique(y_test)
    prec_per, rec_per, f1_per, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    print("  クラスごとの指標:")
    for cls, p, r, f1, sup in zip(labels, prec_per, rec_per, f1_per, support):
        print(
            f"    クラス {cls}: "
            f"適合率={p:.3f}, 再現率={r:.3f}, F1={f1:.3f}, サンプル数={sup}"
        )

    # 混同行列
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\n  混同行列 (行: 正解ラベル, 列: 予測ラベル):")
    print("    ラベル順:", labels.tolist())
    print(cm)
    print()

    # ---- 予測結果 CSV を保存 ----
    df_out = pd.DataFrame(
        {
            "timestamp_iso_us": time_str.values,
            "true_label": y_test,
            "pred_label": y_pred,
        }
    )
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  予測結果を {out_csv} に保存しました。")


def main():
    # ---------- データ読み込み ----------
    train = pd.read_csv(TRAIN_PATH)
    test_nolabel = pd.read_csv(TEST_NOLABEL_PATH)
    test_label = pd.read_csv(TEST_LABEL_PATH)

    # ラベル
    y_train = train["label"].values
    y_test = test_label["label"].values

    # test_nolabel と test_label が同じ順番か一応チェック（timestamp で確認）
    if not test_nolabel["timestamp_iso_us"].equals(test_label["timestamp_iso_us"]):
        raise ValueError(
            "test_nolabel.csv と test_label.csv の行の並び（timestamp）が一致していません。"
        )

    # 時刻（文字列）: 予測結果CSVに書き出すため test_nolabel 側を使う
    time_str = test_nolabel["timestamp_iso_us"].astype(str)

    # ---------- 特徴量作成 ----------
    print("除外するサブキャリア (0 始まりインデックス):", GUARD_DC_INDICES)

    # 1) IQ 特徴量
    X_train_iq = make_features_iq(train["csi_payload"], GUARD_DC_INDICES)
    X_test_iq = make_features_iq(test_nolabel["csi_payload"], GUARD_DC_INDICES)

    # 2) 振幅特徴量
    X_train_amp = make_features_amp(train["csi_payload"], GUARD_DC_INDICES)
    X_test_amp = make_features_amp(test_nolabel["csi_payload"], GUARD_DC_INDICES)

    print("IQ 特徴量次元数:", X_train_iq.shape[1])
    print("振幅 特徴量次元数:", X_train_amp.shape[1])
    print()

    # ---------- 4パターン実験 + 予測 CSV 出力 ----------

    # (1) csi_payload を特徴量（IQ）、クラス不均衡補正なし
    run_experiment(
        X_train_iq,
        y_train,
        X_test_iq,
        y_test,
        time_str,
        class_weight=None,
        description="(1) IQ特徴量, クラス不均衡補正なし",
        out_csv="pred_iq_noweight.csv",
    )

    # (2) csi_payload を特徴量（IQ）、クラス不均衡補正あり
    run_experiment(
        X_train_iq,
        y_train,
        X_test_iq,
        y_test,
        time_str,
        class_weight="balanced",
        description="(2) IQ特徴量, クラス不均衡補正あり (class_weight='balanced')",
        out_csv="pred_iq_balanced.csv",
    )

    # (3) 振幅 sqrt(I^2 + Q^2) を特徴量、クラス不均衡補正なし
    run_experiment(
        X_train_amp,
        y_train,
        X_test_amp,
        y_test,
        time_str,
        class_weight=None,
        description="(3) 振幅特徴量, クラス不均衡補正なし",
        out_csv="pred_amp_noweight.csv",
    )

    # (4) 振幅 sqrt(I^2 + Q^2) を特徴量、クラス不均衡補正あり
    run_experiment(
        X_train_amp,
        y_train,
        X_test_amp,
        y_test,
        time_str,
        class_weight="balanced",
        description="(4) 振幅特徴量, クラス不均衡補正あり (class_weight='balanced')",
        out_csv="pred_amp_balanced.csv",
    )


if __name__ == "__main__":
    main()
