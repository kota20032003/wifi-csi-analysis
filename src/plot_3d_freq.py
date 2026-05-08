import os
import re
import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import serial

# =========================
# 設定（ここだけ調整すればOK）
# =========================
SERIAL_PORT = "COM5"        # ★自分のCOM番号へ
BAUD_RATE = 115200

MAX_TIME_POINTS = 50
NUM_SUBCARRIERS = 64        # 64前提（IQ 64ペア）

CHANNEL_BW_HZ = 20e6        # ★ 20/40/80MHz に合わせて変更
NFFT = 64                   # ★ 通常64

# train.csv の並びがすでに中心(DC)合わせ済み前提なら True
DATA_ALREADY_SHIFTED = True

# 表示（白余白を減らして拡大）
FIGSIZE = (22, 12)          # ★大きめ（ここを更に上げてもOK）
DPI = 120                   # ★少し高解像度
UPDATE_SEC = 0.10
Z_LIM = (0, 60)

# train.csv から「壁ビン」を自動推定
TRAIN_CSV_PATH = "train.csv"    # active_ap に置く
TRAIN_SCAN_ROWS = 800
WALL_RATIO_THRESHOLD = 4.0
# =========================

csi_history = deque(maxlen=MAX_TIME_POINTS)
is_running = True
lock = threading.Lock()


def build_freq_axis_mhz(channel_bw_hz: float, nfft: int) -> np.ndarray:
    df_hz = channel_bw_hz / nfft
    k = np.arange(nfft) - (nfft // 2)  # -32..+31
    return (k * df_hz) / 1e6


def payload_to_amplitudes(payload_str: str, num_subcarriers: int) -> np.ndarray:
    arr = np.fromstring(payload_str, sep=" ", dtype=np.float32)

    need = num_subcarriers * 2
    if arr.size < need:
        arr = np.pad(arr, (0, need - arr.size))
    elif arr.size > need:
        arr = arr[:need]

    I = arr[0::2]
    Q = arr[1::2]
    amps = np.sqrt(I * I + Q * Q)
    return amps.astype(np.float32)


def detect_wall_bin_from_train(train_csv_path: str) -> int:
    if not os.path.exists(train_csv_path):
        print(f"[INFO] {train_csv_path} が見つからないため、壁ビン=0で続行します。")
        return 0

    try:
        import pandas as pd
        df = pd.read_csv(train_csv_path, usecols=["csi_payload"], nrows=TRAIN_SCAN_ROWS)
    except Exception as e:
        print(f"[INFO] {train_csv_path} を読めなかったため、壁ビン=0で続行します。({e})")
        return 0

    if df.empty:
        print("[INFO] train.csv が空なので、壁ビン=0で続行します。")
        return 0

    sums = np.zeros(NUM_SUBCARRIERS, dtype=np.float64)
    cnt = 0

    for s in df["csi_payload"].astype(str):
        amps = payload_to_amplitudes(s, NUM_SUBCARRIERS)
        sums += amps
        cnt += 1

    mean_per_bin = sums / max(cnt, 1)
    median_mean = float(np.median(mean_per_bin))
    max_idx = int(np.argmax(mean_per_bin))
    max_mean = float(mean_per_bin[max_idx])

    ratio = (max_mean / median_mean) if median_mean > 0 else float("inf")

    print(f"[INFO] train.csv 推定: max_mean={max_mean:.3f}, median_mean={median_mean:.3f}, ratio={ratio:.2f}")
    if ratio >= WALL_RATIO_THRESHOLD:
        print(f"[INFO] 壁ビン候補 = {max_idx}")
        return max_idx

    print("[INFO] 突出ビンが弱いので、壁ビン=0で続行します。")
    return 0


WALL_BIN = detect_wall_bin_from_train(TRAIN_CSV_PATH)
FREQ_AXIS_MHZ = build_freq_axis_mhz(CHANNEL_BW_HZ, NFFT)[:NUM_SUBCARRIERS]
print(f"[INFO] WALL_BIN={WALL_BIN}, 周波数(目安)={FREQ_AXIS_MHZ[WALL_BIN]:.3f} MHz")


def csi_data_handler():
    global is_running

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] ポートを開けませんでした: {e}")
        is_running = False
        return

    csi_pattern = re.compile(r"\[(.*?)\]")

    while is_running:
        try:
            if ser.in_waiting <= 0:
                time.sleep(0.01)
                continue

            line = ser.readline().decode("utf-8", errors="ignore")
            if not line.startswith("CSI_DATA"):
                continue

            m = csi_pattern.search(line)
            if not m:
                continue

            nums = np.fromstring(m.group(1), sep=" ", dtype=np.float32)

            need = NUM_SUBCARRIERS * 2
            if nums.size < need:
                nums = np.pad(nums, (0, need - nums.size))
            elif nums.size > need:
                nums = nums[:need]

            I = nums[0::2]
            Q = nums[1::2]
            amps = np.sqrt(I * I + Q * Q).astype(np.float32)

            if not DATA_ALREADY_SHIFTED:
                amps = np.fft.fftshift(amps)

            # ★壁ビンだけ潰す
            if 0 <= WALL_BIN < NUM_SUBCARRIERS:
                amps[WALL_BIN] = 0.0

            with lock:
                csi_history.append(amps.tolist())

        except Exception:
            pass

    try:
        ser.close()
    except Exception:
        pass


def setup_big_window(fig):
    """
    OS/バックエンド差があるので、できる範囲でウィンドウ最大化。
    失敗しても無視して続行。
    """
    try:
        mng = plt.get_current_fig_manager()
        # TkAgg など
        if hasattr(mng, "window") and hasattr(mng.window, "state"):
            mng.window.state("zoomed")
        # Qt系
        elif hasattr(mng, "window") and hasattr(mng.window, "showMaximized"):
            mng.window.showMaximized()
        # WXなど
        elif hasattr(mng, "frame") and hasattr(mng.frame, "Maximize"):
            mng.frame.Maximize(True)
    except Exception:
        pass


def main():
    global is_running

    t = threading.Thread(target=csi_data_handler, daemon=True)
    t.start()

    plt.ion()

    # 重要：constrained_layout を使って余白を減らす
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)

    # 重要：Axes を Figure 全体に貼り付け（白余白を減らす）
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.92], projection="3d")

    setup_big_window(fig)

    # X(Time), Y(Freq) メッシュ（固定）
    X = np.arange(MAX_TIME_POINTS)
    Y = FREQ_AXIS_MHZ
    X, Y = np.meshgrid(X, Y)

    print("3D表示中。ウィンドウを閉じるか Ctrl+C で終了。")

    while is_running:
        try:
            with lock:
                if len(csi_history) == 0:
                    time.sleep(UPDATE_SEC)
                    continue
                Z = np.array(list(csi_history), dtype=np.float32).T  # (subcarrier, time)

            if Z.shape[1] < MAX_TIME_POINTS:
                pad = np.zeros((NUM_SUBCARRIERS, MAX_TIME_POINTS - Z.shape[1]), dtype=np.float32)
                Z = np.hstack((pad, Z))

            ax.clear()
            ax.plot_surface(X, Y, Z, cmap="viridis")

            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency offset (MHz)")
            ax.set_zlabel("Amplitude")
            ax.set_zlim(*Z_LIM)
            ax.set_title("CSI 3D Amplitude (Frequency axis)")

            # 追加：タイトル・ラベルの余白を詰める
            try:
                ax.margins(x=0, y=0, z=0)
            except Exception:
                pass

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(UPDATE_SEC)

            if not plt.fignum_exists(fig.number):
                is_running = False

        except KeyboardInterrupt:
            is_running = False
        except Exception as e:
            print("[ERROR]", e)
            is_running = False

    t.join()
    print("終了しました。")


if __name__ == "__main__":
    main()
