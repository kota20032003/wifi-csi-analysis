import serial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np
import re
from collections import deque
import threading
import time
import csv
from datetime import datetime

# --- 設定項目 ---
SERIAL_PORT = 'COM5'      # ★ご自身のCOMポートに変更
BAUD_RATE = 115200
SUBCARRIER_TO_PLOT = 44
MAX_DATA_POINTS = 100
OUTPUT_FILENAME = 'csi_log_us_timestamp.csv'
CLOCK_FONTSIZE = 28
# -----------------

# 共有データ
y_data = deque([0.0] * MAX_DATA_POINTS, maxlen=MAX_DATA_POINTS)
is_running = True

# 末尾の [...] を抽出（最後の角括弧ペアを採用）
BRACKETS_LAST = re.compile(r'\[([^\[\]]+)\]\s*$')

def ts_iso_us_slash():
    """
    例) 2025/11/17␠␠15:54:44:123456
    （日付/区切り、間に半角スペース2つ、末尾コロン＋6桁マイクロ秒）
    """
    now = datetime.now()
    us = now.microsecond
    return now.strftime('%Y/%m/%d  %H:%M:%S') + f':{us:06d}'

def csi_data_handler():
    """シリアルからCSIを読み、CSV保存＆描画バッファ更新"""
    global is_running, y_data

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"ポートを開けませんでした: {e}")
        is_running = False
        return

    try:
        with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ラベル列は無し
            writer.writerow(['timestamp_iso_us', 'rssi', 'csi_payload'])
            f.flush()
            print(f"'{OUTPUT_FILENAME}' への保存を開始しました。")

            while is_running:
                try:
                    if ser.in_waiting == 0:
                        time.sleep(0.001)
                        continue

                    raw = ser.readline()
                    if not raw:
                        continue

                    line = raw.decode('utf-8', errors='ignore').strip()
                    if not line or 'CSI_DATA' not in line:
                        continue

                    # RSSI はヘッダ順に基づき index=3 を想定（必要なら調整）
                    parts = line.split(',')
                    if len(parts) < 5:
                        continue
                    rssi = parts[3].strip()

                    # 末尾の [ ... ] 部分をCSIペイロードとして抽出
                    m = BRACKETS_LAST.search(line)
                    if not m:
                        continue
                    payload_str = m.group(1).strip()

                    # I/Q の整数列に変換し、指定サブキャリアの振幅を可視化用に算出
                    try:
                        nums = [int(x) for x in payload_str.split()]
                    except ValueError:
                        continue

                    idx_i = (SUBCARRIER_TO_PLOT - 1) * 2
                    idx_q = idx_i + 1
                    if idx_q >= len(nums):
                        continue

                    i_val = nums[idx_i]
                    q_val = nums[idx_q]
                    amp = float(np.sqrt(i_val * i_val + q_val * q_val))
                    y_data.append(amp)

                    # CSVに所望のフォーマットでタイムスタンプ（マイクロ秒）を書き込み
                    writer.writerow([ts_iso_us_slash(), rssi, payload_str])
                    f.flush()

                except Exception:
                    # 例外はスキップして継続
                    continue

    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        is_running = False
        try:
            ser.close()
        except Exception:
            pass

# スレッド開始
data_thread = threading.Thread(target=csi_data_handler, daemon=True)
data_thread.start()

# --- 可視化（不要ならこのブロックを削除OK） ---
plt.ion()
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
fig.set_dpi(120)

(line,) = ax.plot(list(y_data))
ax.set_xlabel('Time (packets)')
ax.set_ylabel('Amplitude')
ax.set_ylim(0, 60)
ax.set_xlim(0, MAX_DATA_POINTS)
ax.set_title(f'Live CSI - Subcarrier {SUBCARRIER_TO_PLOT}')

# 画面に大きい時計オーバーレイ（マイクロ秒表示）
clock_text = ax.text(
    0.01, 0.98, '', transform=ax.transAxes, va='top', ha='left',
    fontsize=CLOCK_FONTSIZE, weight='bold', fontfamily='monospace',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.4')
)
clock_text.set_text(datetime.now().strftime('%H:%M:%S.%f'))  # ←%fはマイクロ秒

print("グラフ描画を開始。ウィンドウを閉じると終了。")

try:
    while is_running and plt.fignum_exists(fig.number):
        line.set_ydata(list(y_data))
        # 時計はマイクロ秒表示
        clock_text.set_text(datetime.now().strftime('%H:%M:%S.%f'))
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.03)
except KeyboardInterrupt:
    pass
finally:
    is_running = False
    data_thread.join()
    print("プログラムを終了しました。")
