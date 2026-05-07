# Wi-Fi CSI Analysis

## 概要

卒業研究で使用したWi-Fi CSI解析用のPythonコードです．
ESP32から取得したCSIデータを用いて，CSIの取得，3D可視化，機械学習による分類，予測結果の後処理を行います．

## 背景

Wi-Fi CSI (Channel State Information: チャネル状態情報) は，Wi-Fi信号が空間を伝わるときの変化を表す情報です．
本研究では，ESP32で取得したCSIデータを用いて，環境変化や状態変化を観測することを目的としました．

## 主な機能

- ESP32からCSIデータを取得し，CSVに保存
- CSIの振幅をリアルタイムに可視化
- 時間，周波数，振幅の3Dグラフを表示
- IQ特徴量と振幅特徴量を用いたロジスティック回帰
- 状態遷移ルールによる予測ラベルのスムージング
- 多数決による予測ラベルのスムージング

## ファイル構成

```text
src/
├── plot_csi_ravel2_old.py
├── plot_3d_freq.py
├── csi_experiment.py
├── smooth_fsm.py
└── smooth_majority.py
