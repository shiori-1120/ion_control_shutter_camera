# Shutter/Camera Trigger GUI 使い方メモ

## 前提
- Windows + NI-DAQmx ドライバ
- Python 環境: C:/Users/shiori/Desktop/ion_control_shutter_camera/myenv/
- 実機無しでも dry モードで UI 挙動を確認できます。

## 起動
- リポジトリルートで実行:
  - C:/Users/shiori/Desktop/ion_control_shutter_camera/myenv/Scripts/python.exe -m src.shutter_camera_trigger.shutter_gui

## 画面ざっくり
- Sequence タブ: DO→AO→DO の連続ループ（手動で開始/停止）。
- Manual タブ: DO を手動でオン/オフして即時反映。
- Sweep タブ: **段階化**（1) ROI check → 2) Threshold → 3) Start spectrum）。しきい値適用後のみスペクトル取得を開始。
- Camera タブ: **Snap**（TTL送出 → 1枚取得 → `.npy` 保存 → 表示）。

## 上部バー
- Device: NI-DAQ デバイス名 (例: Dev3)。
- DAQ mode: dry=模擬、real=実機 NI-DAQ。
- AO width (ms): Sequence タブ用の AO パルス幅。
- Connect / Disconnect: DAQ ワーカーの開始/終了。Disconnect 時に DO を全 OFF。
- 重要: **シーケンスを回していないときは 397nm を基本ON**（冷却/シェルビング維持の安全側デフォルト）。Stop 後や Sweep 終了時も 397 に戻します。All Off は警告が出ます。
- FG VISA / FG Connect: VISA 文字列を入れて接続確認。接続中は FG Disconnect で閉じる。スイープでも同じ文字列を共有します。
- FG amp (mVpp): FG の出力振幅を mVpp 指定（内部では Vpp に換算して設定）。
- Camera mode / Camera check: カメラワーカーの dry/real を選び、接続確認だけ行う。dry のときは下の Dry images でフォルダ指定可能。
- Exposure (ms): カメラ露光時間（ミリ秒）。Camera check と Sweep のカメラ取得に反映。
- Dry images (dry cam): dry カメラ用の画像フォルダ。bright* / dark* のファイル（png/jpg/bmp/tif/npy）を置くと、取得要求ごとにランダムで使います（無ければ従来の乱数合成）。
- Cam trig (source/connector/polarity/active/mode/delay): カメラのトリガ設定（`EXTERNAL` のときは TTL による外部トリガ想定）。GUIから設定し、ワーカーに渡します。

補足:
- 本プロジェクトの DO bit は `b3 b2 b1 b0 = 854 CAM_TRIG 397_SIG 397`（右端が397）です。
- **ROI check / Snap は 397 を開けたまま**カメラトリガを1回出す（パルス）構成です。

## Sweep タブのパラメータ
- Freq start/stop/step (Hz): スイープ周波数レンジと刻み (例: 80e6 → 82e6 を 0.5e6 刻み)。
- n_target: 周波数ごとの目標ショット数（カメラ判定成功件数）。
- max_attempt: 試行の上限（タイムアウト含む）。
- settle_s: 周波数変更後に待つ秒数。
- Sequence JSON: DO シーケンス定義ファイルへのパス。`...` で選択。
- DAQ mode / Camera mode / DAQ device: スイープ時に使うモードとデバイス名。
- FG VISA: 関数発生器の VISA リソース文字列 (例: USB0::0x0699::0x03A2::C040073::INSTR)。上部バーと共有。
- No FG: チェック時は FG 制御をスキップ。外すと FG を pyVISA で制御。
- FG amp (mVpp): FG 振幅。Start sweep 時に適用。
- Update interval (s): グラフ・ステータスの更新間隔。負荷を下げたいときは大きめに。
- 1) ROI check: 397のみ + カメラトリガ1回でフレーム取得し、ROI枠を表示（分布プロットは **しない**）。
- 2) Threshold: 選択中のTTLシーケンスを N 回実行して `S_norm` を収集し、Otsu 由来の `tau` を推定。
  - `S_norm > tau` を bright、`S_norm <= tau` を dark として事後分類。
  - bright/dark の分布を **1枚あたり**（weights=1/N_group）で同一グラフに重ね描き。
  - `threshold.json` を出力し、同意した場合はカメラワーカーに `tau_on/tau_off` を適用。
  - agreement は「単純閾値」と「ヒステリシス判定」の一致率（自己整合指標）です。
- 3) Start spectrum / Stop: しきい値適用後のみ開始可能。開始中は入力がロックされます。進行状況を表示し、完了時に保存します。

## 出力
- Camera Snap:
  - data/output/camera_snap/<timestamp>/snap.npy
- Sweep:
  - data/output/spectrum/<timestamp>/roi_check.npy
  - data/output/spectrum/<timestamp>/threshold.json
  - data/output/spectrum/<timestamp>/shots.csv, spectrum.csv, spectrum.png
  - data/output/spectrum/<timestamp>/config.json（実行時設定）

## モードと接続
- DAQ dry: NI-DAQ なしで動作確認。
- Camera dry: 画像を使わず疑似判定。
- FG: No FG を外し、FG VISA に文字列を入れると接続を試みます。機種は src/lib/instruments/rigol_dg.py に準拠。

## 典型的な手順 (実機なしの dry)
1. 起動後、DAQ mode を dry にして Connect。
2. （任意）Dry images に bright*/dark* 画像を置いたフォルダを指定して Camera check。
3. Camera タブの Snap を押して、`snap.npy` が保存されることを確認。
4. Sweep タブで 1) ROI check → 2) Threshold を実行し、分布と `threshold.json` を確認。
5. （任意）3) Start spectrum を開始し、出力が保存されることを確認。

## 典型的な手順 (実機あり)
1. DAQ mode real、Device を実機名 (Dev3 など) に設定し Connect。
2. Camera mode real にして Camera check（実機が開けるか確認）。
3. FG を使う場合は FG VISA を入れて FG Connect で確認後、Sweep の No FG を外す。
4. Sequence JSON を実験用に差し替え、周波数レンジと settle_s を調整し Start sweep。

## トラブルシュート簡易メモ
- FG 未接続エラー: No FG を一度チェックするか、VISA 文字列が正しいか確認。
- DAQ real で失敗: NI-DAQmx ドライバとデバイス名 (Dev3 等) を確認。dry なら動くかで切り分け。
- カメラ real で失敗: Camera mode を dry にして UI 動作を確認し、実機時はカメラ電源とドライバを確認。

## よく使うリソース文字列例
- USB0::0x0699::0x03A2::C040073::INSTR (device.yaml の tek_mdo_03A2)
- TCPIP::192.168.1.16::INSTR (device.yaml の tek_mso4034b_tcpip)

## 既知の動作メモ
- スイープ中は設定がロックされます。変更は停止後に行ってください。
- 更新間隔を短くしすぎると UI が重くなるので 1.0s 以上を推奨。
