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
- Sweep タブ: 周波数を変えながらシーケンスを 1 ショットずつ実行し、カメラ判定の p_bright をリアルタイム表示・保存。

## 上部バー
- Device: NI-DAQ デバイス名 (例: Dev3)。
- DAQ mode: dry=模擬、real=実機 NI-DAQ。
- AO width (ms): Sequence タブ用の AO パルス幅。
- Connect / Disconnect: DAQ ワーカーの開始/終了。Disconnect 時に DO を全 OFF。
- FG VISA / FG Connect: VISA 文字列を入れて接続確認。接続中は FG Disconnect で閉じる。スイープでも同じ文字列を共有します。
- Camera mode / Camera check: カメラワーカーの dry/real を選び、接続確認だけ行う。dry のときは下の Dry images でフォルダ指定可能。
- Dry images (dry cam): dry カメラ用の画像フォルダ。bright* / dark* のファイル（png/jpg/bmp/tif/npy）を置くと、取得要求ごとにランダムで使います（無ければ従来の乱数合成）。
- ROI bootstrap: スイープ開始時に 729 を OFF のままカメラ TTL だけを数パルス送信（既定: 2 ms ON/2 ms OFF を最大 5 回）。カメラ応答が得られなければスイープを中断。

## Sweep タブのパラメータ
- Freq start/stop/step (Hz): スイープ周波数レンジと刻み (例: 80e6 → 82e6 を 0.5e6 刻み)。
- n_target: 周波数ごとの目標ショット数（カメラ判定成功件数）。
- max_attempt: 試行の上限（タイムアウト含む）。
- settle_s: 周波数変更後に待つ秒数。
- Sequence JSON: DO シーケンス定義ファイルへのパス。`...` で選択。
- DAQ mode / Camera mode / DAQ device: スイープ時に使うモードとデバイス名。
- FG VISA: 関数発生器の VISA リソース文字列 (例: USB0::0x0699::0x03A2::C040073::INSTR)。上部バーと共有。
- No FG: チェック時は FG 制御をスキップ。外すと FG を pyVISA で制御。
- Update interval (s): グラフ・ステータスの更新間隔。負荷を下げたいときは大きめに。
- Start sweep / Stop: スイープ開始・停止。開始中は入力がロックされます。
- グラフ: 周波数ごとの p_bright を表示。完了時に spectrum.png を出力。
- ROI ブートストラップ: スイープ開始前にカメラ TTL パルスだけを流し、応答が得られない場合はエラーで停止。

## 出力
- data/output/spectrum/<timestamp>/shots.csv, spectrum.csv, spectrum.png を保存。
- config.json に実行時設定（ROI パラメータ含む）を記録。

## モードと接続
- DAQ dry: NI-DAQ なしで動作確認。
- Camera dry: 画像を使わず疑似判定。
- FG: No FG を外し、FG VISA に文字列を入れると接続を試みます。機種は src/lib/instruments/rigol_dg.py に準拠。

## 典型的な手順 (実機なしの dry)
1. 起動後、DAQ mode を dry にして Connect。
2. （任意）Dry images に bright*/dark* 画像を置いたフォルダを指定して Camera check。
3. Sweep タブで周波数レンジを設定し、No FG にチェックのまま Start sweep。
4. グラフが更新されることを確認。

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
