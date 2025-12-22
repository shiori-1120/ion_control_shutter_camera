# ion_control_shutter_camera

## ディレクトリ構成と役割

### ルート

- `README.md`: 本ドキュメント。
- `src/`: 主要ソースコード。
- `data/`: 入出力データ（コードで固定の出力先）。
	- `input/`: 入力データ置き場。
	- `output/`: 出力（機能別サブフォルダ）。
		- `camera/`: 撮影結果（画像/派生データ）。
		- `waveform/`: 波形CSV/レポート。
		- `shutter/`: シャッター実行ログ等。
- `logs/`: 実行時ログ（必要に応じて機能別に記録）。

### 設定・共通ライブラリ

- `src/config/device.yaml`: 計測器のVISA/TCPIPプロファイルと共通パラメータ。
	- `visa_profiles`: 機器ごとの接続文字列（例: Tektronix/Rigol/Agilent）。
	- `current_profile`: 使用中プロファイル名（切替はここを編集）。
- `src/lib/`: 共通ライブラリ。
	- `utils/config.py`: YAML読込・保存、プロファイル解決ヘルパ。
	- `lib/dcam.py`, `lib/dcamapi4.py`: カメラDCAM関連。
	- `lib/device/controlDevice.py`, `lib/caio.py`: デバイス制御ユーティリティ。
	- `lib/CommonFunction.py`: 汎用関数群。

### カメラ

- `src/camera/initial_preparation.py`: 撮影前の初期セットアップ。
- `src/camera/camera_loop.py`: 撮影ループ（長時間/連続撮影）。
- `src/camera/main_take-one-shot.py`: 1ショット撮影のエントリ。
- `src/camera/tif_to_npy.py`: TIF→NPY変換。
- `src/camera/visualize_tif.py`, `visualize_npy.py`: 可視化ユーティリティ。
- `src/camera/ion_state_detection.py`: イオン状態検出処理。
- 出力先: `data/output/camera`（コードで固定）。

### 波形

- `src/waveform/check_device.py`: 接続可能なVISA/TCPIPリソースの列挙とIDN確認。
- `src/waveform/acquire_waveform.py`: オシロから波形取得→CSV保存→プロット。
	- 設定読込: `src/config/device.yaml`（`current_profile`で機器切替）。
	- 出力先: `data/output/waveform`（コードで固定）。
- `src/waveform/analyze_waveform.py`, `plot_analyze_csv.py`: 解析・可視化。
- `src/waveform/get_wav.py`, `get_wav_binary.py`: 波形取得ユーティリティ。

### シャッター

- `src/shutter_camera_trigger/`: シャッター制御スクリプト群。
	- `shutter_ctrl_with_int.py`: NI-DAQのDO制御（ライン単位のON/OFF、新規シーケンス実装の土台）。
	- `test_all_shutters.py`, `test_all_shutters_multi.py`, `test_ao_shutter.py`: 動作検証・試験用。
	- `archive/`: 旧版スクリプトの保管（履歴参照用）。
- シーケンス定義: Pythonの `.py` 内で2進数表現により明示的に定義。
- 出力先: `data/output/shutter`（コードで固定）。

## 運用のポイント

- 計測機器の切替は `src/config/device.yaml` の `current_profile` を編集。
- 物理機器が無い場合のデバッグは `pyvisa-sim` を導入（必要に応じて）。
- すべての出力は `data/output/*` に保存されるため、時刻付きフォルダで整理・追跡可能。

## Spectrum runner（dry bring-up）

関数発生器（FG）なし・カメラなしでも、同一PCのrunner/worker構成が一周することを確認できます。

```powershell
Set-Location "c:\Users\shiori\Desktop\ion_control_shutter_camera"

python -m src.runner.run_spectrum `
	--camera-mode dry `
	--no-fg `
	--daq-mode dry `
	--sequence-json src/shutter_camera_trigger/sequence_examples/minimal_sequence.json `
	--freq-start 80e6 --freq-stop 82e6 --freq-step 0.5e6 `
	--n-target 50
```

出力は `data/output/spectrum/<timestamp>/` に保存されます。

## 依存環境（例: uv）

```powershell
Set-Location "c:\\Users\\shiori\\Desktop\\ion_control_shutter_camera"
uv venv myenv
. .\\myenv\\Scripts\\Activate.ps1
uv pip install pyyaml numpy pandas matplotlib pyvisa
# 必要に応じて
uv pip install pyvisa-py pyvisa-sim nidaqmx
```

プロジェクトに関する質問や改善提案があれば、IssueやREADMEのPRでお知らせください。
conda activate myenv

python --version 
Pytyon 3.13.1


