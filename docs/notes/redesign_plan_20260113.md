# 再設計プラン（sweep + GUI）2026-01-13

## 目標（最優先）
- sweepの各プロセスをモジュール化し、dry/realで同じ処理フローを通す。
- 装置I/Oとデータ処理を分離し、dryで処理検証、realで取得検証ができるようにする。
- GUIをUIモジュールとコントローラに分離し、設定は上部タブに集約する。
- ログを強化し、特にカメラ接続のデバッグを容易にする。
- 無限ループや過負荷を避け、安全停止を保証する。

## コントローラーモジュールとは
コントローラーはワークフローの司令塔。状態遷移を管理し、ハードウェアと処理パイプラインを呼び出すが、UIやデバイスドライバを直接持たない。

責務:
- sweep状態機械（idle -> prepared -> roi -> threshold -> running -> stopping）を管理
- 入力検証と実行コンテキストの構築（装置設定、シーケンス仕様、出力先）
- ハードウェアI/F（DAQ/Camera/FG）と処理パイプラインの呼び出し
- UI向けイベントとログの発行
- 例外時の安全停止とクリーンアップ

非責務:
- Tkウィジェットの生成や直接操作
- nidaqmxやカメラSDKの直接呼び出し
- UIスレッド上の長時間ブロッキング処理

## GUIログ表示（下部パネル）
方針:
- GUI下部にログ表示パネル（Text + Scrollbar）を配置
- logging.Handlerでログをキューに積み、UIがafter()で取り出して追記
- GUI上でログレベル切替（INFO/DEBUG）とcameraログのフィルタを提供
- ログは必ずファイルにも保存し、重要イベントはGUIにも表示
- カメラの重要ログは常時出力し、verboseは詳細ログの追加にだけ使う（抑制しない）

カメラログの重点項目:
- 接続試行、デバイス列挙、選択ID、open/close成功、タイムアウト
- 露光/トリガ設定、ROI/subarray設定、取得開始/停止
- 失敗時は例外種別、スタックトレース、直前コマンド

## SequenceSpecの粒度（案A: 粗い同期）
- DOシーケンス（順序保証） + AO挿入位置（1回） + カメラTTLはDOに含める
- 729シャッターはAOで幅を保証、DOは順序を絶対保証
- ソフトタイミングの揺らぎは数ms以内を目標、実測ログで揺らぎを可視化
## シーケンス分離（意図と遅延の懸念）
現状の痛点:
- カメラ/DO/AOが混在した1つのシーケンスがdry/realの検証を難しくする

分離の考え方:
- 上位のSequenceSpec（意図）を定義
  - DaqAction: DO/AOのパルスとタイミング
  - CameraAction: 露光/取得イベントとタイミング
  - Sync marker: カメラとDAQの同期点

実行モデル:
- 実行前にSequenceSpecをハードウェア向けコマンドにコンパイル
- 実行中はパースをしない（タイミング影響を避ける）
- 解析処理は取得後に別パイプラインで行う

遅延への回答:
- 分離は設計上の整理であり、実行時は事前コンパイルで遅延を増やさない
- DO/AOのハードウェアタイミングは維持できる
- カメラトリガはTTLで同期し、制御経路は変えない

## 装置設定の一括管理
方針:
- device registryでDAQ/Camera/FG/ROI設定を一元管理
- GUI上部タブが唯一の入力元
- sweepやworkerは起動時に設定スナップショットを受け取る

固定するカメラトリガ設定:
- source=EXTERNAL
- connector=BNC
- polarity=POSITIVE（立ち上がり）
- active=EDGE
- mode=NORMAL

## Setupタブに残す「必要最小限」の設定項目（叩き台）
- DAQ
  - device（例: Dev1）
  - mode（real/dry）
- Camera
  - mode（real/dry）
  - exposure_ms
  - subarray（enable, x, y, width, height）
  - verbose（詳細ログのみ追加）
- FG
  - visa_resource
  - amp_mvpp
  - wave（例: SIN）
  - offset_v
  - start_hz
  - stop_hz
  - time_s
  - no_fg
- Sweepの実行に必要な共通項目
  - sequence_json_path

※ max_attempt / update_interval はデバッグ時のみUI表示にして、最終は定数化
※ dry_image_dir は削除

## エラーの可視化（案A）
- Diagnosticsタブに「最後のエラー」1件だけ表示
- 例外時のログファイルへのリンク（パス表示）を添付して追跡しやすくする
## GUI構成（案3: Setup / Run / Diagnostics）
- Setup: 装置設定を一括管理（DAQ/Camera/FG/Trigger/Subarray/Exposure）
- Run: Sequence / Sweep / Camera Snapの実行UI
- Diagnostics: ログ表示、接続チェック、カメラテスト、手動シャッタ（ツールに移動）

## 提案モジュール構成
- src/shutter_camera_trigger/
  - config/
    - device_registry.py   # 設定の読み書きと検証
  - hardware/
    - daq_iface.py         # DAQのreal/dry I/F
    - camera_iface.py      # Cameraのreal/dry I/F
  - pipeline/
    - roi_processing.py    # ROI/threshold/state判定
    - spectrum_calc.py     # スペクトル計算
  - sweep/
    - controller.py        # 状態機械とオーケストレーション
    - workflow.py          # 手順実装（I/Fとpipeline利用）
    - model.py             # dataclass定義
  - ui/
    - panels/              # 設定/ログ/実行パネルなど
    - controller_bridge.py # UIとcontrollerの接着

## API仕様（hardware層: 案B）
- DaqDevice
  - open(device: str) -> None
  - set_do(value: int) -> None
  - run_sequence_once(spec: SequenceSpec) -> None
  - all_off() -> None
  - close() -> None
- CameraDevice
  - open(cfg: CameraConfig) -> None
  - prime(timeout_s: float) -> None
  - capture(timeout_s: float) -> FrameResult
  - close() -> None
- FgDevice
  - open(resource: str) -> None
  - apply(cfg: FgConfig) -> None
  - close() -> None

## API仕様（controller層: 案B）
- SweepController
  - prepare(inputs: SweepInput) -> PrepareResult
  - roi_check(state: SweepState) -> RoiResult
  - threshold_check(state: SweepState) -> ThresholdResult
  - start(state: SweepState) -> None
  - stop(state: SweepState, *, reason: str) -> None
  - status(state: SweepState) -> SweepStatus

## API仕様（pipeline層: 案A）
- analyze_roi(frames: np.ndarray, roi: RoiSpec) -> RoiStats
- estimate_threshold(samples: np.ndarray) -> ThresholdResult
- classify_frame(frame: np.ndarray, th: float) -> bool
- summarize_sweep(results: list[ShotResult]) -> SpectrumResult

## データ型仕様（実装前に確定する叩き台）
- DeviceRegistry
  - daq: DaqConfig
  - camera: CameraConfig
  - fg: FgConfig
  - sweep_defaults: SweepDefaults
  - sequence_json_path: str
- DaqConfig
  - device: str
  - mode: Literal["real", "dry"]
- CameraConfig
  - mode: Literal["real", "dry"]
  - exposure_ms: float
  - subarray: SubarrayConfig
  - verbose: bool
  - trigger: TriggerConfig (固定: EXTERNAL/BNC/POSITIVE/EDGE/NORMAL)
- SubarrayConfig
  - enabled: bool
  - x: int
  - y: int
  - width: int
  - height: int
- TriggerConfig
  - source: Literal["EXTERNAL"]
  - connector: Literal["BNC"]
  - polarity: Literal["POSITIVE"]
  - active: Literal["EDGE"]
  - mode: Literal["NORMAL"]
- FgConfig
  - visa_resource: str
  - amp_mvpp: float
  - wave: Literal["SIN"]  # 固定
  - offset_v: float
  - start_hz: float
  - stop_hz: float
  - time_s: float
  - no_fg: bool
- SweepDefaults
  - n_target: int
  - max_attempt: int  # デバッグ中のみUI表示、安定後は定数化
  - settle_s: float
  - update_interval: float  # デバッグ中のみUI表示、安定後は定数化
- SequenceSpec（案A）
  - do_sequence: list[DoStep]
  - ao_insert_index: int  # -1は無効
  - ao_width_ms: float
  - ao_rate_hz: float  # 固定: 5000
  - ao_v_high: float  # 固定: 3.0
  - ao_v_low: float  # 固定: 0.0
- DoStep
  - value: int  # 4-bit DO
  - hold_s: float
- SweepInput
  - freqs: list[float]
  - sequence: SequenceSpec
  - n_target: int
  - max_attempt: int
  - settle_s: float
  - update_interval: float
  - daq: DaqConfig
  - camera: CameraConfig
  - fg: FgConfig
- FrameResult
  - frame: np.ndarray
  - roi: RoiSpec | None
  - meta: dict[str, Any]
- RoiSpec
  - x: int
  - y: int
  - width: int
  - height: int
- ThresholdResult
  - threshold: float
  - method: str
  - meta: dict[str, Any]
- ShotResult
  - freq_hz: float
  - bright: int
  - dark: int
- SpectrumResult
  - points: list[tuple[float, int, int]]
  - meta: dict[str, Any]

## IPC（QueueとSocketの違い）
- Queue: 同一PCのプロセス間通信に最適、設定が簡単
- Socket: リモート操作も可能、認証やポート管理が必要
- 推奨: まずQueueで統一し、必要ならSocketアダプタを追加

## 安全停止の統一仕様（案B）
- 停止要求/例外発生 -> ログ記録 -> カメラ停止 -> FG停止 -> DAQ ALL_OFF -> ワーカー終了
- すべての機器は拘束解除（close/stop/release）を実行してから終了する
- 実機は安全優先で、待ち時間は短く（タイムアウト厳守）
## 設定の固定化ルール（案A）
- デバッグ中はUI表示
- 安定後はconfigの定数に移動し、UIからは非表示
- 変更が必要ならconfigを編集
## 段階的プラン
フェーズ1: 設計とドキュメント
- device registryのスキーマを決める
- SequenceSpecとアクション型の仕様を書く
- controllerの状態機械とイベント仕様をまとめる

フェーズ2: ログ
- logging基盤とGUIログパネルの設計
- カメラログ項目の標準化
- cam verboseに依存せず重要ログを常時出す（詳細のみverboseで追加）

フェーズ3: ハードウェアI/F
- DAQ/Camera/FGのreal/dry I/Fを作る
- 既存実装の最小移植で動作維持

フェーズ4: sweepコントローラ
- sweepロジックをcontroller/workflowへ移動
- GUIはcontroller APIのみを呼ぶ

フェーズ5: シーケンス分離
- SequenceSpecとコンパイル機構を導入
- DAQ/Camera実行部を置換

## 検証プラン
- Dry:
  - 同じSequenceSpecで処理パイプラインを検証
- Real:
  - 接続と取得ログでデータが取れていることを確認
- 共通:
  - 例外時に安全停止（ALL_OFF）されること

## 一般的な設計・運用ルール（追加）
- 1ファイルは200?500行を目安（大きく超える場合は分割を検討）
- ファイル名と中身の責務が一致しない場合は必ず確認・相談してから進める
- printではなくloggingを使用し、例外時は安全停止を優先
- 重要な制御ループは停止条件を必ず持つ（無限ループ禁止）
- 出力先のルールを統一し、どの処理がどこに保存するかを明文化する
- dry -> real の順で検証する運用順序を守る

## ログ保存ルール（確定）
- 保存先: logs/YYYY-MM-DD/
- ファイル名:
  - app.log（GUI/統合ログ）
  - daq_worker.log
  - camera_worker.log
  - sweep.log
  - fg.log
- ログ行の先頭に timestamp + run_id（timestamp）を必ず含める

## 出力データの保存ルール（確定）
- すべて data/output/<run_type>/<timestamp>/ に保存する
- 共通ファイル:
  - config.json（使用した設定のスナップショット）
  - manifest.json（生成物一覧）
- run_type:
  - spectrum: スイープ本体
  - camera_snap: 単発スナップ
  - camera_check: 接続確認
  - shutter: DO/AOシーケンス単体

## Sweep結果の保存フォーマット（確定）
- shots.csv:
  - columns: freq_hz, shot_idx, bright, dark, t_s
- spectrum.csv:
  - columns: freq_hz, bright, dark
- plot.png:
  - スペクトルプロットのPNG

## GUIエラー表示テンプレ（確定）
- Diagnosticsの「最後のエラー」表示:
  - Error: {label} | {message}
  - Log: {log_path}

## 確認したいこと
- GUI上部タブに集約すべき設定項目は何か
- ログ保存先とローテーション方針
- 厳密なリアルタイムが必要なシーケンスはどれか

---
## DeviceRegistry スキーマ（フェーズ1・設計）
**目的**: GUIの入力を一元管理し、sweep/workerが起動時にスナップショットを受け取る。

### データ構造（案A）
- DeviceRegistry
  - daq: DaqConfig
  - camera: CameraConfig
  - fg: FgConfig
  - sweep_defaults: SweepDefaults
  - sequence_json_path: str
  - io_paths: IoPaths
  - ui: UiFlags
  - version: str
- DaqConfig
  - device: str
  - mode: Literal["real", "dry"]
- CameraConfig
  - mode: Literal["real", "dry"]
  - exposure_ms: float
  - subarray: SubarrayConfig
  - verbose: bool
  - trigger: TriggerConfig  # 固定: EXTERNAL/BNC/POSITIVE/EDGE/NORMAL
- SubarrayConfig
  - enabled: bool
  - x: int
  - y: int
  - width: int
  - height: int
- TriggerConfig
  - source: Literal["EXTERNAL"]
  - connector: Literal["BNC"]
  - polarity: Literal["POSITIVE"]
  - active: Literal["EDGE"]
  - mode: Literal["NORMAL"]
- FgConfig
  - visa_resource: str
  - amp_mvpp: float
  - wave: Literal["SIN"]
  - offset_v: float
  - start_hz: float
  - stop_hz: float
  - time_s: float
  - no_fg: bool
- SweepDefaults
  - n_target: int
  - max_attempt: int
  - settle_s: float
  - update_interval: float
- IoPaths
  - logs_root: str  # 例: logs/YYYY-MM-DD/
  - output_root: str  # 例: data/output/
- UiFlags
  - show_debug_fields: bool
  - camera_verbose_additional_only: bool

### 保存・読み込みルール
- 保存先: `config/device_registry.json`
- 起動時にロード -> UIへ反映 -> 変更時に保存
- sweep/worker起動時にスナップショットを渡す（以降はUIを参照しない）
- `version` でマイグレーションを管理（例: "1.0"）

### バリデーションの方針
- `mode` は strict に "real"/"dry"
- `exposure_ms` は 0 より大きい
- `subarray` は enabled の時に x,y,width,height を必須
- `sequence_json_path` は存在確認（UIで警告、起動前は必須）
- `logs_root`/`output_root` は実行時に自動作成

### 例（device_registry.json の最小構成）
```json
{
  "version": "1.0",
  "daq": { "device": "Dev1", "mode": "dry" },
  "camera": {
    "mode": "dry",
    "exposure_ms": 1.0,
    "subarray": { "enabled": false, "x": 0, "y": 0, "width": 0, "height": 0 },
    "verbose": false,
    "trigger": {
      "source": "EXTERNAL",
      "connector": "BNC",
      "polarity": "POSITIVE",
      "active": "EDGE",
      "mode": "NORMAL"
    }
  },
  "fg": {
    "visa_resource": "",
    "amp_mvpp": 790.0,
    "wave": "SIN",
    "offset_v": 0.0,
    "start_hz": 1000.0,
    "stop_hz": 10000.0,
    "time_s": 1.0,
    "no_fg": true
  },
  "sweep_defaults": {
    "n_target": 50,
    "max_attempt": 100,
    "settle_s": 0.02,
    "update_interval": 1.0
  },
  "sequence_json_path": "src/shutter_camera_trigger/sequence_examples/minimal_sequence.json",
  "io_paths": { "logs_root": "logs", "output_root": "data/output" },
  "ui": { "show_debug_fields": true, "camera_verbose_additional_only": true }
}
```

---
## SequenceSpec とアクション型（フェーズ1・設計）
**目的**: 意図(SequenceSpec)と実行コマンドを分離し、dry/realで同一フローを通す。

### SequenceSpec（上位仕様）
- SequenceSpec
  - do_sequence: list[DoStep]
  - ao_insert_index: int  # -1は無効
  - ao_width_ms: float
  - ao_rate_hz: float  # 固定: 5000
  - ao_v_high: float  # 固定: 3.0
  - ao_v_low: float  # 固定: 0.0
  - camera_actions: list[CameraAction]  # 任意
  - sync_markers: list[SyncMarker]  # 任意
- DoStep
  - value: int  # 4-bit DO
  - hold_s: float
- CameraAction
  - t_s: float
  - kind: Literal["expose", "capture"]
  - meta: dict[str, Any]
- SyncMarker
  - t_s: float
  - label: str

### コンパイル仕様（意図->実行）
- 事前コンパイル: 実行前にSequenceSpecをハードウェア向けコマンドへ変換
- 実行中はパースしない（タイミング影響を避ける）
- Camera TTLはDOに含める（SequenceSpecのdo_sequenceに含める）

### JSONスキーマ（案A）
- sequence_json (SequenceSpec)
  - do_sequence: list[{value:int, hold_s:float}]
  - ao_insert_index: int
  - ao_width_ms: float
  - ao_rate_hz: float
  - ao_v_high: float
  - ao_v_low: float
  - camera_actions: list[{t_s:float, kind:str, meta:dict}]  # optional
  - sync_markers: list[{t_s:float, label:str}]  # optional

### 例（sequence_json）
```json
{
  "do_sequence": [
    { "value": 1, "hold_s": 0.002 },
    { "value": 5, "hold_s": 0.002 },
    { "value": 1, "hold_s": 0.002 }
  ],
  "ao_insert_index": 1,
  "ao_width_ms": 0.5,
  "ao_rate_hz": 5000.0,
  "ao_v_high": 3.0,
  "ao_v_low": 0.0,
  "camera_actions": [
    { "t_s": 0.002, "kind": "capture", "meta": { "tag": "roi_bootstrap" } }
  ],
  "sync_markers": [
    { "t_s": 0.002, "label": "cam_ttl" }
  ]
}
```

### 実行コマンド型（DAQ/Camera向け）
- DaqSequenceCommand
  - do_sequence: list[tuple[int, float]]
  - ao_insert_index: int
  - ao_width_ms: float
  - ao_rate_hz: float
  - ao_v_high: float
  - ao_v_low: float
- CameraCommand
  - kind: Literal["prime", "capture", "close"]
  - timeout_s: float
  - meta: dict[str, Any]

### バリデーションの方針
- do_sequenceは空不可
- hold_sは0より大きい
- ao_insert_indexは-1..len(do_sequence)-1
- ao_width_msは0以上
- ao_rate_hz, ao_v_high, ao_v_lowは固定値のみ許可（UI非表示）

---
## SweepController 状態機械とイベント仕様（フェーズ1・設計）
**目的**: 状態遷移を明確化し、安全停止とUI通知を一貫させる。

### 状態定義（SweepPhase）
- idle: 初期状態。ワーカー未起動。
- prepared: セッション生成完了。ROIチェック可能。
- roi_done: ROI確定済み。Threshold可能。
- threshold_done: しきい値確定済み。Sweep開始可能。
- running: sweep実行中。
- stopping: 停止処理中（安全停止）。
- error: 例外発生。停止済み/要復帰。

### 遷移（現行実装）
- idle -> prepared: prepare_session()
- prepared -> roi_done: roi_check()でROI検出
- roi_done -> threshold_done: threshold_check()で適用
- threshold_done -> running: start_sweep()
- running -> stopping: stop_sweep()または例外
- stopping -> idle: cleanup完了
- any -> error: 例外発生（ログ記録 + 安全停止）

### ガード（SweepController）
- prepare_session: idle/roi_done/threshold_done/prepared は許可、running/stopping/errorは拒否
- roi_check: prepared/roi_done/threshold_doneのみ許可
- threshold_check: prepared/roi_done/threshold_doneのみ許可（ROI未設定はエラー）
- start_sweep: threshold_doneのみ許可
- stop_sweep: idle以外は許可

### イベント（UI向け）
- on_status(text): 進捗/状態ラベル
- on_warning(text): FG/カメラの注意
- on_error(label, message, log_path): Diagnostics用
- on_state_change(prev, next)
- on_plot_reset()
- on_plot_update(step_idx, freq_hz, bright, dark)
- on_worker_ready(name)
- on_worker_stopped(name)

### 例外時の安全停止（統一）
- 例外発生 -> on_error -> カメラ停止 -> FG停止 -> DAQ all_off -> ワーカー終了
- stopping中は新規コマンド受付禁止

### UI操作の禁止条件
- running/stopping中は設定編集を無効化
- prepared/roi_done/threshold_done中にモード変更は警告表示





---
## 現状UI/設定との差分整理（フェーズ1）
### UIに存在するがDeviceRegistryに未定義
- camera_trigger_delay_s: 現在トップバーにDelay(s)があるが、設計案では固定トリガのみで未定義。
- dry_image_dir: 現在トップバーに入力があるが、設計案では削除方針。
- ao_width(ms)入力: トップバーのAO widthはSequenceSpec側へ統合する前提で未定義。
- Sequenceタブの生テキスト編集: SequenceSpec(JSON)一本化の方針とは未整合。

### DeviceRegistryにあるがUIに未露出
- io_paths.logs_root / output_root: UIには未表示（configに固定でOK）。
- ui.show_debug_fields / camera_verbose_additional_only: 既存UIは常時表示。

### UI内で重複/分散している設定
- DAQ device/mode: Top barのdevice_var/device_mode_varとSweepタブのsw_device/sw_daq_modeが重複。
- FG設定: Top barのFG VISA/ampとSweepタブのsw_visa/sw_fg_amp_mvppが重複。
- Camera mode/exposure: Top barにあるがSweep/Cameraタブ側で参照（統合は未完）。
- Sequence JSON: Sweepタブでパス指定、Sequenceタブはテキスト編集（一本化が必要）。

### 移行の指針（暫定）
- Setupタブ(Top)を唯一の入力元にし、Sweep/Cameraは参照専用。
- dry_image_dirとcamera_trigger_delayは設計確定まで非表示またはDebug扱い。
- SequenceSpec(JSON)に統合し、Sequenceタブは閲覧/検証専用に移行。

---
## ログ基盤とGUIログパネル（フェーズ2・設計）
**目的**: 重要イベントを常時記録し、GUIから迅速に確認できるようにする。

### ログ出力ルール（確定準拠）
- 保存先: `logs/YYYY-MM-DD/`
- ファイル名:
  - app.log（GUI/統合ログ）
  - daq_worker.log
  - camera_worker.log
  - sweep.log
  - fg.log
- ログ行の先頭に `timestamp + run_id` を必ず含める

### run_id
- 生成: `YYYYMMDD_HHMMSS` 形式
- 生成タイミング: GUI起動時に作成し、sweep/camera_check/camera_snapで共有
- GUIとworkerのログは同一run_idを引き回す

### 実装方針（案A）
- logging基盤は以下の2系統で構成
  - FileHandler: 各ログファイルへ出力
  - QueueHandler/QueueListener: GUI表示用にメッセージをキューへ転送
- GUIの下部パネルにText+Scrollbarを配置し、`after()`でキューを吸い上げて追記
- ログレベル切替（INFO/DEBUG）と cameraフィルタを提供
- `camera_verbose` は詳細ログの追加のみ（重要ログは常時出す）

### GUIログパネル仕様
- Panel: 下部の固定領域
- Controls:
  - Level: INFO/DEBUG
  - Filter: camera only / all
- 表示: 直近N行（例: 2000行）でリングバッファ運用
- スクロールは自動追従（ユーザー操作時は追従解除）

### 例外時のUI通知
- 例外検知時:
  - `on_error(label, message, log_path)` を発火
  - Diagnosticsタブの「最後のエラー」を更新
  - ログファイルへのパスを表示

### カメラ重要ログ（常時）
- 接続試行/列挙/選択ID/成功/タイムアウト
- 露光/トリガ/ROI設定/取得開始/停止
- 失敗時の例外種別/スタックトレース/直前コマンド

### 実装タスク（フェーズ2）
- logging初期化ユーティリティ（run_id/handlersを統一）
- GUIログパネル実装（Text+Queue）
- worker側にrun_id伝搬（cfgに追加）
- 重要ログの追加（camera/daq/fg/sweep）

---
## 進捗メモ
- 2026-01-13: GUI構成をSetup/Run/Diagnosticsに再整理し、導線を統合。
- 2026-01-13: ログ/Diagnosticsの表示・操作（履歴/フィルタ/コピー）を整備。
- 2026-01-13: Sequence JSONを唯一の入力元に統一、関連タブは参照表示へ。
- 2026-01-13: device_registryの読み書き配線とテンプレート追加。
- 2026-01-13: hardware/pipelineの骨格とAdapter(DAQ/Camera/FG)を追加。
- 2026-01-14: SweepのEvents/IO/Deps境界とPhase状態機械を導入。
- 2026-01-14: SweepのROI/Threshold/Spectrumをworkflowへ移行。
- 2026-01-14: output_root/logs_rootの解決を統一。
- 2026-01-14: Sequence JSONのcamera_actions/sync_markers対応を追加。
- 2026-01-14: sweep config.jsonへupdate_interval/camera_actions/sync_markersを保存。

- 2026-01-14: run_shutter_sequenceにmanifest.json出力を追加。
- 2026-01-14: camera_check/camera_snapの出力にmanifest.jsonを追加。
- 2026-01-14: SequenceSpecのデータ型とコンパイルヘルパーを追加。
- 2026-01-14: SequenceSpecコンパイルをrun_spectrum/sequence実行に適用。
- 2026-01-14: run_spectrumでcamera_actionsのタイミング送信をサポート。
- 2026-01-14: sweepのspectrumでcamera_actionsのタイミング送信をサポート。
- 2026-01-14: sweep出力にcamera_actions/sync_markersのJSONを保存。
## Adapter移行状況（暫定）
### DaqClientDevice（GUI側）
- camera_tab: snap/TTL/priming
- sequence/controller: run_sequence_once, set_do
- manual_actions: set_do
- daq/controller: connect/disconnect set_do

### DaqQueueDevice（worker/queue側）
- sweep: roi_bootstrap/roi_check/threshold/spectrum/stop/priming
- camera_check: DAQ priming

### CameraWorkerDevice
- camera_check/camera_snap

### RigolFgDevice
- FG connect/disconnect
