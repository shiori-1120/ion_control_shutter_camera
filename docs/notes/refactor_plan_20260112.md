# Refactor計画（camera / shutter_gui）2026-01-12

## 0. この計画の目的（最重要）
この計画の最優先ゴールは「あなたが自力でコードの全体像を把握し、実験運用中に“AIへ丸投げ”せずに、どこを直せばよいか自分で判断できる状態」になることです。

あなたの要望（原文）:

> 今からやりたいのはこのプロジェクトの特にcamera, shutter_gui.pyのプログラムのリファクタリングです。  
> AIにコードを書かせていたのでどこに何が開いてあるかわかりません。また、shutter_guiはgui以外の機能がほとんどであり、コード数が非常に多いため分割しディレクトリ構成から作り直すべきだと思います。目標は私がコードの全体像を把握して、実験でコードの実行を行ったときにAIにデバッグを丸投げするのではなく、自分でどこを直したらいいかわかるまで理解すること、およびその理解を助けるためのリファクタリングです

このために、この計画は「動く状態を維持しつつ」「理解のための境界（責務）を切る」ことを第一原則にします。

---

## 0.1 進捗メモ（2026-01-13 時点）
「巨大な `shutter_gui.py` を壊さずに理解できる形へ」進めるために、まず **起動/停止・IPC・Sweep開始手順** の境界を先に切りました。

（補足）この一連のリファクタは PR #5 として `main` にマージ済みで、`origin/main` の先頭は `a0fec59` です。

### 2026-01-13 追加引継ぎ（ROI checkから進まない問題の修正）
- 【発生事象】GUIでSweep開始後、"ready step 1 roi check" から進まずフリーズする問題が発生。
- 【原因】sweep/session_ready.py の prepare_sweep_session が workers/queue を返さず、GUI側で self._sw_queues などにセットされていなかった。
- 【修正内容】prepare_sweep_session の返り値に workers を追加し、shutter_gui.py 側で self._sw_queues, self._sw_procs に正しくセットするよう修正。
- 【検証】py_compile・GUIスモークでエラーなし、ROI check以降のステップも進行することを確認。
- 【今後の注意】この種の「責務分離時の参照漏れ」は、返り値・状態の受け渡し設計を都度見直すこと。


### 直近でできたこと（成果物）
- worker 起動/停止の境界化
   - [src/shutter_camera_trigger/workers/daq_worker_process.py](../../src/shutter_camera_trigger/workers/daq_worker_process.py)
   - [src/shutter_camera_trigger/workers/camera_worker_process.py](../../src/shutter_camera_trigger/workers/camera_worker_process.py)
- Queue IPC のクライアント化（request/response をロックで直列化）
   - [src/shutter_camera_trigger/clients/daq_client.py](../../src/shutter_camera_trigger/clients/daq_client.py)
   - [src/shutter_camera_trigger/clients/camera_client.py](../../src/shutter_camera_trigger/clients/camera_client.py)
- Sweep の「開始準備」周りを段階的に外出し（GUI 側を“目次”に寄せる）
   - [src/shutter_camera_trigger/sweep/session_workers.py](../../src/shutter_camera_trigger/sweep/session_workers.py)（worker/queue 生成）
   - [src/shutter_camera_trigger/sweep/session_config.py](../../src/shutter_camera_trigger/sweep/session_config.py)（config.json 生成、session dict 構築）
   - [src/shutter_camera_trigger/sweep/session_parse.py](../../src/shutter_camera_trigger/sweep/session_parse.py)（周波数式パース、sequence_json 読み出し）
   - [src/shutter_camera_trigger/sweep/session_start.py](../../src/shutter_camera_trigger/sweep/session_start.py)（DAQ ready→camera start→priming→camera ready）
   - [src/shutter_camera_trigger/sweep/roi_bootstrap.py](../../src/shutter_camera_trigger/sweep/roi_bootstrap.py)（TTL でカメラ応答確認）
   - [src/shutter_camera_trigger/sweep/stages.py](../../src/shutter_camera_trigger/sweep/stages.py)（GUI 依存の薄い“ステージ”ラッパ）

### いま残っている（次に切り出す候補）
- Sweep の本体（ROI check / Threshold / Spectrum 実行）の状態機械と保存・描画が、まだ [src/shutter_camera_trigger/shutter_gui.py](../../src/shutter_camera_trigger/shutter_gui.py) に残っている
- 「ROI bootstrap 成功後〜Session ready」の UI 更新/フラグ更新/失敗時 cleanup も、まだ GUI 側に残る（次に薄くする価値が高い）

---

## 0.2 別チャット引き継ぎメモ（そのまま貼れる）

### 2026-01-13 引き継ぎ（Gitの状態）
- PR #5 を `main` にマージ済み（`origin/main` = `a0fec59`）
- 現在の作業ブランチ: `resume/fd0984e`（`fd0984e docs: copilot手順の追記`）
- 作業ツリー: `src/shutter_camera_trigger/shutter_gui.py` が未コミット変更あり（続きの作業はここから）
- stash が残っている（必要なら適用/整理）:
   - `stash@{0}`: `wip: before restoring fd0984e`（復元直前に退避した `.gitignore` 変更）
   - `stash@{1}`: `backup: before reset to origin (2026-01-13)`（このstashは apply 済みだが、popしていないので残っている）
   - `stash@{2}`: `sequence: wip: debug label_bright logging + sweep analyzer`

（再開の最短コマンド）
```powershell
cd C:\Users\shiori\Desktop\ion_control_shutter_camera
git status -sb
git checkout resume/fd0984e
# もし .gitignore の退避も必要なら:
# git stash apply "stash@{0}"
```

（ローカルの `main` は behind 表示になり得るので、必要なら更新する）
```powershell
git checkout main
git pull
```

### 作業の目的
- 目的は「動作維持しつつ、理解できる境界（責務）で分割」すること
- Big-bang を避けて、毎回 import/py_compile と dry スモークで壊していないことを確認する

### 現在地（何がどこにあるか）
- 入口（GUI）: [src/shutter_camera_trigger/shutter_gui.py](../../src/shutter_camera_trigger/shutter_gui.py)
- Sweep の開始準備（外出し済み）: [src/shutter_camera_trigger/sweep/](../../src/shutter_camera_trigger/sweep/)
   - workers/queue 生成: [src/shutter_camera_trigger/sweep/session_workers.py](../../src/shutter_camera_trigger/sweep/session_workers.py)
   - config.json と session dict: [src/shutter_camera_trigger/sweep/session_config.py](../../src/shutter_camera_trigger/sweep/session_config.py)
   - 周波数式/sequence_json 読み: [src/shutter_camera_trigger/sweep/session_parse.py](../../src/shutter_camera_trigger/sweep/session_parse.py)
   - ready待ち/priming/起動: [src/shutter_camera_trigger/sweep/session_start.py](../../src/shutter_camera_trigger/sweep/session_start.py)
   - ROI bootstrap（TTLでカメラ応答確認）: [src/shutter_camera_trigger/sweep/roi_bootstrap.py](../../src/shutter_camera_trigger/sweep/roi_bootstrap.py)
   - GUI側の薄いステージラッパ: [src/shutter_camera_trigger/sweep/stages.py](../../src/shutter_camera_trigger/sweep/stages.py)
- worker 起動/停止（境界化済み）: [src/shutter_camera_trigger/workers/](../../src/shutter_camera_trigger/workers/)
- IPC クライアント（lock付き）: [src/shutter_camera_trigger/clients/](../../src/shutter_camera_trigger/clients/)

### 直近の変更点（2026-01-13）
- ROI bootstrap の GUI ラッパを消して、sweep 側のステージ関数に寄せた
   - GUI は `run_roi_bootstrap_stage(...)` を呼ぶだけになっている

### 次にやると効果が高いこと（推奨順）
1) 「ROI bootstrap 成功後〜Session ready」の UI/フラグ/cleanup を `sweep/` 側へ寄せて、GUIを薄くする
    - 例: `sweep/session_ready.py` や `sweep/stages.py` にまとめて、GUI側は bool/例外で分岐だけにする
2) ROI check / Threshold / Spectrum 実行の“状態機械”を `sweep/controller.py` 的なモジュールへ寄せる
    - GUI側は「ボタン→controller呼び出し」「描画の呼び出し」「messagebox」だけに近づける
3) sweep 実行中の Queue 直叩きを `DaqClient` / `CameraClient` 経由に置き換える（必要箇所だけ）

### すぐ使える検証（myenv 前提）
- import/py_compile:
   - `C:\Users\shiori\Desktop\ion_control_shutter_camera\myenv\Scripts\python.exe -c "import py_compile; import src.shutter_camera_trigger.shutter_gui as g; py_compile.compile(g.__file__, doraise=True); import src.shutter_camera_trigger.sweep.stages as s; py_compile.compile(s.__file__, doraise=True); print('OK')"`
- GUI 起動:
   - `C:\Users\shiori\Desktop\ion_control_shutter_camera\myenv\Scripts\python.exe -m src.shutter_camera_trigger.shutter_gui`

### 注意点（ハマりどころ）
- GUI（Tk）側から Queue get を直に待つと固まりやすいので、UIポンプ/timeout を常に意識する
- Sweep の「開始準備」は分離が進んだが、ROI/Threshold/Spectrum の本体がまだ GUI に残るので、ここを触るときは小さく切って都度動作確認する

---

## 1. 現状の把握（入口と責務）

### 1.1 主要な入口（まずここだけ覚える）
- GUI: [src/shutter_camera_trigger/shutter_gui.py](../../src/shutter_camera_trigger/shutter_gui.py)
- Runner（dry bring-up）: [src/runner/run_spectrum.py](../../src/runner/run_spectrum.py)
- Camera worker: [src/camera/ion_state_worker.py](../../src/camera/ion_state_worker.py)

### 1.2 shutter_gui.py の「中に入っているもの」
`App` クラスに以下が混在しています（理解しづらくなる原因）:
- Tk/ttk のUI生成
- DAQ worker の起動/停止/IPC（Queue）
- Camera worker の起動/停止/IPC
- Sequence（DO→AO→DO）パース/描画/実行
- Sweep（ROI→Threshold→Spectrum）の状態遷移と結果保存
- 画像表示（matplotlib）
- 設定の保存/復元

このため、「GUI」ではなく「制御アプリ全部」が 1ファイルに入っている状態です。

### 1.3 camera 側の特徴
- `src/camera/lib/` が大きく、DCAM/デバイス依存と解析ロジックが混在しがち
- `ion_state_worker.py` は実験運用の中心（GUI/runner から叩かれる）

---

## 2. リファクタリングの原則（壊さない＋理解を作る）

### 2.1 ルール
- **Big-bang禁止**: 一気にディレクトリ作り直しはしない。小さく切り出して都度importが通る状態を維持。
- **動作は同じ**: まずは「整理（移動/抽出）」で挙動を変えない。挙動変更は別PR/別ステップ。
- **責務で切る**: 「UI」「状態機械」「I/O（DAQ/Camera/FG）」「解析（ROI/threshold）」「永続化（config/log/output）」。
- **入口を固定**: 入口の `python -m ...` は当面変えない（変える場合は互換入口を残す）。

### 2.2 “理解のため”に必ず作るもの
- 依存関係図（簡易でOK）
- 用語集（ROI, tau, S_norm, “dry/real”, “prime” など）
- 状態遷移図（Sweep: idle→roi→threshold→running→stopped）

---

## 3. 進め方（フェーズ分割）

### フェーズ0: 現状把握の足場（完了）
すでに “巨大ファイルの中の汎用ヘルパ” を切り出して、読むべき範囲を減らしました。
- 追加: [src/shutter_camera_trigger/gui_support/](../../src/shutter_camera_trigger/gui_support/)

**理解チェック（あなた向け）**
- Q0-1: `shutter_gui.py` は「GUI」だけでなく何をしている？（3つ挙げる）
- Q0-2: “入口” はどの3ファイル？

### フェーズ1: shutter_gui を「UI」と「アプリ制御」に分割
目的: `App(tk.Tk)` を“薄く”して読みやすくする。

**現状（2026-01-13）**
- まだ `App` は巨大だが、worker 起動/停止と IPC の一部は外部モジュールへ移動済み
- 次の狙いは「GUIの状態更新（ボタン/ラベル/フラグ）」と「Sweep手順」をさらに分離して、`_sw_prepare_session` を“目次”へ近づける

**やること**
1) `App` の責務を2層に分ける
   - UI層: Tk/ttk widget, messagebox, after, layout
   - アプリ層: 接続状態、Sequence/Sweepの状態機械、worker制御
2) `App` のメソッド群を、まず「まとまりごと」にクラス分割（ファイル分割は後）
   - 例: `DaqClient`, `CameraClient`, `SequenceController`, `SweepController`, `PrefsStore`

**成果物（あなたが読むもの）**
- `App` は「画面部品の作成」と「controller呼び出し」だけになる
- “どの処理がどこにあるか” がクラス名でわかる

**理解チェック（毎回の停止点）**
- Q1-1: 「UI層」と「アプリ層」の境界はどこ？（具体的な関数名で1つ）
- Q1-2: Sweepはどんな状態遷移？（矢印で5状態くらい）

### フェーズ2: Sweep（ROI/Threshold/Spectrum）を独立モジュールに
目的: 一番複雑な“手順物”を `shutter_gui.py` から外し、読む単位を小さくする。

**やること**
- `src/shutter_camera_trigger/sweep/` を作り、
  - `model.py`（設定/結果のdataclass）
  - `controller.py`（状態遷移）
  - `io.py`（output保存、CSV/JSON）
  - `plot.py`（matplotlib描画）
  に分解
- GUIは `SweepController` のAPIだけ呼ぶ

**現状（2026-01-13）**
- Sweep開始準備（workers/config/入力パース/ready待ち/priming/roi_bootstrap）は `sweep/` に移動済み
- ROI check / Threshold / Spectrum 実行（ループ・保存・描画）は GUI 側に残っている（次の分割対象）

**理解チェック**
- Q2-1: Threshold（tau）は「何から」推定して「何に」使う？
- Q2-2: ROI checkは “なぜ” 分布プロットをしない？

### フェーズ3: worker IPC を“クライアント”として抽象化
目的: Queueのput/getが散らばると理解が崩れるので、通信を1箇所に集約。

**やること**
- `DaqClient.request(cmd)->resp` のみで触れるようにする（timeout/エラー整形も集約）
- `CameraClient` も同様
- “プロセス掃除” も `ProcessManager` 的にまとめる（PID記録/cleanup）

**現状（2026-01-13）**
- `DaqClient` / `CameraClient` は導入済み
- Sweep準備以外（ROI/Threshold/Spectrum実行中）の Queue 直叩きが残っているので、必要箇所だけ段階的に置き換える

**理解チェック**
- Q3-1: DAQ worker へのコマンドは大きく何種類？（set_do / run_sequence_once …）
- Q3-2: request/response をロックしている理由は？

### フェーズ4: camera 側の責務整理（解析 vs デバイス）
目的: 実験でトラブルが起きる箇所（ドライバ/トリガ/ROI/threshold）をすぐ辿れるようにする。

**やること（優先順）**
1) `ion_state_worker.py` の “受け付けコマンド一覧” を先頭にまとめる（doc + dispatch整理）
2) `src/camera/lib/` を
   - device（DCAM等のI/O）
   - processing（ROI, threshold, profiles, image ops）
   に近づける
3) trigger設定（env fallback / GUI設定 / cam_cfg）を一本化する

**理解チェック**
- Q4-1: 「カメラが見つからない」問題はどこを見る？（ファイル名で2つ）
- Q4-2: trigger設定は“今”どこから来る？（GUI→worker→device）

### フェーズ5: スモークテスト（dry）を固定化
目的: リファクタ後でも「最低限は動く」をあなたが自分で確認できるようにする。

**やること**
- READMEの dry bring-up コマンドを“この計画に紐づく”形で整備
- `python -m ...` の動作確認手順を固定化（GUI起動/runner起動/ログの場所）

**現状（2026-01-13）**
- `myenv` で import と py_compile を通す確認は継続して実施
- dry のスモーク（camera trigger / runner）も実行して、出力生成まで到達することを確認

**理解チェック**
- Q5-1: dryで「DAQだけ」「cameraだけ」「sweep一周」をどう切り分けて確認する？

---

## 4. 進行中の“理解確認”の運用ルール（重要）
毎回の作業の最後に、あなたに以下を短く確認します。
- いま触ったコードは「何の責務」？
- 入口からそこに辿る経路は？（ファイル→クラス/関数）
- 次にバグったらどこから疑う？（3択）

※答えられない場合は、リファクタを止めて「図/メモを増やす」「命名を直す」「責務の境界を修正する」を優先します。

---

## 5. 近い次アクション（提案）
次の着手は、理解効果が高くて安全な順で以下を推奨します。
1) Sweep周り（`_sw_*`）を `sweep/` モジュールへ分割（フェーズ2）
2) 次に `DaqClient` / `CameraClient` 化（フェーズ3）
3) その後で camera 側（フェーズ4）

---

## 付録A: 参考ドキュメント
- GUIの運用メモ: [docs/shutter_gui_usage.md](../shutter_gui_usage.md)
- 引き継ぎ（2026-01-07）: [docs/notes/handoff_20260107.md](handoff_20260107.md)

---

## 6. Update (2026-01-13)
- Sweep controller extracted: src/shutter_camera_trigger/sweep/controller.py
- Sweep UI split: src/shutter_camera_trigger/sweep/ui_tab.py
- GUI tabs split: src/shutter_camera_trigger/gui_tabs/{camera_tab.py,sequence_tab.py,manual_tab.py,top_bar.py}
- Sequence loop extracted: src/shutter_camera_trigger/sequence/controller.py
- Sweep input collection moved: src/shutter_camera_trigger/sweep/input.py
- worker cleanup + camera prefs extracted: src/shutter_camera_trigger/gui_support/{worker_cleanup.py,camera_prefs.py}
- Restored entrypoint: python -m src.shutter_camera_trigger.shutter_gui

## 7. Update (2026-01-13 follow-up)
- App をさらに薄くするため、Sweep のUI補助を sweep/ui_helpers.py に移動
- Manual のアクションを gui_tabs/manual_actions.py へ移動し、App から削除
- Sequence の Start/Stop は sequence/controller の関数を直接呼び出す形へ変更
- camera_tab から worker_pids と trigger/subarray 参照を validators/worker_cleanup に集約

## 8. Update (2026-01-13 follow-up 2)
- top_bar/camera/sweep �� callback �����ɕύX���AApp �̏����� UI �A�N�V�������폜
- sweep �� UI �ˑ��A�N�V������ sweep/ui_actions.py �Ɉړ�
- file dialog �� gui_support/dialogs.py �ɏW��

## 9. Update (2026-01-13 follow-up 3)
- App �̃t�H���g/�ۑ�/�I�������� gui_support/app_lifecycle.py �Ɉړ�
- shutter_gui.py �̏����� UI/�ۑ����\�b�h���팸

## 10. Update (2026-01-13 follow-up 4)
- DAQ worker/request �̔������b�p�[���폜���Aapp._daq.request �𒼐ڎg���悤����
- daq/controller, daq/workers ���� App �����\�b�h�ˑ����폜

## 11. Update (2026-01-13 follow-up 5)
- require_connected �𒼐ڗ��p���AApp �� _require_connected ���폜
