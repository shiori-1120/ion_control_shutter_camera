# standard
import json
import time
import datetime
import os
# third party
import numpy as np

import sys
# my module
from lib.ControlDevice import Control_qCMOScamera
from lib.dcamapi4 import DCAMERR


WAIT_MARGIN_SEC = 0.2


def getJSON():
    path_jsonfile = "./dataset.json"
    with open(path_jsonfile) as js:
        dict_json = json.load(js)
        return dict_json


def writeJSON(dict_json):
    try:
        path_jsonfile = "./dataset.json"
        js = open(path_jsonfile, "w")
        json.dump(dict_json, js, indent=4)
        print("writeJSON() true")
    except:
        print("writeJSON() false")


def makeDir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)


parent_path = "one-shot/output"
makeDir(parent_path)
timestamp = time.time()

now = time.time()
dt = datetime.datetime.fromtimestamp(now)
today_path = parent_path + "/" + dt.strftime("%Y-%m-%d") + "/raw-data"
makeDir(today_path)
# working_path = today_path + "/" + "qCMOS"

start_time = time.time()
try:
    print("open")
    JSON = getJSON()

    qCMOS = Control_qCMOScamera()
    qCMOS.OpenCamera_GetHandle()
    qCMOS.SetParameters(JSON["qCMOS.expose-time"], JSON["qCMOS.subarray.h-width"],
                        JSON["qCMOS.subarray.v-width"], JSON["qCMOS.subarray.h-start"], JSON["qCMOS.subarray.v-start"])

    meas_id = JSON["measurement-id.take-one-shot"]
    exposure_time = float(JSON["qCMOS.expose-time"])
    wait_timeout_sec = max(exposure_time + WAIT_MARGIN_SEC, 0.05)

    now = time.time()
    dt = datetime.datetime.fromtimestamp(now)
    dir_name = "take-one-shot"
    makeDir(today_path + "/" + dir_name)

    qCMOS.StartCapture()
    start_time = time.time()
    # この実行ループ中は同じタイムスタンプを使い、成功保存ごとに meas_id を進める
    loop_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    count = 0
    failure_attempts = 0
    max_failures = 10
    while count < 10:
        wait_ok, wait_err = qCMOS.wait_for_frame_ready(wait_timeout_sec)
        if not wait_ok:
            err_label = wait_err.name if isinstance(
                wait_err, DCAMERR) else str(wait_err)
            print(f"Frame wait failed: {err_label}. Retrying...")
            failure_attempts += 1
            if failure_attempts >= max_failures:
                print("Reached maximum wait retries. Stopping capture loop.")
                break
            continue
        data = qCMOS.GetLastFrame()
        img = data[1].astype(np.float64)
        # 成功条件: 全要素がゼロでないフレームのみ保存
        if img.size == 0 or not np.any(img):
            failure_attempts += 1
            if failure_attempts >= max_failures:
                print("Image buffer stayed empty. Stopping capture loop.")
                break
            continue
        filename = f"{loop_ts}_{meas_id:06d}.npy"
        np.save(today_path + "/" + dir_name + "/" + filename, img)
        meas_id += 1
        count += 1
        failure_attempts = 0


finally:

    stop_time = time.time()
    print("measurement-id", meas_id)
    JSON["measurement-id.take-one-shot"] = meas_id
    writeJSON(JSON)

    print("計測時間：", stop_time - start_time, "秒")
    qCMOS.StopCapture()
    qCMOS.ReleaseBuf()
    time.sleep(0.1)

    qCMOS.CloseUninitCamera()

    print("finish %s" % dir_name)
