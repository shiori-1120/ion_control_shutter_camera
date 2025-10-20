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
    
    
parent_path = "output/one-shot"
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
    qCMOS.SetParameters(JSON["qCMOS.expose-time"]
                        , JSON["qCMOS.subarray.h-width"]
                        , JSON["qCMOS.subarray.v-width"]
                        , JSON["qCMOS.subarray.h-start"]
                        , JSON["qCMOS.subarray.v-start"])                   
                            
    qCMOS.StartCapture()
    meas_id = JSON["measurement-id.take-one-shot"]
    
    now = time.time()
    dt = datetime.datetime.fromtimestamp(now)
    dir_name = "take-one-shot"    
    makeDir(today_path + "/" + dir_name)
    
    while True:
        time.sleep(JSON["qCMOS.expose-time"])
        time.sleep(0.1)
        data = qCMOS.GetLastFrame()
        print("data=", data)
        time.sleep(0.006)
        img = data[1].astype(np.float64)
        np.save(today_path + "/" + dir_name + "/img-%d.npy"   %meas_id, img)
    
    
finally:
    
    stop_time  = time.time()
    print("measurement-id", meas_id)
    meas_id += 1
    JSON["measurement-id.take-one-shot"] = meas_id
    writeJSON(JSON)
    
    print("計測時間：", stop_time - start_time, "秒")
    qCMOS.StopCapture()
    qCMOS.ReleaseBuf()
    time.sleep(0.1)

    qCMOS.CloseUninitCamera()


    print("finish %s" %dir_name)

