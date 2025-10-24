import time
import numpy as np
import ctypes as c
from typing import Optional, Tuple

import lib.caio as caio
from lib.CommonFunction import *
import lib.dcamapi4 as dcamapi4
from lib.dcam import Dcam


class Control_CONTEC():
    def __init__(self):
        Iret = c.c_long()
        self.Id = c.c_short()
        buf = "AIO000"
        self.Ret = c.c_long()

        Iret.value = caio.AioInit(buf.encode(), c.byref(self.Id))
        self._init_ret = Iret.value
        self.Ret.value = caio.AioOutputDoBit(self.Id, 0, 0x0000)

        self.Ret = c.c_long()
        self.Ret.value = caio.AioResetDevice(self.Id)
        # print(self.Ret.value)

    def SendTrigger(self):
        self.Ret.value = caio.AioOutputDoBit(self.Id, 0, 0x0001)
        self.Ret.value = caio.AioOutputDoBit(self.Id, 0, 0x0000)
        time.sleep(0.01)
        # print(self.Ret.value)

    def ControlShutter(self, onoff):
        if onoff == 1:
            self.Ret.value = caio.AioOutputDoBit(self.Id, 1, 0x0000)
        if onoff == 0:
            self.Ret.value = caio.AioOutputDoBit(self.Id, 1, 0x0001)

    def is_connected(self) -> bool:
        """簡易判定: 初期化リターンが成功コード(0)かつ Id が妥当であることを確認
        使い方(例): connected = ctrl.is_connected()
        """
        return getattr(self, '_init_ret', None) == 0 and getattr(self, 'Id', None) not in (None, 0)

    def check_connection(self, retries=3, delay=0.5):
        """厳密チェック: AioInit -> AioResetDevice を試行して応答を確認
        戻り値: (ok:bool, info:dict)
        使い方(例): ok, info = ctrl.check_connection()
        """
        info = {'attempts': 0, 'last_errno': None,
                'id': None, 'reset_ret': None}
        buf = "AIO000"
        for i in range(retries):
            info['attempts'] += 1
            try:
                Iret = c.c_long()
                Id = c.c_short()
                Iret.value = caio.AioInit(buf.encode(), c.byref(Id))
                info['last_errno'] = int(Iret.value)
                info['id'] = int(Id.value) if Id.value is not None else None
                if Iret.value == 0:
                    # 初期化成功 -> デバイス操作で応答確認
                    try:
                        ret = caio.AioResetDevice(Id)
                        info['reset_ret'] = int(ret) if hasattr(
                            ret, 'value') or isinstance(ret, int) else ret
                    except Exception as ex:
                        info['reset_ret'] = str(ex)
                    return True, info
            except Exception as ex:
                info['last_errno'] = str(ex)
            time.sleep(delay)
        return False, info


class Control_qCMOScamera():
    def __init__(self):
        self.dcam = Dcam()
        # DCAM-APIを初期化
        paraminit = dcamapi4.DCAMAPI_INIT()
        dcamapi4.dcamapi_init(paraminit)
        print('number of connected cameras :',
              paraminit.iDeviceCount)  # 接続されているカメラ台数を表示
        self._device_count = int(paraminit.iDeviceCount)

    def OpenCamera_GetHandle(self):
        if not self.dcam.dev_open():
            err = self.dcam.lasterr()
            raise RuntimeError(f"Failed to open camera: {err}")

        self.__hdcam = getattr(self.dcam, '_Dcam__hdcam', 0)
        if not self.__hdcam:
            raise RuntimeError(
                "Failed to acquire camera handle after opening.")

        self.__bufframe = dcamapi4.DCAMBUF_FRAME()

    def ReleaseBuf(self):
        self.dcam.buf_release()

    def CloseUninitCamera(self):
        self.dcam.dev_close()
        self.__hdcam = 0
        dcamapi4.dcamapi_uninit()

    def dcammisc_alloc_ndarray(self):
        framebundlenum = 1
        height = self.__bufframe.height * framebundlenum

        if self.__bufframe.type == dcamapi4.DCAM_PIXELTYPE.MONO16:
            return np.zeros((height, self.__bufframe.width), dtype='uint16')

        if self.__bufframe.type == dcamapi4.DCAM_PIXELTYPE.MONO8:
            return np.zeros((height, self.__bufframe.width), dtype='uint8')

        return False

    def dcammisc_setupframe(self):
        fValue = c.c_double()

        idprop = dcamapi4.DCAM_IDPROP.IMAGE_PIXELTYPE
        err = dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))
        self.__bufframe.type = int(fValue.value)

        idprop = dcamapi4.DCAM_IDPROP.IMAGE_WIDTH
        err = dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))
        self.__bufframe.width = int(fValue.value)

        idprop = dcamapi4.DCAM_IDPROP.IMAGE_HEIGHT
        err = dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))
        self.__bufframe.height = int(fValue.value)

        idprop = dcamapi4.DCAM_IDPROP.FRAMEBUNDLE_MODE
        err = dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))

        if not int(fValue.value) == dcamapi4.DCAMPROP.MODE.ON:
            idprop = dcamapi4.DCAM_IDPROP.FRAMEBUNDLE_ROWBYTES
            err = dcamapi4.dcamprop_getvalue(
                self.__hdcam, idprop, c.byref(fValue))
            self.__bufframe.rowbytes = int(fValue.value)
        else:
            idprop = dcamapi4.DCAM_IDPROP.IMAGE_ROWBYTES
            err = dcamapi4.dcamprop_getvalue(
                self.__hdcam, idprop, c.byref(fValue))
            self.__bufframe.rowbytes = int(fValue.value)

        return err

    def SetParameters(self, exposure_time, h_width, v_width, h_start, v_start):
        # カメラの撮像状況を取得する
        cInt32 = c.c_int32()
        dcamcapstatus = dcamapi4.DCAMCAP_STATUS
        dcamapi4.dcamcap_status(self.__hdcam, c.byref(cInt32))
        print(dcamcapstatus(cInt32.value))

        # TRIGGERSOURCEの値をEXTERNALモードに変更する
        cDouble = c.c_double(2)  # 1:INTERNAL, 2:EXTERNAL
        idprop = dcamapi4.DCAM_IDPROP.TRIGGERSOURCE
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        print(cDouble.value)

       # SENSORMODEをPHOTON NUNBER RESOLVINGモードに設定する。
        cDouble = c.c_double(18)
        idprop = dcamapi4.DCAM_IDPROP.SENSORMODE
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)

        # subarrayの情報を設定する
        # subarray mode
        cDouble = c.c_double(2)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYMODE  # 4202832
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        print('subarray mode : ', cDouble.value)
        # 水平方向の幅
        cDouble = c.c_double(h_width)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYHSIZE  # 4202784
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        print('Horizontal size', cDouble.value)
        # 水平方向の起点
        cDouble = c.c_double(h_start)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYHPOS  # 4202768
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        print('Horizon pos', cDouble.value)
        # 垂直方向の幅
        cDouble = c.c_double(v_width)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYVSIZE  # 4202832
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        print('Vertical size', cDouble.value)
        # 垂直方向の起点
        cDouble = c.c_double(v_start)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYVPOS  # 4202800
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        print('Vertical pos', cDouble.value)

        # exposure time
        # 露光時間の制御をオンにする
        cDouble = c.c_double(2)
        idprop = dcamapi4.DCAM_IDPROP.EXPOSURETIME_CONTROL  # 2031920
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        # 露光時間を設定する
        cDouble = c.c_double(exposure_time)
        idprop = dcamapi4.DCAM_IDPROP.EXPOSURETIME  # 2031888
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        print('exposure time [sec.]', cDouble.value)
        # 撮像を開始してから初めて入力されるトリガ信号のふるまいを決める
        cDouble = c.c_double(1)
        idprop = dcamapi4.DCAM_IDPROP.FIRSTTRIGGER_BEHAVIOR
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)

        # output trigger
        # トリガ信号をPROGRAMABLEに設定する。これでトリガの種類やトリガ信号の長さを指定することができる
        cDouble = c.c_double(3)
        idprop = dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_KIND  # 1835360
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        # グローバル露光の立ち上がりが起点となる
        cDouble = c.c_double(1)
        idprop = dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_SOURCE  # 1835280
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        # トリガ信号の種類をEDGEに設定する
        cDouble = c.c_double(1)
        idprop = dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_ACTIVE  # 1835312
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        # トリガ信号の極性を正とする
        cDouble = c.c_double(2)
        idprop = dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_POLARITY  # 1835296
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        # トリガ信号の長さを露光時間と同じに設定する
        cDouble = c.c_double(exposure_time)  # in [sec.]
        idprop = dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_PERIOD  # 1835344
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        # _1CHANNEL
        cDouble = c.c_double(1)
        idprop = dcamapi4.DCAM_IDPROP.OUTPUTTRIGGER_CHANNELSYNC  # 1835056
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)

        # 画像取り込みモードの指定 SNAP or SEQUENCE
        # カメラの撮像状況を取得する
        cInt32 = c.c_int32()
        dcamcapstatus = dcamapi4.DCAMCAP_STATUS
        dcamapi4.dcamcap_status(self.__hdcam, c.byref(cInt32))
        print(dcamcapstatus(cInt32.value))

        # バッファ制御
        nFrame = 10
        cFrame = c.c_int32(nFrame)
        dcamapi4.dcambuf_alloc(self.__hdcam, cFrame)

        # バッファを取り込むためのフレームの準備
        self.dcammisc_setupframe()

    def StartCapture(self):
        print('--- Begin Capture --- ')
        # 再度、カメラの撮像状況を取得する
        cInt32 = c.c_int32()
        dcamcapstatus = dcamapi4.DCAMCAP_STATUS
        dcamapi4.dcamcap_status(self.__hdcam, c.byref(cInt32))
        print(dcamcapstatus(cInt32.value))
        mode = dcamapi4.DCAMCAP_START.SEQUENCE
        dcamapi4.dcamcap_start(self.__hdcam, mode)

    def StopCapture(self):
        print('--- End Capture --- ')
        dcamcaptrans = dcamapi4.DCAMCAP_TRANSFERINFO()
        dcamapi4.dcamcap_transferinfo(self.__hdcam, dcamcaptrans)
        print("取り込んだフレーム数 : ", dcamcaptrans.nFrameCount)

        dcamapi4.dcamcap_stop(self.__hdcam)

    def GetLastFrame(self):
        framebundlenum = 1
        iFrame = -1
        self.__bufframe.iFrame = -1
        npBuf = self.dcammisc_alloc_ndarray()

        aFrame = dcamapi4.DCAMBUF_FRAME()
        aFrame.iFrame = -1

        aFrame.buf = npBuf.ctypes.data_as(c.c_void_p)
        aFrame.rowbytes = self.__bufframe.rowbytes
        aFrame.type = self.__bufframe.type
        aFrame.width = self.__bufframe.width
        aFrame.height = self.__bufframe.height

        dcamapi4.dcambuf_copyframe(self.__hdcam, c.byref(aFrame))

        return (aFrame, npBuf)

    def wait_for_frame_ready(self, timeout_sec: float) -> Tuple[bool, Optional[dcamapi4.DCAMERR]]:
        """Wait for FRAMEREADY event and return (success, error)."""
        if not self.__hdcam:
            return False, dcamapi4.DCAMERR.INVALIDHANDLE

        timeout_ms = max(int(max(timeout_sec, 0.0) * 1000), 1)
        if self.dcam.wait_capevent_frameready(timeout_ms):
            return True, None

        err_code = self.dcam.lasterr()
        if isinstance(err_code, dcamapi4.DCAMERR):
            return False, err_code

        try:
            return False, dcamapi4.DCAMERR(int(err_code))
        except Exception:
            return False, None

    def capture_roi_frame(self, exposure_time: float, roi, wait_margin: float = 0.01) -> np.ndarray:
        """Set a subarray ROI, capture a single frame, and return it as ndarray."""
        h_width, v_width, h_start, v_start = map(int, roi)

        # Configure camera for requested ROI and exposure
        self.SetParameters(exposure_time, h_width, v_width, h_start, v_start)

        self.StartCapture()
        try:
            # Wait for exposure plus small margin before grabbing the frame
            time.sleep(max(exposure_time, 0.0) + wait_margin)
            _, frame = self.GetLastFrame()
            return frame.copy()
        finally:
            self.StopCapture()

    # ---- Status helper methods ----
    def get_capture_status(self):
        """Return raw DCAM capture status integer (use DCAMCAP_STATUS to interpret)."""
        cInt32 = c.c_int32()
        dcamapi4.dcamcap_status(self.__hdcam, c.byref(cInt32))
        return int(cInt32.value)

    def get_buffered_frame_count(self):
        """Return number of frames currently transferred into the capture buffer."""
        info = dcamapi4.DCAMCAP_TRANSFERINFO()
        dcamapi4.dcamcap_transferinfo(self.__hdcam, info)
        return int(info.nFrameCount)

    def get_trigger_source(self):
        """Return current TRIGGERSOURCE property value (numeric)."""
        fValue = c.c_double()
        idprop = dcamapi4.DCAM_IDPROP.TRIGGERSOURCE
        dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))
        return float(fValue.value)

    def get_exposure_time(self):
        """Return current EXPOSURETIME value in seconds."""
        fValue = c.c_double()
        idprop = dcamapi4.DCAM_IDPROP.EXPOSURETIME
        dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))
        return float(fValue.value)

    def is_connected(self):
        """簡易判定: 初期化時に検出したカメラ台数が1以上か
        使い方(例): connected = cam.is_connected()
        """
        return getattr(self, '_device_count', 0) > 0

    def check_connection(self, retries=3, delay=0.5, try_open=True):
        """厳密チェック:
        - dcamapi_init を呼んでデバイス数を確認
        - (オプション) 実際に dcamdev_open し、dcamprop_getvalue でプロパティが取得できるか検証
        戻り値: (ok:bool, info:dict)
        使い方(例): ok, info = cam.check_connection()
        """
        info = {'attempts': 0, 'device_count': 0,
                'open_ok': False, 'prop_err': None}
        for i in range(retries):
            info['attempts'] += 1
            try:
                paraminit = dcamapi4.DCAMAPI_INIT()
                dcamapi4.dcamapi_init(paraminit)
                info['device_count'] = int(paraminit.iDeviceCount)
                if info['device_count'] <= 0:
                    time.sleep(delay)
                    continue

                if not try_open:
                    return True, info

                # 実際に1台を開いてプロパティを取得してみる
                paramdevopen = dcamapi4.DCAMDEV_OPEN()
                dcamapi4.dcamdev_open(paramdevopen)
                hdcam = paramdevopen.hdcam
                # 例: IMAGE_WIDTH を取得してハンドルが有効かを判定
                fValue = c.c_double()
                idprop = dcamapi4.DCAM_IDPROP.IMAGE_WIDTH
                err = dcamapi4.dcamprop_getvalue(
                    hdcam, idprop, c.byref(fValue))
                try:
                    info['prop_err'] = int(err)
                except Exception:
                    info['prop_err'] = getattr(err, 'value', str(err))
                # close/uninit
                try:
                    dcamapi4.dcamdev_close(hdcam)
                except Exception:
                    try:
                        dcamapi4.dcamdev_close(paramdevopen.hdcam)
                    except Exception:
                        pass
                dcamapi4.dcamapi_uninit()
                if int(err) == 0 or getattr(err, 'value', 0) == 0:
                    info['open_ok'] = True
                    return True, info
            except Exception as ex:
                info['prop_err'] = str(ex)
            time.sleep(delay)
        return False, info
