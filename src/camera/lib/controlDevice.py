import time
import numpy as np
import ctypes as c
import os
import logging
from typing import Optional, Tuple

from . import caio
from .CommonFunction import *
from . import dcamapi4
from .dcam import Dcam

_CAM_VERBOSE = False


def _legacy_trigger_cfg_from_env() -> dict:
    """Legacy fallback for code paths that still rely on env vars.

    The preferred configuration path is passing trigger settings via cam_cfg
    (see src/camera/ion_state_worker.py). This fallback keeps backward
    compatibility for ad-hoc scripts that instantiate Control_qCMOScamera()
    directly.
    """
    return {
        "source": os.environ.get("ION_CONTROL_CAMERA_TRIGGER_SOURCE", "EXTERNAL"),
        "connector": os.environ.get("ION_CONTROL_CAMERA_TRIGGER_CONNECTOR", ""),
        "active": os.environ.get("ION_CONTROL_CAMERA_TRIGGER_ACTIVE", ""),
        "polarity": os.environ.get("ION_CONTROL_CAMERA_TRIGGER_POLARITY", ""),
        "mode": os.environ.get("ION_CONTROL_CAMERA_TRIGGER_MODE", ""),
        "delay_s": os.environ.get("ION_CONTROL_CAMERA_TRIGGER_DELAY_S", ""),
    }


def _vprint(*args, **kwargs):
    if _CAM_VERBOSE:
        msg = ' '.join(str(a) for a in args)
        logging.debug(msg)


def _cfg_str(cfg: Optional[dict], *keys: str, default: str = "") -> str:
    if not cfg:
        return default
    for k in keys:
        try:
            v = cfg.get(k)
        except Exception:
            v = None
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return default


def _trigger_source_value(trigger_cfg: Optional[dict]) -> float:
    """Return DCAM TRIGGERSOURCE value.

    DCAM convention (as used in this repo):
    - 1: INTERNAL
    - 2: EXTERNAL

    Expected config:
      trigger_cfg = {"source": "INTERNAL"|"EXTERNAL"|"1"|"2"}
    """
    v = _cfg_str(trigger_cfg, "source", "trigger_source", default="EXTERNAL").strip().upper()
    if v in ("INTERNAL", "INT", "1"):
        return 1.0
    if v in ("EXTERNAL", "EXT", "2", ""):
        return 2.0
    # Unknown value -> default to EXTERNAL for safety.
    _vprint(f"[camera] Unknown trigger.source={v!r}; using EXTERNAL")
    return 2.0


def _parse_enum_value(value: str, mapping: dict[str, float], *, name: str) -> Optional[float]:
    """Parse a value into a numeric DCAM enum value."""
    v = (value or "").strip().upper()
    if not v:
        return None
    if v in mapping:
        return float(mapping[v])
    try:
        return float(v)
    except Exception:
        _vprint(f"[camera] Unknown {name}={value!r}; ignoring")
        return None


def _trigger_connector_value(trigger_cfg: Optional[dict], *, trig_source: float) -> Optional[float]:
    """Return TRIGGER_CONNECTOR numeric value.

    If using EXTERNAL trigger and no env override is provided, default to BNC,
    since it's the most common lab wiring for external TTL.
    """
    v = _parse_enum_value(
        _cfg_str(trigger_cfg, "connector", "trigger_connector", default=""),
        {
            "INTERFACE": float(dcamapi4.DCAMPROP.TRIGGER_CONNECTOR.INTERFACE),
            "BNC": float(dcamapi4.DCAMPROP.TRIGGER_CONNECTOR.BNC),
            "MULTI": float(dcamapi4.DCAMPROP.TRIGGER_CONNECTOR.MULTI),
        },
        name="trigger.connector",
    )
    if v is not None:
        return v
    # Default only when external trigger is requested.
    if trig_source == float(dcamapi4.DCAMPROP.TRIGGERSOURCE.EXTERNAL):
        return float(dcamapi4.DCAMPROP.TRIGGER_CONNECTOR.BNC)
    return None


def _trigger_active_value(trigger_cfg: Optional[dict]) -> Optional[float]:
    return _parse_enum_value(
        _cfg_str(trigger_cfg, "active", "trigger_active", default=""),
        {
            "EDGE": float(dcamapi4.DCAMPROP.TRIGGERACTIVE.EDGE),
            "LEVEL": float(dcamapi4.DCAMPROP.TRIGGERACTIVE.LEVEL),
            "SYNCREADOUT": float(dcamapi4.DCAMPROP.TRIGGERACTIVE.SYNCREADOUT),
            "POINT": float(dcamapi4.DCAMPROP.TRIGGERACTIVE.POINT),
        },
        name="trigger.active",
    )


def _trigger_polarity_value(trigger_cfg: Optional[dict]) -> Optional[float]:
    return _parse_enum_value(
        _cfg_str(trigger_cfg, "polarity", "trigger_polarity", default=""),
        {
            "NEGATIVE": float(dcamapi4.DCAMPROP.TRIGGERPOLARITY.NEGATIVE),
            "NEG": float(dcamapi4.DCAMPROP.TRIGGERPOLARITY.NEGATIVE),
            "FALLING": float(dcamapi4.DCAMPROP.TRIGGERPOLARITY.NEGATIVE),
            "POSITIVE": float(dcamapi4.DCAMPROP.TRIGGERPOLARITY.POSITIVE),
            "POS": float(dcamapi4.DCAMPROP.TRIGGERPOLARITY.POSITIVE),
            "RISING": float(dcamapi4.DCAMPROP.TRIGGERPOLARITY.POSITIVE),
        },
        name="trigger.polarity",
    )


def _trigger_mode_value(trigger_cfg: Optional[dict]) -> Optional[float]:
    return _parse_enum_value(
        _cfg_str(trigger_cfg, "mode", "trigger_mode", default=""),
        {
            "NORMAL": float(dcamapi4.DCAMPROP.TRIGGER_MODE.NORMAL),
            "PIV": float(dcamapi4.DCAMPROP.TRIGGER_MODE.PIV),
            "START": float(dcamapi4.DCAMPROP.TRIGGER_MODE.START),
        },
        name="trigger.mode",
    )


def _trigger_delay_s_value(trigger_cfg: Optional[dict]) -> Optional[float]:
    s = _cfg_str(trigger_cfg, "delay_s", "trigger_delay_s", default="").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        _vprint(f"[camera] Invalid trigger.delay_s={s!r}; ignoring")
        return None


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
    def __init__(self, *, trigger_cfg: Optional[dict] = None, verbose: bool = False):
        global _CAM_VERBOSE
        # If verbose wasn't provided, preserve legacy env-var behavior.
        if verbose is False:
            try:
                _CAM_VERBOSE = os.environ.get("ION_CONTROL_CAMERA_VERBOSE", "").strip() == "1"
            except Exception:
                _CAM_VERBOSE = False
        else:
            _CAM_VERBOSE = True

        if trigger_cfg:
            self._trigger_cfg = dict(trigger_cfg)
        else:
            # Backward-compatible behavior.
            self._trigger_cfg = _legacy_trigger_cfg_from_env()
        self.dcam = Dcam()
        # DCAM-APIを初期化 (check return and handle transient enumeration failures)
        init_code: int | None = None
        device_count: int = 0

        import os
        import sys
        import logging
        dll_path = getattr(self.dcam, '__file__', None)
        logging.info(f"[DCAM INIT] sys.path={sys.path}")
        logging.info(f"[DCAM INIT] os.environ PATH={os.environ.get('PATH')}")
        logging.info(f"[DCAM INIT] DLL path={dll_path}")
        logging.info(f"[DCAM INIT] DCAMAPI4 DLL={getattr(dcamapi4, '__file__', None)}")
        logging.info(f"[DCAM INIT] DCAMERR.NOCAMERA={getattr(dcamapi4.DCAMERR, 'NOCAMERA', None)}")
        for attempt in range(3):
            paraminit = dcamapi4.DCAMAPI_INIT()
            err = dcamapi4.dcamapi_init(c.byref(paraminit))
            try:
                init_code = int(err)
            except Exception:
                init_code = getattr(err, 'value', None)

            try:
                device_count = int(getattr(paraminit, 'iDeviceCount', 0) or 0)
            except Exception:
                device_count = 0

            logging.info(f"[DCAM INIT] attempt={attempt} init_code={init_code} device_count={device_count}")
            _vprint('number of connected cameras :', device_count)

            # If init itself failed (except ALREADYINITIALIZED), don't retry.
            if init_code is not None and init_code < 0 and init_code != int(dcamapi4.DCAMERR.ALREADYINITIALIZED):
                if init_code == int(dcamapi4.DCAMERR.NOCAMERA) and attempt < 2:
                    logging.warning(f"[DCAM INIT] init_code=NOCAMERA; retrying (attempt {attempt + 1}/3)")
                    try:
                        dcamapi4.dcamapi_uninit()
                    except Exception as e:
                        logging.warning(f"[DCAM INIT] dcamapi_uninit exception: {e}")
                    time.sleep(0.5)
                    continue
                logging.error(f"[DCAM INIT] init_code failure: {init_code}")
                break

            # If no camera detected, uninit and retry once or twice.
            if device_count > 0:
                break
            try:
                dcamapi4.dcamapi_uninit()
            except Exception as e:
                logging.warning(f"[DCAM INIT] dcamapi_uninit exception: {e}")
            time.sleep(0.3)

        logging.info(f"[DCAM INIT] final init_code={init_code} device_count={device_count}")

        self._device_count = int(device_count)

        # Fail fast with an actionable error message.

        if init_code is not None and init_code < 0 and init_code != int(dcamapi4.DCAMERR.ALREADYINITIALIZED):
            err_name = None
            try:
                err_name = dcamapi4.DCAMERR(int(init_code)).name
            except Exception:
                err_name = None
            suffix = f" ({err_name})" if err_name else ""
            logging.error(f"[DCAM INIT] dcamapi_init failed: {init_code}{suffix}")
            raise RuntimeError(f"dcamapi_init failed: {init_code}{suffix}")

        if self._device_count <= 0:
            # Keep API uninitialized in the no-camera case.
            try:
                dcamapi4.dcamapi_uninit()
            except Exception as e:
                logging.warning(f"[DCAM INIT] dcamapi_uninit exception (no camera): {e}")
            logging.error("[DCAM INIT] No camera detected by DCAM (device_count=0).")
            raise RuntimeError("No camera detected by DCAM (device_count=0).")

    def OpenCamera_GetHandle(self):
        if not self.dcam.dev_open():
            err = int(self.dcam.lasterr())

            # Some environments transiently report NOCAMERA just after a crash or
            # device reconnect. Retry once to reduce flakiness.
            try:
                if err == int(dcamapi4.DCAMERR.NOCAMERA):
                    time.sleep(0.2)
                    if self.dcam.dev_open():
                        err = 0
            except Exception:
                pass

            if err != 0:
                err_name = None
                try:
                    err_name = dcamapi4.DCAMERR(err).name
                except Exception:
                    err_name = None

                suffix = f" ({err_name})" if err_name else ""
                raise RuntimeError(f"Failed to open camera: {err}{suffix}")

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

    def SetParameters(self, exposure_time, h_width=None, v_width=None, h_start=None, v_start=None):
        # カメラの撮像状況を取得する
        cInt32 = c.c_int32()
        dcamcapstatus = dcamapi4.DCAMCAP_STATUS
        dcamapi4.dcamcap_status(self.__hdcam, c.byref(cInt32))
        _vprint(dcamcapstatus(cInt32.value))

        # Trigger settings
        trig_source = _trigger_source_value(self._trigger_cfg)  # 1:INTERNAL, 2:EXTERNAL
        cDouble = c.c_double(trig_source)
        idprop = dcamapi4.DCAM_IDPROP.TRIGGERSOURCE
        dcamapi4.dcamprop_setgetvalue(self.__hdcam, idprop, c.byref(cDouble), 0)
        _vprint("TRIGGERSOURCE=", cDouble.value)

        # When using external trigger, also ensure connector/edge/polarity/mode are sensible.
        # These can be overridden per-run via env vars.
        if trig_source == float(dcamapi4.DCAMPROP.TRIGGERSOURCE.EXTERNAL):
            # Connector
            conn = _trigger_connector_value(self._trigger_cfg, trig_source=trig_source)
            if conn is not None:
                cDouble = c.c_double(conn)
                idprop = dcamapi4.DCAM_IDPROP.TRIGGER_CONNECTOR
                dcamapi4.dcamprop_setgetvalue(self.__hdcam, idprop, c.byref(cDouble), 0)
                _vprint("TRIGGER_CONNECTOR=", cDouble.value)

            # Trigger mode (per-frame trigger usually NORMAL)
            mode = _trigger_mode_value(self._trigger_cfg)
            if mode is None:
                mode = float(dcamapi4.DCAMPROP.TRIGGER_MODE.NORMAL)
            cDouble = c.c_double(mode)
            idprop = dcamapi4.DCAM_IDPROP.TRIGGER_MODE
            dcamapi4.dcamprop_setgetvalue(self.__hdcam, idprop, c.byref(cDouble), 0)
            _vprint("TRIGGER_MODE=", cDouble.value)

            # Active edge/level
            active = _trigger_active_value(self._trigger_cfg)
            if active is None:
                active = float(dcamapi4.DCAMPROP.TRIGGERACTIVE.EDGE)
            cDouble = c.c_double(active)
            idprop = dcamapi4.DCAM_IDPROP.TRIGGERACTIVE
            dcamapi4.dcamprop_setgetvalue(self.__hdcam, idprop, c.byref(cDouble), 0)
            _vprint("TRIGGERACTIVE=", cDouble.value)

            # Polarity (default POSITIVE = rising edge)
            pol = _trigger_polarity_value(self._trigger_cfg)
            if pol is None:
                pol = float(dcamapi4.DCAMPROP.TRIGGERPOLARITY.POSITIVE)
            cDouble = c.c_double(pol)
            idprop = dcamapi4.DCAM_IDPROP.TRIGGERPOLARITY
            dcamapi4.dcamprop_setgetvalue(self.__hdcam, idprop, c.byref(cDouble), 0)
            _vprint("TRIGGERPOLARITY=", cDouble.value)

            # Optional delay
            delay_s = _trigger_delay_s_value(self._trigger_cfg)
            if delay_s is not None:
                cDouble = c.c_double(float(delay_s))
                idprop = dcamapi4.DCAM_IDPROP.TRIGGERDELAY
                dcamapi4.dcamprop_setgetvalue(self.__hdcam, idprop, c.byref(cDouble), 0)
                _vprint("TRIGGERDELAY=", cDouble.value)

       # SENSORMODEをPHOTON NUNBER RESOLVINGモードに設定する。
        cDouble = c.c_double(18)
        idprop = dcamapi4.DCAM_IDPROP.SENSORMODE
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)

        # subarrayの情報を設定する
        # ROI 未指定のときはフルフレームを採用
        if h_width is None or v_width is None or h_start is None or v_start is None:
            fValue = c.c_double()
            idprop = dcamapi4.DCAM_IDPROP.IMAGE_WIDTH
            dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))
            full_w = int(fValue.value)
            idprop = dcamapi4.DCAM_IDPROP.IMAGE_HEIGHT
            dcamapi4.dcamprop_getvalue(self.__hdcam, idprop, c.byref(fValue))
            full_h = int(fValue.value)
            h_width = full_w
            v_width = full_h
            h_start = 0
            v_start = 0
        # subarray mode
        cDouble = c.c_double(2)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYMODE  # 4202832
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        _vprint('subarray mode : ', cDouble.value)
        # 水平方向の幅
        cDouble = c.c_double(h_width)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYHSIZE  # 4202784
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        _vprint('Horizontal size', cDouble.value)
        # 水平方向の起点
        cDouble = c.c_double(h_start)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYHPOS  # 4202768
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        _vprint('Horizon pos', cDouble.value)
        # 垂直方向の幅
        cDouble = c.c_double(v_width)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYVSIZE  # 4202832
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        _vprint('Vertical size', cDouble.value)
        # 垂直方向の起点
        cDouble = c.c_double(v_start)
        idprop = dcamapi4.DCAM_IDPROP.SUBARRAYVPOS  # 4202800
        dcamapi4.dcamprop_setgetvalue(
            self.__hdcam, idprop, c.byref(cDouble), 0)
        _vprint('Vertical pos', cDouble.value)

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
        _vprint('exposure time [sec.]', cDouble.value)
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
        _vprint(dcamcapstatus(cInt32.value))

        # バッファ制御
        nFrame = 10
        cFrame = c.c_int32(nFrame)
        dcamapi4.dcambuf_alloc(self.__hdcam, cFrame)

        # バッファを取り込むためのフレームの準備
        self.dcammisc_setupframe()

    def StartCapture(self):
        _vprint('--- Begin Capture --- ')
        # 再度、カメラの撮像状況を取得する
        cInt32 = c.c_int32()
        dcamcapstatus = dcamapi4.DCAMCAP_STATUS
        dcamapi4.dcamcap_status(self.__hdcam, c.byref(cInt32))
        _vprint(dcamcapstatus(cInt32.value))
        mode = dcamapi4.DCAMCAP_START.SEQUENCE
        dcamapi4.dcamcap_start(self.__hdcam, mode)

    def StopCapture(self):
        _vprint('--- End Capture --- ')
        dcamcaptrans = dcamapi4.DCAMCAP_TRANSFERINFO()
        dcamapi4.dcamcap_transferinfo(self.__hdcam, dcamcaptrans)
        _vprint("取り込んだフレーム数 : ", dcamcaptrans.nFrameCount)

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
                init_err = dcamapi4.dcamapi_init(c.byref(paraminit))
                info['device_count'] = int(getattr(paraminit, 'iDeviceCount', 0) or 0)
                try:
                    info['init_err'] = int(init_err)
                except Exception:
                    info['init_err'] = getattr(init_err, 'value', str(init_err))

                if info['device_count'] <= 0:
                    try:
                        dcamapi4.dcamapi_uninit()
                    except Exception:
                        pass
                    time.sleep(delay)
                    continue

                if not try_open:
                    try:
                        dcamapi4.dcamapi_uninit()
                    except Exception:
                        pass
                    return True, info

                # 実際に1台を開いてプロパティを取得してみる
                paramdevopen = dcamapi4.DCAMDEV_OPEN()
                open_err = dcamapi4.dcamdev_open(c.byref(paramdevopen))
                try:
                    info['open_err'] = int(open_err)
                except Exception:
                    info['open_err'] = getattr(open_err, 'value', str(open_err))
                if int(getattr(open_err, 'value', open_err)) < 0:
                    try:
                        dcamapi4.dcamapi_uninit()
                    except Exception:
                        pass
                    time.sleep(delay)
                    continue
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
                try:
                    dcamapi4.dcamapi_uninit()
                except Exception:
                    pass
                if int(err) == 0 or getattr(err, 'value', 0) == 0:
                    info['open_ok'] = True
                    return True, info
            except Exception as ex:
                info['prop_err'] = str(ex)
                try:
                    dcamapi4.dcamapi_uninit()
                except Exception:
                    pass
            time.sleep(delay)
        return False, info
