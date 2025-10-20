#================================================================
#================================================================
# API-AIO(WDM)
# アナログ簡易出力サンプル
# コモンファンクションファイル
#                                                CONTEC Co., Ltd.
#================================================================
#================================================================
#----------------------------------------
# インポートモジュール
#----------------------------------------
import lib.caio


#========================================
# 関数名    isnum
# 引数      str     文字列
# 引数      base    ベース
# 詳細      文字列を数値に変換できるかどうか確認する関数
# 戻り値    True：変換できる
#           False：変換できない
#========================================
def isnum(str, base):
    try:
        if 10 == base:
            int(str, 10)
        else:
            float(str)
    except:
        return False
    return True


#========================================
# 関数名    GetRangeString
# 引数      ao_range           レンジの設定値
# 詳細      レンジの設定値から文字列を取得する関数になります
# 戻り値    ao_range_string    レンジの設定値にあった文字列
#========================================
def GetRangeString(ao_range):
    if ao_range.value == caio.PM10:
        ao_range_string = "±10[V]"
    elif ao_range.value == caio.PM5:
        ao_range_string = "±5[V]"
    elif ao_range.value == caio.PM25:
        ao_range_string = "±2.5[V]"
    elif ao_range.value == caio.PM125:
        ao_range_string = "±1.25[V]"
    elif ao_range.value == caio.P10:
        ao_range_string = "0-10[V]"
    elif ao_range.value == caio.P5:
        ao_range_string = "0-5[V]"
    elif ao_range.value == caio.P25:
        ao_range_string = "0-2.5[V]"
    elif ao_range.value == caio.P125:
        ao_range_string = "0-1.25[V]"
    elif ao_range.value == caio.P20MA:
        ao_range_string = "0-20[mA]"
    elif ao_range.value == caio.P4TO20MA:
        ao_range_string = "4-20[mA]"
    elif ao_range.value == caio.P1TO5:
        ao_range_string = "1-5[V]"
    else:
        ao_range_string = "-"
    return ao_range_string