def get_waveform(inst, ch):
    inst.write(f"DAT:SOU CH{ch}")
    inst.write("DAT:STAR 1")
    inst.write("DAT:STOP 10000")
    
    datapoint = inst.query("WFMO:NR_pt?")
    xzero = inst.query("WFMO:XZE?")
    xinc = inst.query("WFMO:XIN?")
    
    time_list = []
    for i in range(int(datapoint)):
        time_list.append(float(xzero)+float(xinc)*(i-1))
    
    yzero = inst.query("WFMO:YZE?")
    ymult = inst.query("WFMO:YMU?")
    
    curve_ascii = inst.query("CURV?")
    curve_ascii_split = curve_ascii.split(",")
    curve_int = []
    for i in curve_ascii_split:
        curve_int.append(float(yzero)+float(ymult)*int(i))

    return time_list, curve_int