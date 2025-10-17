import pyvisa


def main():
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    if not resources:
        print("No VISA resources found.")
        return

    for r in resources:
        try:
            inst = rm.open_resource(r, timeout=2000)
        except Exception as e:
            print(f" - {r} -> open failed: {e}")
            continue

        try:
            idn = inst.query("*IDN?")
            print(f" - {r} -> {idn.strip()}")
        except Exception as e:
            print(f" - {r} -> query failed: {e}")
        finally:
            try:
                inst.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
