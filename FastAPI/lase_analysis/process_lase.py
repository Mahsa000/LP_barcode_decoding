# FILE: process_lase.py
#!/usr/bin/env python3
import sys, pathlib
# add the parent directory of cc_codes to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse
from pathlib import Path

from .lase_cc_helpers import load_filtered_data, analyze_connected_components, save_cc_data

def main():
    p = argparse.ArgumentParser(description="Extract CC data from a .map.lase file")
    p.add_argument("lase_file", help=".map.lase input")
    p.add_argument("-g","--group",    help="specific group (e.g. grp_0)", default=None)
    p.add_argument("-a","--area",     help="area code filter",      type=int, default=None)
    p.add_argument("-o","--output",   help="output pickle path",    default="cc_data.pkl")
    args = p.parse_args()

    raw = load_filtered_data(args.lase_file, group=args.group, area=args.area)
    cc  = analyze_connected_components(raw)
    save_cc_data(cc, Path(args.output))
    print(f"→ saved connected-component data to {args.output}")

if __name__=="__main__":
    main()
