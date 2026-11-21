import sys
from pathlib import Path
SRC = Path('/home/dogdiving/Documents/Captstone-Repo/test-bench/src')
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tkinter as tk
from test_bench.frame_view import FrameView

def repro():
    try:
        root = tk.Tk()
        canvas = tk.Canvas(root)
        fv = FrameView(canvas)
        # Test rendering without setting _src
        fv._render()
        print('Rendered without _src (expected None guard)')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    repro()
