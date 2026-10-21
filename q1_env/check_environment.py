"""Reproducible environment checks; results are logs, not screenshots."""
from pathlib import Path
import sys
import platform
import json
import os
from datetime import datetime
from importlib.metadata import version

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import yaml

def main():
    assert sys.version_info >= (3,10)
    assert np.allclose(np.eye(3)@np.ones(3),np.ones(3))
    ok,encoded=cv2.imencode('.png',np.zeros((16,16,3),np.uint8));assert ok
    assert cv2.imdecode(encoded,cv2.IMREAD_COLOR).shape==(16,16,3)
    assert hasattr(cv2,'solvePnP') and hasattr(cv2,'calibrateCamera')
    assert yaml.safe_load('enabled: true')['enabled'] is True
    fig,ax=plt.subplots();ax.plot([0,1],[0,1]);fig.canvas.draw();plt.close(fig)
    result=dict(checked_at=datetime.now().astimezone().isoformat(),python=sys.version,
        platform=platform.platform(),versions={name:version(name) for name in
        ['numpy','opencv-python','matplotlib','PyYAML']},
        checks=dict(python_310_plus='PASS',numpy='PASS',opencv_png='PASS',
                    opencv_calibration_functions='PASS',matplotlib_agg='PASS',pyyaml='PASS'))
    text=json.dumps(result,ensure_ascii=False,indent=2)
    (HERE/'environment_check.json').write_text(text,encoding='utf-8')
    (HERE/'environment_check.txt').write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__':main()
