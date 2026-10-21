"""Dedicated middle-distance spin tuning; translation parameters stay separate."""
from pathlib import Path
import sys
import tune_thresholds
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    video=HERE.parent/'data'/'armor'/'中距离陀螺.mp4'
    sys.argv=[sys.argv[0], '--video',str(video),
              '--config',str(HERE/'thresholds_spin.json'),
              '--panel-output',str(HERE/'figures'/'threshold_panel_spin.png'),
              '--title','Spin armor tuner',*sys.argv[1:]]
    tune_thresholds.main()
