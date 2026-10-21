"""Validate submitted formats, basic numerical invariants and media readability."""
from pathlib import Path
import ast
import csv
import json
import hashlib
from datetime import datetime
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
NAMES={'远距离平移':1546,'中距离陀螺':2550,'近距离陀螺':2105,'装甲板_蓝':1684}
DETECTION=['frame','lt_x','lt_y','lb_x','lb_y','rb_x','rb_y','rt_x','rt_y','x','y','z','yaw']

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);return reader.fieldnames,list(reader)

def main():
    report=dict(checked_at=datetime.now().astimezone().isoformat(),python_syntax=[],variants={})
    for file in ROOT.rglob('*.py'):
        if '.git' in file.parts:continue
        ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
        report['python_syntax'].append(file.relative_to(ROOT).as_posix())
    for folder in ['qA_aim','qA_yolo']:
        variant={}
        for name,total in NAMES.items():
            fields,data=rows(ROOT/folder/f'detections_{name}.csv');assert fields==DETECTION
            previous=-1
            for row in data:
                frame=int(row['frame']);assert previous<=frame<total;previous=frame
                assert np.isfinite([float(row[k]) for k in fields[1:9]]).all()
                pose=[row[k] for k in ['x','y','z','yaw']]
                assert all(pose) or not any(pose)
                if all(pose):assert float(row['z'])>0 and np.isfinite([float(v) for v in pose]).all()
                if name=='装甲板_蓝':assert not any(pose)
            video=ROOT/folder/'figures'/f'detected_{name}.mp4'
            cap=cv2.VideoCapture(str(video));assert cap.isOpened(),str(video)
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert count==total,(str(video),count,total)
            expected_fps=24 if name=='装甲板_蓝' else 150
            assert abs(cap.get(cv2.CAP_PROP_FPS)-expected_fps)<.1
            for frame in [0,total//2,total-1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok
                assert image.shape[1::-1]==((640,480) if name=='装甲板_蓝' else (1440,1080))
            cap.release()
            variant[name]=dict(frames=total,candidates=len(data),valid_poses=sum(bool(r['x']) for r in data),video='PASS')
        fields,data=rows(ROOT/folder/'pred_远距离平移.csv');assert fields==['t','x','y','z'] and len(data)==1546
        for i,row in enumerate(data):assert abs(float(row['t'])-i/150)<1e-8
        for name in ['中距离陀螺','近距离陀螺']:
            fields,data=rows(ROOT/folder/f'pred_{name}.csv');assert fields==['t','cx','cy','cz','omega'] and len(data)==NAMES[name]
            for i,row in enumerate(data):
                assert abs(float(row['t'])-i/150)<1e-8
                if i<90:assert all(not row[k] for k in fields[1:])
        report['variants'][folder]=variant
    # Re-evaluate main forecast against exactly the submitted observation cache.
    _,cache=rows(ROOT/'qA_aim/filter_diagnostics.csv');_,pred=rows(ROOT/'qA_aim/pred_远距离平移.csv')
    obs=np.array([[float(r['obs_'+a]) if r['obs_'+a] else np.nan for a in 'xyz'] for r in cache])
    forecast=np.array([[float(r[a]) if r[a] else np.nan for a in 'xyz'] for r in pred])
    future=np.full_like(obs,np.nan);future[:-38]=(obs[37:-1]+obs[38:])/2
    mask=np.isfinite(future).all(axis=1)&np.isfinite(forecast).all(axis=1)&(np.arange(len(obs))/150>=.15)
    rmse=float(np.sqrt(np.mean(np.sum((forecast[mask]-future[mask])**2,axis=1))))
    saved=json.loads((ROOT/'qA_aim/run_summary.json').read_text(encoding='utf-8'))['prediction']
    assert int(mask.sum())==saved['valid_comparisons'] and abs(rmse-saved['rmse_euclidean_m'])<1e-10
    report['main_forecast_recomputed']=dict(samples=int(mask.sum()),rmse_m=rmse)
    files=[f for f in ROOT.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts and '.plot_cache' not in f.parts]
    assert all(f.stat().st_size<100*1024**2 for f in files),'GitHub per-file limit exceeded'
    report['status']='PASS';report['note']='Format/consistency checks, not accuracy certification.'
    (ROOT/'tools/submission_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
