"""Rebuild a visualization from submitted CSV without changing observations."""
from pathlib import Path
import argparse
import csv
from collections import defaultdict
import cv2
import numpy as np
from run_analysis import source

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',default='中距离陀螺')
    args=parser.parse_args();name=args.video
    groups=defaultdict(list)
    with (HERE/f'detections_{name}.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):groups[int(row['frame'])].append(row)
    cap=cv2.VideoCapture(str(source(name+'.mp4')))
    if not cap.isOpened():raise RuntimeError('Cannot read source')
    size=int(cap.get(3)),int(cap.get(4));fps=24 if name=='装甲板_蓝' else 150
    out=HERE/'figures'/f'detected_{name}.mp4';out.parent.mkdir(exist_ok=True)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    if not writer.isOpened():raise RuntimeError('Cannot write video')
    frame_id=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            for row in groups[frame_id]:
                points=np.array([[float(row[k+'_x']),float(row[k+'_y'])] for k in ['lt','lb','rb','rt']])
                cv2.polylines(frame,[points.astype(np.int32)],True,(0,255,0),2)
                for label,point in zip(['LT','LB','RB','RT'],points):
                    xy=tuple(point.astype(int));cv2.circle(frame,xy,3,(255,255,0),-1)
                    cv2.putText(frame,label,xy,0,.5,(0,255,0),1)
                if row['z']:
                    cv2.putText(frame,f"z={float(row['z']):.2f}m yaw={float(row['yaw']):.2f}",
                        tuple((points[0]-[0,15]).astype(int)),0,.45,(255,255,255),1)
            cv2.putText(frame,f'Frame {frame_id}  t={frame_id/fps:.3f}s  plates={len(groups[frame_id])}',
                        (15,30),0,.7,(255,255,255),2)
            writer.write(frame);frame_id+=1
    finally:cap.release();writer.release()
    print(f'Rebuilt {frame_id} frames from preserved detection CSV; no observation values modified.')

if __name__=='__main__':main()
