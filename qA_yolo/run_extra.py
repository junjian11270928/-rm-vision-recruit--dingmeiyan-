"""Additional tasks: near red spin pose and blue armor recognition only."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
import argparse
import json
import cv2
import numpy as np
from armor_step1 import detect,annotate
from blue_detector import detect_blue
from run_analysis import source,pose,write_csv


def process_extra(name):
    blue=name=='装甲板_蓝'
    from yolo_detector import config
    default={}
    if not blue:
        info=json.loads(source('camera_info.json').read_text(encoding='utf-8-sig'))
        K=np.array(info['camera_matrix'],np.float64);D=np.array(info['distortion_coefficients'],np.float64)
    cap=cv2.VideoCapture(str(source(name+'.mp4')))
    if not cap.isOpened():raise RuntimeError('Cannot open '+name)
    width,height=int(cap.get(3)),int(cap.get(4));reported=cap.get(cv2.CAP_PROP_FPS)
    fps=reported if blue else 150.
    if not blue and (width,height)!=(1440,1080):raise ValueError('Calibration resolution mismatch')
    frames_meta=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fig=HERE/'figures';fig.mkdir(exist_ok=True)
    writer=cv2.VideoWriter(str(fig/f'detected_{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('Cannot create video')
    selected=set(int(v) for v in np.linspace(0,frames_meta-1,8,dtype=int))
    rows=[];quality=[];frame_stats=[];tiles=[];index=0;found=0;solved=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            mask,lights,armors=(detect_blue(frame,**default) if blue else detect(frame,**default))
            view=annotate(frame,lights,armors);valid_frame=0
            for points in armors:
                estimate=None if blue else pose(points,K,D)
                values=[np.nan]*4 if estimate is None else [*estimate[1],estimate[2]]
                rows.append([index,*points.ravel(),*values])
                if estimate is not None:
                    solved+=1;valid_frame+=1;quality.append([index,estimate[0]])
                    xy=tuple(np.rint(points[0]-[0,15]).astype(int))
                    cv2.putText(view,f'z={estimate[1][2]:.2f}m yaw={estimate[2]:.2f}',xy,0,.45,(255,255,255),1)
            found+=bool(armors);frame_stats.append([index,index/fps,len(lights),len(armors),valid_frame])
            cv2.putText(view,f'Frame {index} t={index/fps:.3f}s plates={len(armors)}',(12,28),0,.6,(255,255,255),1)
            if blue:cv2.putText(view,'DETECTION ONLY - no camera calibration',(12,52),0,.5,(255,255,255),1)
            writer.write(view)
            if index in selected:tiles.append(cv2.resize(view,(640,480)))
            index+=1
            if index%500==0:print(f'{name}: {index} frames',flush=True)
    finally:cap.release();writer.release()
    if index!=frames_meta:raise RuntimeError(f'Incomplete decoding: {index}/{frames_meta}')
    write_csv(HERE/f'detections_{name}.csv',['frame','lt_x','lt_y','lb_x','lb_y','rb_x','rb_y','rt_x','rt_y','x','y','z','yaw'],rows)
    if not blue:write_csv(HERE/f'quality_{name}.csv',['frame','reprojection_rmse_px'],quality)
    write_csv(HERE/f'frame_status_{name}.csv',['frame','t','lights','armor_candidates','valid_poses'],frame_stats)
    if len(tiles)==8:
        montage=np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])])
        encoded_ok,encoded=cv2.imencode('.jpg',montage)
        if not encoded_ok:raise RuntimeError('Cannot encode contact sheet')
        encoded.tofile(fig/f'contact_{name}.jpg')
    result=dict(frames=index,width=width,height=height,fps=fps,frames_with_candidates=found,candidate_rows=len(rows),valid_pose_rows=None if blue else solved,failed_pose_rows=None if blue else len(rows)-solved,pose_enabled=not blue,yolo_settings=config(),detector='YOLO ONNX exp-2',
        notes='Candidate coverage is not precision or recall. Hidden/occluded plates are not extrapolated. Blue pose columns intentionally blank.' if blue else 'No independent ground truth. Raw pose rows can include perspective/PnP noise; identity tracking not implemented.')
    if quality:result['reprojection_median_px']=float(np.median([r[1] for r in quality]))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',choices=['近距离陀螺','装甲板_蓝','all'],default='all');args=parser.parse_args()
    path=HERE/'extra_summary.json';report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for name in (['近距离陀螺','装甲板_蓝'] if args.video=='all' else [args.video]):
        report[name]=process_extra(name)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
