"""Interactive threshold tool: space pauses; S exports the current processing panel; Q exits."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from armor_step1 import detect, annotate
HERE=Path(__file__).resolve().parent


def panel(frame, parameters):
    mask,lights,armors=detect(frame,**parameters)
    result=annotate(frame,lights,armors)
    tiles=[cv2.resize(img,(640,480)) for img in (frame,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),result)]
    for image,title in zip(tiles,['Original','Red mask','Detected lights and armor']):
        cv2.putText(image,title,(12,28),0,.65,(255,255,255),2)
    header=np.zeros((70,1920,3),np.uint8)
    label='  '.join(f'{k}={v}' for k,v in parameters.items())
    cv2.putText(header,label,(15,28),0,.7,(255,255,255),1)
    cv2.putText(header,'SPACE: pause/play    S: save parameters + panel    Q: exit',(15,57),0,.65,(255,255,255),1)
    return np.vstack([header,np.hstack(tiles)])


def save(frame,parameters,config=None,panel_output=None):
    config=HERE/'thresholds.json' if config is None else Path(config)
    panel_output=HERE/'figures'/'threshold_panel.png' if panel_output is None else Path(panel_output)
    panel_output.parent.mkdir(parents=True,exist_ok=True)
    config.parent.mkdir(parents=True,exist_ok=True)
    (HERE/'figures').mkdir(exist_ok=True)
    cv2.imwrite(str(panel_output),panel(frame,parameters))
    config.write_text(json.dumps(parameters,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',default=str(HERE.parent/'data'/'armor'/'远距离平移.mp4'))
    parser.add_argument('--export-preview',action='store_true',help='Export a processing panel without opening the interactive window')
    parser.add_argument('--config',type=Path,default=HERE/'thresholds.json')
    parser.add_argument('--panel-output',type=Path,default=HERE/'figures'/'threshold_panel.png')
    parser.add_argument('--title',default='Threshold tuner')
    args=parser.parse_args()
    cap=cv2.VideoCapture(args.video)
    ok,frame=cap.read()
    if not ok: raise RuntimeError('Cannot read input video')
    parameters=dict(red_blue=45,red_green=25,brightness=110)
    config=args.config
    if config.exists(): parameters.update(json.loads(config.read_text(encoding='utf-8')))
    if args.export_preview:
        save(frame,parameters,args.config,args.panel_output); cap.release(); return
    window=args.title
    cv2.namedWindow(window,cv2.WINDOW_NORMAL); cv2.resizeWindow(window,1440,550)
    for name,value in parameters.items(): cv2.createTrackbar(name,window,value,255,lambda value:None)
    paused=True
    try:
        while True:
            parameters={name:cv2.getTrackbarPos(name,window) for name in parameters}
            cv2.imshow(window,panel(frame,parameters))
            key=cv2.waitKey(30)&255
            if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
            if key==32: paused=not paused
            if key==ord('s'): save(frame,parameters,args.config,args.panel_output)
            if not paused:
                ok,new=cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,new=cap.read()
                if ok: frame=new
    finally:
        cap.release(); cv2.destroyAllWindows()


if __name__=='__main__': main()
