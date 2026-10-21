"""YOLO preview with confidence/IoU sliders, four points and optional PnP."""
import argparse
import json
import cv2
import numpy as np
from yolo_detector import HERE, detector, annotate, config
from run_analysis import source, pose

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='中距离陀螺')
    args = parser.parse_args()
    name = args.video.removesuffix('.mp4')
    path = source(name+'.mp4')
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    if not ok: raise RuntimeError('Cannot read video')
    blue = name == '装甲板_蓝'
    if not blue:
        info = json.loads(source('camera_info.json').read_text(encoding='utf-8-sig'))
        K, D = np.array(info['camera_matrix']), np.array(info['distortion_coefficients'])
    settings = config(); window = 'YOLO armor - SPACE play/pause, S save, Q exit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL); cv2.resizeWindow(window, 1200, 850)
    cv2.createTrackbar('Confidence %', window, round(settings['confidence']*100), 100, lambda _: None)
    cv2.createTrackbar('NMS IoU %', window, round(settings['iou']*100), 100, lambda _: None)
    paused = True; index = 0; previous = None; dirty = True
    try:
        while True:
            settings = dict(confidence=max(.01,cv2.getTrackbarPos('Confidence %',window)/100),
                            iou=max(.01,cv2.getTrackbarPos('NMS IoU %',window)/100))
            if dirty or settings != previous:
                objects = detector().infer(frame, 'blue' if blue else 'red', **settings)
                view = annotate(frame, [], [o['points'] for o in objects])
                if not blue:
                    for obj in objects:
                        estimate = pose(obj['points'], K, D)
                        if estimate:
                            cv2.putText(view, f'z={estimate[1][2]:.2f}m yaw={estimate[2]:.2f}',
                                        tuple((obj['points'][1]+[0,20]).astype(int)),0,.55,(255,255,255),1)
                cv2.putText(view, f'YOLO ONNX | frame {index} | {detector().inference_ms:.1f} ms inference',
                            (15,30),0,.65,(255,255,255),2)
                cv2.imshow(window, view); previous = settings.copy(); dirty = False
            key = cv2.waitKey(20)&255
            if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
            if key == 32: paused = not paused
            if key == ord('s'):
                (HERE/'yolo_config.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
                cv2.imencode('.png',view)[1].tofile(HERE/'figures'/'yolo_preview.png')
                print('Saved YOLO confidence, IoU and screenshot',flush=True)
            if not paused:
                ok, new = cap.read(); index += 1
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,new=cap.read();index=0
                if not ok: break
                frame = new; dirty = True
    finally: cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__': main()
