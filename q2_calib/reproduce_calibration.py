"""Recalibrate from the preserved 20 frames, without camera capture."""
from pathlib import Path
import argparse
import json
import csv
import cv2
import numpy as np
from calibrate_usb import detect

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=HERE/'calibration_runs'/'20261016_192448_766416')
    args=parser.parse_args()
    saved=json.loads((args.run/'calibration.json').read_text(encoding='utf-8'))
    pattern=tuple(saved['inner_corners_cols_rows']);size=tuple(saved['image_size_wh'])
    images=sorted((args.run/'images').glob('*.png'))
    obj=np.zeros((pattern[0]*pattern[1],3),np.float32)
    obj[:,:2]=np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)*saved['square_size_mm']
    points=[];names=[];out=HERE/'reproduced_calibration';(out/'corners').mkdir(parents=True,exist_ok=True)
    for file in images:
        image=cv2.imdecode(np.fromfile(file,np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1]!=size:raise ValueError(f'Invalid input: {file.name}')
        found,corners=detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),pattern)
        if not found:raise ValueError(f'Corners missing: {file.name}')
        points.append(corners);names.append(file.name)
        cv2.drawChessboardCorners(image,pattern,corners,True)
        cv2.imencode('.png',image)[1].tofile(out/'corners'/file.name)
    rms,K,D,rvecs,tvecs=cv2.calibrateCamera([obj.copy() for _ in points],points,size,None,None)
    errors=[]
    for name,observed,r,t in zip(names,points,rvecs,tvecs):
        projected,_=cv2.projectPoints(obj,r,t,K,D)
        errors.append([name,float(np.sqrt(np.mean(np.sum((observed.reshape(-1,2)-projected.reshape(-1,2))**2,axis=1))))])
    result=dict(image_count=len(points),image_size_wh=size,inner_corners_cols_rows=pattern,
        square_size_mm=saved['square_size_mm'],opencv_rms_px=float(rms),camera_matrix=K.tolist(),
        distortion_coefficients=D.ravel().tolist(),
        maximum_K_difference_from_saved_px=float(np.max(abs(K-np.array(saved['camera_matrix'])))),
        note='Recomputed from preserved images, not new captures; corner detections can differ by OpenCV version.')
    (out/'calibration.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'per_image_errors.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['image','rmse_px']);writer.writerows(errors)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
