"""Compare a conservative endpoint/PnP refinement on exactly the same spin candidates.
Existing detection files, translation settings, and color thresholds are not changed.
"""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
import csv
import json
import hashlib
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_analysis import source,pose,write_csv
from spin_pose_refined import refine_points,solve_refined


def read_points(row):
    return np.array([[float(row[k+'_x']),float(row[k+'_y'])] for k in ['lt','lb','rb','rt']],np.float64)


def draw(frame,points,estimates,header,show_pose=True):
    view=frame.copy()
    for pts,est in zip(points,estimates):
        cv2.polylines(view,[np.rint(pts).astype(np.int32)],True,(0,255,0),1)
        for label,pt in zip(['LT','LB','RB','RT'],pts):
            xy=tuple(np.rint(pt).astype(int));cv2.circle(view,xy,2,(255,255,0),-1)
            cv2.putText(view,label,xy,0,.35,(0,255,0),1)
        if est is not None:
            text=f'z={est[1][2]:.2f}m yaw={est[2]:.2f}'
        else:text='POSE FAILED'
        if show_pose:
            xy=tuple(np.rint(pts[0]-[0,12]).astype(int));cv2.putText(view,text,xy,0,.4,(255,255,255),1)
    cv2.putText(view,header,(15,30),0,.65,(255,255,255),2)
    return view


def letterbox(image,width=760,height=255):
    canvas=np.zeros((height,width,3),np.uint8)
    scale=min(width/image.shape[1],height/image.shape[0]);w=max(1,int(image.shape[1]*scale));h=max(1,int(image.shape[0]*scale))
    resized=cv2.resize(image,(w,h),interpolation=cv2.INTER_NEAREST)
    canvas[(height-h)//2:(height-h)//2+h,(width-w)//2:(width-w)//2+w]=resized
    return canvas


def main():
    baseline=HERE/'detections_中距离陀螺.csv'
    rows=list(csv.DictReader(baseline.open(encoding='utf-8-sig')))
    groups={}
    for row in rows:groups.setdefault(int(row['frame']),[]).append(row)
    out=HERE/'spin_refined';out.mkdir(exist_ok=True)
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    # Snapshot the inputs used for this comparison.
    (out/'baseline_detections.csv').write_bytes(baseline.read_bytes())
    thresholds=json.loads((HERE/'thresholds_spin.json').read_text(encoding='utf-8-sig'))
    info=json.loads(source('camera_info.json').read_text(encoding='utf-8-sig'))
    K=np.array(info['camera_matrix'],np.float64);D=np.array(info['distortion_coefficients'],np.float64)
    cap=cv2.VideoCapture(str(source('中距离陀螺.mp4')))
    width,height=int(cap.get(3)),int(cap.get(4))
    if (width,height)!=(info['image_width'],info['image_height']):raise ValueError('Video/calibration size mismatch')
    video=figures/'detected_中距离陀螺_refined.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),150.,(width,height))
    if not writer.isOpened():raise RuntimeError('Cannot create output video')
    new_rows=[];comparison=[];old_positions=[];new_positions=[];panels=[]
    selected={0,39,100,106,584,654}
    index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame_rows=groups.get(index,[]);old_points=[];new_points=[];old_estimates=[];new_estimates=[]
            for plate_index,row in enumerate(frame_rows):
                original=read_points(row);refined=refine_points(frame,original,thresholds,'row_center')
                old=pose(original,K,D);new=solve_refined(refined,K,D)
                old_points.append(original);new_points.append(refined);old_estimates.append(old);new_estimates.append(new)
                new_rows.append([index,*refined.ravel(),*([*new[1],new[2]] if new else [np.nan]*4)])
                comparison.append([index,plate_index,old[0] if old else np.nan,new[0] if new else np.nan,
                                   old[1][2] if old else np.nan,new[1][2] if new else np.nan,
                                   int(old is not None),int(new is not None),float(np.mean(np.linalg.norm(refined-original,axis=1)))])
                old_positions.append([index,*old[1],old[2]] if old else [index,*[np.nan]*4])
                new_positions.append([index,*new[1],new[2]] if new else [index,*[np.nan]*4])
            new_view=draw(frame,new_points,new_estimates,f'Refined endpoint + PnP LM | frame {index} | t={index/150:.3f}s')
            writer.write(new_view)
            if index in selected and old_points:
                old_view=draw(frame,old_points,old_estimates,'',False)
                crop_new_view=draw(frame,new_points,new_estimates,'',False)
                xy=np.vstack(old_points+new_points)
                x0=max(0,int(xy[:,0].min())-35);x1=min(width,int(xy[:,0].max())+35)
                y0=max(0,int(xy[:,1].min())-35);y1=min(height,int(xy[:,1].max())+25)
                before=letterbox(old_view[y0:y1,x0:x1]);after=letterbox(crop_new_view[y0:y1,x0:x1])
                heading=np.zeros((35,1520,3),np.uint8)
                cv2.putText(heading,f'Frame {index} - BEFORE',(10,24),0,.65,(255,255,255),1)
                cv2.putText(heading,'AFTER: same frame and same crop',(770,24),0,.65,(255,255,255),1)
                panels.append(np.vstack([heading,np.hstack([before,after])]))
            index+=1
            if index%500==0:print(f'Compared {index} frames',flush=True)
    finally:cap.release();writer.release()
    if index<=max(groups):raise RuntimeError('Video ended before all baseline detections were compared')
    write_csv(out/'detections_中距离陀螺.csv',['frame','lt_x','lt_y','lb_x','lb_y','rb_x','rb_y','rt_x','rt_y','x','y','z','yaw'],new_rows)
    write_csv(out/'candidate_comparison.csv',['frame','candidate_in_frame','old_reprojection_px','new_reprojection_px','old_z_m','new_z_m','old_pose_valid','new_pose_valid','mean_keypoint_change_px'],comparison)
    a=np.array(comparison);common=np.isfinite(a[:,2])&np.isfinite(a[:,3])
    old_valid=a[:,6].astype(bool);new_valid=a[:,7].astype(bool)
    def metrics(valid,error_col,z_col):
        return dict(valid_poses=int(valid.sum()),failed_poses=int((~valid).sum()),reprojection_median_px=float(np.median(a[valid,error_col])),reprojection_p95_px=float(np.percentile(a[valid,error_col],95)),depth_p01_m=float(np.percentile(a[valid,z_col],1)),depth_median_m=float(np.median(a[valid,z_col])),depth_p99_m=float(np.percentile(a[valid,z_col],99)))
    summary=dict(frames=index,candidates=len(a),same_candidates=True,baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),thresholds=thresholds,
                 before=metrics(old_valid,2,4),after=metrics(new_valid,3,5),
                 common_poses=int(common.sum()),common_before_median_px=float(np.median(a[common,2])),common_after_median_px=float(np.median(a[common,3])),
                 recovered_poses=int((~old_valid&new_valid).sum()),lost_poses=int((old_valid&~new_valid).sum()),
                 mean_keypoint_change_px=float(np.mean(a[:,8])),
                 note='No independent ground truth. Reprojection is a fitting residual, not distance accuracy. Depth tails remain; no forced circle, fixed-depth constraint, smoothing, or post-hoc point removal.')
    (out/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if panels:cv2.imwrite(str(figures/'spin_endpoints_before_after.jpg'),np.vstack(panels))
    old_data=np.array(old_positions);new_data=np.array(new_positions)
    # Identical axis limits avoid making one distribution look artificially tighter.
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharey='row')
    finite_z=np.r_[old_data[np.isfinite(old_data[:,3]),3],new_data[np.isfinite(new_data[:,3]),3]]
    for col,(data,title) in enumerate([(old_data,'Before'),(new_data,'After: row-center endpoints + PnP LM')]):
        valid=np.isfinite(data[:,3])
        dots=axes[0,col].scatter(data[valid,1],data[valid,3],c=data[valid,0]/150,s=3,cmap='viridis',vmin=0,vmax=(index-1)/150,alpha=.6)
        axes[0,col].set(title=title,xlabel='x (m)',ylabel='z (m)',xlim=(-.4,.3),ylim=(finite_z.min()-.03,finite_z.max()+.03));axes[0,col].grid(alpha=.25)
        axes[1,col].scatter(data[valid,0]/150,data[valid,3],s=2,alpha=.5)
        axes[1,col].set(xlabel='Time (s)',ylabel='z (m)');axes[1,col].grid(alpha=.25)
    fig.suptitle('Same spin-video candidates: position comparison (no ground truth)')
    fig.tight_layout();fig.savefig(figures/'spin_pose_before_after.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.hist(a[common,2],bins=np.linspace(0,3,31),histtype='step',lw=2,label='Before')
    ax.hist(a[common,3],bins=np.linspace(0,3,31),histtype='step',lw=2,label='After')
    ax.set(xlabel='Reprojection RMSE (pixels)',ylabel='Matched candidates',title='Fitting residual on the same valid candidates; not range accuracy')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figures/'spin_reprojection_before_after.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
