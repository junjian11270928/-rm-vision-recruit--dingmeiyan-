"""Offline competition-video analysis. Units: m, rad, s; camera x right/y down/z forward."""
from pathlib import Path
import argparse
import csv
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from armor_step1 import detect, annotate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OBJECT = np.array([[-.0675,-.0275,0],[-.0675,.0275,0],[.0675,.0275,0],[.0675,-.0275,0]], np.float64)
FPS = 150.0


def source(name):
    return next(ROOT.rglob(name))


def pose(points, K, distortion):
    result = cv2.solvePnPGeneric(OBJECT, points.astype(np.float64), K, distortion, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rvec, tvec in zip(result[1], result[2]):
        R, _ = cv2.Rodrigues(rvec)
        if np.min((OBJECT @ R.T + tvec.reshape(3))[:,2]) <= 0:
            continue
        projected, _ = cv2.projectPoints(OBJECT, rvec, tvec, K, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-points)**2,axis=1))))
        if error > 3:
            continue
        yaw = float(np.arctan2(R[0,2], R[2,2]))
        candidate = (error, tvec.reshape(3), yaw)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(['' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for v in row])


def process(name, K, D, thresholds):
    cap = cv2.VideoCapture(str(source(name+'.mp4')))
    if not cap.isOpened():
        raise RuntimeError('Cannot open '+name)
    width, height = int(cap.get(3)), int(cap.get(4))
    if (width,height) != (1440,1080):
        raise ValueError('Camera calibration requires 1440x1080 input')
    video_path = HERE/'figures'/('detected_'+name+'.mp4')
    writer = cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(width,height))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable')
    rows, quality, observations, tiles = [], [], [], []
    frame_id, found, solved = 0, 0, 0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples=set(np.linspace(0,total-1,8,dtype=int))
    last=None
    last_frame=-1000
    try:
        while True:
            ok, frame=cap.read()
            if not ok: break
            _, lights, armors=detect(frame,**thresholds)
            found+=bool(armors)
            results=[]
            for points in armors:
                estimate=pose(points,K,D)
                values=[np.nan]*4 if estimate is None else [*estimate[1],estimate[2]]
                rows.append([frame_id,*points.reshape(-1).tolist(),*values])
                if estimate is not None:
                    solved+=1
                    quality.append([frame_id,estimate[0]])
                    results.append((points,estimate))
            observation=np.full(3,np.nan)
            if results:
                # Translation has one main front-facing plate. Associate by previous 3D location.
                if last is None or frame_id-last_frame>75:
                    chosen=max(results,key=lambda item:cv2.contourArea(item[0].astype(np.float32)))
                else:
                    chosen=min(results,key=lambda item:np.linalg.norm(item[1][1]-last))
                if last is None or frame_id-last_frame>75 or np.linalg.norm(chosen[1][1]-last)<.6:
                    observation=chosen[1][1]
                    last=observation.copy(); last_frame=frame_id
            observations.append(observation)
            view=annotate(frame,lights,armors)
            cv2.putText(view,f'Frame {frame_id}  t={frame_id/FPS:.3f}s  plates={len(armors)}',(20,40),0,.8,(255,255,255),2)
            for points,estimate in results:
                x,y,z=estimate[1]
                xy=tuple(points[0].astype(int)+np.array([0,-15]))
                cv2.putText(view,f'z={z:.2f}m yaw={estimate[2]:.2f}',xy,0,.45,(255,255,255),1)
            writer.write(view)
            if frame_id in samples:
                tiles.append(cv2.resize(view,(720,540)))
            frame_id+=1
            if frame_id%500==0: print(name,frame_id,'frames',flush=True)
    finally:
        cap.release(); writer.release()
    write_csv(HERE/('detections_'+name+'.csv'),['frame','lt_x','lt_y','lb_x','lb_y','rb_x','rb_y','rt_x','rt_y','x','y','z','yaw'],rows)
    write_csv(HERE/('quality_'+name+'.csv'),['frame','reprojection_rmse_px'],quality)
    if len(tiles)==8:
        cv2.imwrite(str(HERE/'figures'/('contact_'+name+'.jpg')),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
    # These are coverage figures, not precision/recall: no ground truth labels exist.
    return np.array(observations),dict(frames=frame_id,frames_with_candidates=found,candidate_rows=len(rows),valid_pose_rows=solved,frame_coverage=found/max(frame_id,1),reprojection_median_px=float(np.median([r[1] for r in quality])) if quality else None)


def predict(observations, acceleration_noise=3., measurement_noise=.03):
    dt=1/FPS
    F=np.eye(6); F[:3,3:]=np.eye(3)*dt
    G=np.vstack([np.eye(3)*dt*dt/2,np.eye(3)*dt])
    Q=G@G.T*acceleration_noise**2
    H=np.hstack([np.eye(3),np.zeros((3,3))]); R=np.eye(3)*measurement_noise**2
    state=None; covariance=np.eye(6)
    predicted=np.full_like(observations,np.nan); filtered=predicted.copy()
    accepted=np.zeros(len(observations),bool); last=-1000
    for i,obs in enumerate(observations):
        if state is not None:
            state=F@state; covariance=F@covariance@F.T+Q
        if np.isfinite(obs).all():
            if state is None or i-last>75:
                state=np.r_[obs,[0.,0.,0.]]; covariance=np.diag([.01]*3+[4.]*3)
                accepted[i]=True; last=i
            else:
                innovation=obs-H@state
                S=H@covariance@H.T+R
                # Reject severe jumps; retain raw measurements in exported CSV.
                if innovation@np.linalg.solve(S,innovation)<25:
                    gain=np.linalg.solve(S,H@covariance).T
                    state=state+gain@innovation
                    A=np.eye(6)-gain@H
                    covariance=A@covariance@A.T+gain@R@gain.T
                    accepted[i]=True; last=i
        if state is not None and i-last<=30:
            filtered[i]=state[:3]
            predicted[i]=state[:3]+.25*state[3:]
    return filtered,predicted,accepted


def plots(observations):
    t=np.arange(len(observations))/FPS
    filtered,prediction,accepted=predict(observations)
    write_csv(HERE/'pred_远距离平移.csv',['t','x','y','z'],[[time,*p] for time,p in zip(t,prediction)])
    write_csv(HERE/'filter_diagnostics.csv',['frame','t','obs_x','obs_y','obs_z','filtered_x','filtered_y','filtered_z','accepted'],[[i,t[i],*observations[i],*filtered[i],int(accepted[i])] for i in range(len(t))])
    future=np.full_like(observations,np.nan)
    # 0.25 s = 37.5 frames. Only interpolate between two adjacent valid future observations.
    if len(t)>38:
        future[:-38]=(observations[37:-1]+observations[38:])/2
    error=prediction-future
    valid=np.isfinite(error).all(axis=1) & (t>=.15)
    if not np.any(valid): raise RuntimeError('No valid prediction comparison samples')
    distances=np.linalg.norm(error[valid],axis=1)
    stats=dict(comparison='prediction_at_t_vs_interpolated_observation_at_t_plus_0.25s',ground_truth=False,interpolation='mean of adjacent frames f+37 and f+38; both must be valid',startup_excluded_s=.15,valid_comparisons=int(valid.sum()),mean_euclidean_error_m=float(np.mean(distances)),rmse_euclidean_m=float(np.sqrt(np.mean(distances**2))),p95_euclidean_error_m=float(np.percentile(distances,95)),axis_rmse_m=np.sqrt(np.mean(error[valid]**2,axis=0)).tolist(),rejected_observations=int((np.isfinite(observations).all(axis=1)&~accepted).sum()),Q_acceleration_std_m_s2=3.,R_measurement_std_m=.03)
    write_csv(HERE/'prediction_comparison.csv',['t','pred_x','pred_y','pred_z','future_obs_x','future_obs_y','future_obs_z','euclidean_error_m'],[[t[i],*prediction[i],*future[i],np.linalg.norm(error[i])] for i in np.flatnonzero(valid)])
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t,observations[:,j],'.',ms=2,alpha=.5,label='PnP observation')
        ax.set_ylabel('xyz'[j]+' (m)'); ax.grid(alpha=.3); ax.legend()
    axes[-1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(HERE/'figures'/'position_time.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(observations[:,0],observations[:,2],'.',ms=2); ax.set(xlabel='x (m)',ylabel='z (m)',title='Top view: observed plate trajectory'); ax.grid(alpha=.3); ax.axis('equal'); fig.tight_layout(); fig.savefig(HERE/'figures'/'trajectory_xz.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t,observations[:,j],'.',ms=2,alpha=.3,label='Observation at t')
        ax.plot(t,filtered[:,j],lw=1,label='Filtered at t')
        ax.plot(t,prediction[:,j],lw=1,label='Forecast for t+0.25s')
        ax.set_ylabel('xyz'[j]+' (m)'); ax.grid(alpha=.3); ax.legend()
    axes[-1].set_xlabel('Forecast issue time t (s)'); fig.tight_layout(); fig.savefig(HERE/'figures'/'filter_prediction.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
    for j,ax in enumerate(axes[:3]):
        ax.plot(t,prediction[:,j],lw=1,label='Forecast for t+0.25s')
        ax.plot(t,future[:,j],lw=1,alpha=.7,label='Observation at t+0.25s')
        ax.set_ylabel('xyz'[j]+' (m)'); ax.grid(alpha=.3); ax.legend()
    axes[3].plot(t[valid],distances); axes[3].set(xlabel='Forecast issue time t (s)',ylabel='3D error (m)'); axes[3].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE/'figures'/'prediction_comparison.png',dpi=160); plt.close(fig)
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',choices=['远距离平移','中距离陀螺','all'],default='all')
    args=parser.parse_args()
    (HERE/'figures').mkdir(exist_ok=True)
    info=json.loads(source('camera_info.json').read_text(encoding='utf-8-sig'))
    K=np.array(info['camera_matrix'],np.float64); D=np.array(info['distortion_coefficients'],np.float64)
    thresholds=dict(red_blue=45,red_green=25,brightness=110)
    config=HERE/'thresholds.json'
    if config.exists(): thresholds.update(json.loads(config.read_text(encoding='utf-8')))
    report={}
    for name in (['远距离平移','中距离陀螺'] if args.video=='all' else [args.video]):
        observations,summary=process(name,K,D,thresholds)
        report[name]=summary
        if name=='远距离平移': report['prediction']=plots(observations)
    (HERE/'run_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__': main()
