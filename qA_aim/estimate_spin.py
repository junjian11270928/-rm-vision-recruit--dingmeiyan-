"""Causal rotating-rig estimator from armor poses.
Assumptions: four equally spaced plates, shared effective radius, approximately
horizontal rotation. Center height is defined at the mean armor-center plane.
No fixed center, fixed speed, or future samples are inserted as observations.
"""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
import argparse
import csv
import json
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from run_analysis import write_csv

CONFIG=dict(plates=4,fps=150.,initialization_s=.6,center_window_s=.6,
            phase_measurement_std_rad=.12,angular_acceleration_density=4.,
            center_position_std_m=[.02,.02,.08],minimum_center_samples=20,
            min_radius_m=.05,max_radius_m=.8,max_unobserved_s=.2)


def wrapped(value,period):
    return (value+period/2)%period-period/2


def initial_phase(history,now,config):
    n=config['plates'];times=np.array([r[0] for r in history]);values=np.array([r[1] for r in history])
    # Circular regression handles plate changes without a persistent plate ID.
    # Only the initialization window available at the current time is used.
    omegas=np.arange(-60.,60.001,.1)
    score=np.array([abs(np.sum(values*np.exp(-1j*n*om*(times-now)))) for om in omegas])
    idx=int(np.argmax(score));omega=float(omegas[idx])
    fine=np.linspace(omega-.1,omega+.1,41)
    scores=np.array([abs(np.sum(values*np.exp(-1j*n*om*(times-now)))) for om in fine])
    omega=float(fine[int(np.argmax(scores))])
    phasor=np.sum(values*np.exp(-1j*n*omega*(times-now)))
    theta=float(np.angle(phasor)/n)
    coherence=float(abs(phasor)/max(np.sum(np.abs(values)),1e-9))
    return theta,omega,coherence


def fit_center(records,theta,omega,now,config):
    if len(records)<config['minimum_center_samples']:return None
    a=np.array(records,float)
    times=a[:,0];positions=a[:,1:4];raw_yaw=a[:,4];quality=a[:,5]
    period=2*np.pi/config['plates']
    predicted=theta+omega*(times-now)
    k=np.rint((raw_yaw-predicted)/period)
    angle=predicted+k*period
    # If the four-plate phase model contradicts an individual yaw substantially,
    # downweight it rather than silently changing the observed position.
    phase_error=wrapped(raw_yaw-angle,2*np.pi)
    qweight=1/(1+quality**2)/(1+(phase_error/.25)**2)
    n=len(a)
    design=np.zeros((n,3,4));design[:,0,0]=1;design[:,1,1]=1;design[:,2,2]=1
    design[:,0,3]=-np.sin(angle);design[:,2,3]=-np.cos(angle)
    A=design.reshape(-1,4);b=positions.reshape(-1)
    sigma=np.array(config['center_position_std_m'])
    robust=np.ones(n)
    for _ in range(6):
        weights=(qweight*robust)[:,None]/sigma[None,:]**2
        sqrtw=np.sqrt(weights.ravel())
        coefficients,_,rank,_=np.linalg.lstsq(A*sqrtw[:,None],b*sqrtw,rcond=None)
        if rank<4:return None
        residual=positions-(A@coefficients).reshape(n,3)
        norm=np.linalg.norm(residual/sigma,axis=1)
        robust=np.minimum(1.,2.5/np.maximum(norm,1e-9))
    radius=float(coefficients[3])
    if not config['min_radius_m']<radius<config['max_radius_m']:return None
    residual=positions-(A@coefficients).reshape(n,3)
    rms=float(np.sqrt(np.average(np.sum(residual**2,axis=1),weights=qweight*robust)))
    return coefficients[:3],radius,rms,float(np.median(abs(phase_error))),n


def estimate(groups,total_frames,config=None):
    config=CONFIG.copy() if config is None else config.copy()
    fps=config['fps'];dt=1/fps;n=config['plates'];period=2*np.pi/n
    state=None;P=np.diag([.05**2,2.**2]);F=np.array([[1.,dt],[0.,1.]])
    Q=config['angular_acceleration_density']*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
    buffer=deque();phase_history=[];last_seen=-1000.;init_coherence=None
    estimates=[];diagnostics=[];phase_series=[]
    for frame in range(total_frames):
        now=frame/fps;observations=groups.get(frame,[])
        if state is not None:state=F@state;P=F@P@F.T+Q
        phasor=None;coherence=np.nan;innovation=np.nan
        if observations:
            obs=np.array(observations,float)
            # observation = [x,y,z,yaw,reprojection error]
            weight=1/(1+obs[:,4]**2)
            circular=np.sum(weight*np.exp(1j*n*obs[:,3]))/weight.sum()
            coherence=float(abs(circular))
            if coherence>.2:
                phasor=circular;last_seen=now
                if state is None:phase_history.append((now,circular))
            for o in obs:buffer.append([now,*o])
        while buffer and buffer[0][0]<now-config['center_window_s']:buffer.popleft()
        if state is None and phasor is not None and phase_history and now-phase_history[0][0]>=config['initialization_s']:
            theta,omega,init_coherence=initial_phase(phase_history,now,config)
            state=np.array([theta,omega]);phase_history=[]
        elif state is not None and phasor is not None:
            measured=float(np.angle(phasor)/n)
            innovation=float(wrapped(measured-state[0],period))
            R=config['phase_measurement_std_rad']**2/max(coherence**2,.1)
            S=float(P[0,0]+R)
            inflation=max(1.,innovation**2/(9*S))
            R*=min(inflation,20.)
            gain=P[:,0]/(P[0,0]+R)
            state=state+gain*innovation
            A=np.eye(2)-np.outer(gain,[1.,0.])
            P=A@P@A.T+np.outer(gain,gain)*R;P=(P+P.T)/2
        center=np.full(3,np.nan);radius=np.nan;fit_rms=np.nan;phase_error=np.nan;samples=0
        omega=np.nan;theta=np.nan
        if state is not None and now-last_seen<=config['max_unobserved_s']:
            theta=float(state[0]);omega=float(state[1])
            if abs(omega)<60:
                fitted=fit_center(buffer,theta,omega,now,config)
                if fitted is not None:center,radius,fit_rms,phase_error,samples=fitted
        estimates.append([now,*center,omega])
        diagnostics.append([frame,now,radius,fit_rms,coherence,innovation,phase_error,samples,int(np.isfinite(center).all()),int(np.isfinite(omega))])
        phase_series.append([now,theta])
    return np.array(estimates),np.array(diagnostics),np.array(phase_series),init_coherence


def load_data(name):
    rows=list(csv.DictReader((HERE/f'detections_{name}.csv').open(encoding='utf-8-sig')))
    quality=list(csv.DictReader((HERE/f'quality_{name}.csv').open(encoding='utf-8-sig')))
    groups={};raw=[];index=0
    for row in rows:
        if not all(row[k] for k in ['x','y','z','yaw']):continue
        q=quality[index];index+=1
        if int(q['frame'])!=int(row['frame']):raise ValueError('Quality row alignment mismatch')
        frame=int(row['frame']);obs=[float(row[k]) for k in ['x','y','z','yaw']]+[float(q['reprojection_rmse_px'])]
        groups.setdefault(frame,[]).append(obs);raw.append([frame/CONFIG['fps'],*obs])
    if index!=len(quality):raise ValueError('Quality row count mismatch')
    metadata=HERE/('run_summary.json' if name=='中距离陀螺' else 'extra_summary.json')
    total=int(json.loads(metadata.read_text(encoding='utf-8-sig'))[name]['frames'])
    if max(groups)>=total:raise ValueError('Frame range mismatch')
    return groups,total,np.array(raw)


def make_plots(name,estimates,diagnostics,phase,raw):
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    t=estimates[:,0];valid=np.isfinite(estimates[:,1:4]).all(axis=1)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for j,ax in enumerate(axes):
        ax.scatter(raw[:,0],raw[:,j+1],s=3,alpha=.3,label='装甲板PnP观测')
        ax.plot(t,estimates[:,j+1],color='#d34432',lw=1.5,label='滑动窗口车体中心估计')
        ax.set_ylabel('xyz'[j]+'（米）');ax.grid(alpha=.25);ax.legend(loc='upper right',fontsize=9)
    axes[-1].set_xlabel('时间（秒）');fig.suptitle(name+'：装甲板位置与车体中心估计\n四块等间隔装甲板、共享有效半径的近似模型；无独立中心真值')
    fig.tight_layout();fig.savefig(figures/f'center_comparison_{name}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(t,estimates[:,4],color='#2674b2');axes[0].set_ylabel('角速度（rad/s）');axes[0].grid(alpha=.25)
    axes[1].plot(phase[:,0],phase[:,1],color='#548d48');axes[1].set_ylabel('累计模型相位（rad）');axes[1].set_xlabel('时间（秒）');axes[1].grid(alpha=.25)
    fig.suptitle(name+'：旋转角速度与连续相位\n角速度带符号；开始0.6秒用于初始化，未倒填结果')
    fig.tight_layout();fig.savefig(figures/f'angular_velocity_{name}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7))
    ax.scatter(raw[:,1],raw[:,3],s=3,alpha=.25,label='装甲板PnP观测')
    ax.plot(estimates[valid,1],estimates[valid,3],'.',ms=3,color='#d34432',label='各时刻估计的车体中心')
    ax.set(xlabel='x（米）',ylabel='z（米）',title=name+'：俯视观测与中心估计（同坐标系）');ax.grid(alpha=.25);ax.legend();ax.set_aspect('equal',adjustable='datalim')
    fig.tight_layout();fig.savefig(figures/f'center_topview_{name}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    axes[0].plot(t,diagnostics[:,2]);axes[0].set_ylabel('有效半径（米）')
    axes[1].plot(t,diagnostics[:,3]);axes[1].set_ylabel('模型拟合RMS（米）')
    axes[2].plot(t,diagnostics[:,4]);axes[2].set_ylabel('同帧相位一致性');axes[2].set_xlabel('时间（秒）')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle(name+'：模型诊断（不是位置精度评估）');fig.tight_layout();fig.savefig(figures/f'spin_diagnostics_{name}.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',choices=['中距离陀螺','近距离陀螺','all'],default='all');args=parser.parse_args()
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    report={}
    for name in (['中距离陀螺','近距离陀螺'] if args.video=='all' else [args.video]):
        groups,total,raw=load_data(name)
        estimates,diagnostics,phase,coherence=estimate(groups,total)
        write_csv(HERE/f'pred_{name}.csv',['t','cx','cy','cz','omega'],estimates)
        write_csv(HERE/f'spin_state_diagnostics_{name}.csv',['frame','t','radius_m','model_fit_rms_m','frame_phase_coherence','phase_innovation_rad','median_phase_mismatch_rad','window_samples','center_valid','omega_valid'],diagnostics)
        write_csv(HERE/f'spin_phase_{name}.csv',['t','phase_rad'],phase)
        make_plots(name,estimates,diagnostics,phase,raw)
        valid=np.isfinite(estimates[:,1:4]).all(axis=1);omega=np.isfinite(estimates[:,4])
        center=estimates[valid,1:4];speed=estimates[omega,4]
        summary=dict(frames=total,valid_center_frames=int(valid.sum()),valid_omega_frames=int(omega.sum()),
            first_valid_center_s=float(estimates[np.flatnonzero(valid)[0],0]) if valid.any() else None,
            center_median_m=np.median(center,axis=0).tolist() if valid.any() else None,
            center_std_m=np.std(center,axis=0).tolist() if valid.any() else None,
            center_p05_m=np.percentile(center,5,axis=0).tolist() if valid.any() else None,
            center_p95_m=np.percentile(center,95,axis=0).tolist() if valid.any() else None,
            omega_median_rad_s=float(np.median(speed)) if omega.any() else None,
            omega_p05_rad_s=float(np.percentile(speed,5)) if omega.any() else None,
            omega_p95_rad_s=float(np.percentile(speed,95)) if omega.any() else None,
            median_abs_turns_per_s=float(np.median(abs(speed))/(2*np.pi)) if omega.any() else None,
            radius_median_m=float(np.nanmedian(diagnostics[:,2])),
            model_fit_median_m=float(np.nanmedian(diagnostics[:,3])),
            init_phase_coherence=coherence,config=CONFIG,
            caveat='Model-dependent estimates, not ground truth. Center height is mean armor-center plane. Four equal angular spacings/shared radius assumed; original pose outliers remain. Window uses current and past observations only.')
        report[name]=summary
        print(json.dumps({name:summary},ensure_ascii=True,indent=2),flush=True)
    path=HERE/'spin_estimation_summary.json'
    previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {};previous.update(report)
    path.write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
