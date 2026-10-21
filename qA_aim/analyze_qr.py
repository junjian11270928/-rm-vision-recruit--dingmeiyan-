"""Controlled Q/R sensitivity experiment using a frozen set of PnP observations.
No detection, filter configuration, or main prediction output is changed.
"""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
import argparse
import csv
import hashlib
import json
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from run_analysis import predict,write_csv,FPS

SCALES=[.01,.1,1.,10.,100.]
BASE=dict(process_noise_density=[3.,.001,.001],measurement_std_m=[.01,.015,.06],soft_gate_sigma=4.,max_variance_multiplier=100.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh-observations',action='store_true')
    args=parser.parse_args()
    out=HERE/'qr_analysis';out.mkdir(exist_ok=True)
    snapshot=out/'observations.csv'
    if not snapshot.exists() or args.refresh_observations:
        rows=list(csv.DictReader((HERE/'filter_diagnostics.csv').open(encoding='utf-8-sig')))
        write_csv(snapshot,['frame','t','x','y','z'],[[r['frame'],r['t'],r['obs_x'],r['obs_y'],r['obs_z']] for r in rows])
    rows=list(csv.DictReader(snapshot.open(encoding='utf-8-sig')))
    if any(int(row['frame'])!=i or abs(float(row['t'])-i/FPS)>1e-8 for i,row in enumerate(rows)):
        raise ValueError('Observation snapshot must have consecutive 150 fps timestamps')
    z=np.array([[float(r[k]) if r[k] else np.nan for k in 'xyz'] for r in rows])
    t=np.arange(len(z))/FPS
    future=np.full_like(z,np.nan)
    future[:-38]=(z[37:-1]+z[38:])/2
    # Use exactly the same future-observation mask for every configuration.
    evaluation=np.isfinite(future).all(axis=1)&(t>=.15)
    cache={};records=[]
    for alpha in SCALES:
        for beta in SCALES:
            config=deepcopy(BASE)
            config['process_noise_density']=(np.array(BASE['process_noise_density'])*alpha).tolist()
            # beta scales covariance R. Config takes standard deviations, so sqrt(beta).
            config['measurement_std_m']=(np.array(BASE['measurement_std_m'])*np.sqrt(beta)).tolist()
            filtered,forecast,accepted,softened=predict(z,config,True)
            if not np.isfinite(forecast[evaluation]).all():
                raise RuntimeError('A configuration lacks predictions; common-mask comparison required')
            error=forecast-future
            distance=np.linalg.norm(error[evaluation],axis=1)
            # This is agreement with raw observations, NOT filtering accuracy.
            obs_mask=np.isfinite(z).all(axis=1)&(t>=.15)&np.isfinite(filtered).all(axis=1)
            displacement=filtered-z
            second=np.diff(filtered[:,0],n=2)
            roughmask=(t[2:]>=.15)&np.isfinite(second)
            # Offline diagnostic: how much later filtered x matches observed x.
            # Search nonnegative delay; observations are a noisy reference, not truth.
            lag_scores=[]
            for lag in range(31):
                source_ids=np.arange(len(z)-lag)
                target_ids=source_ids+lag
                good=(t[source_ids]>=.5)&np.isfinite(z[source_ids,0])&np.isfinite(filtered[target_ids,0])
                lag_scores.append(float(np.mean((filtered[target_ids[good],0]-z[source_ids[good],0])**2)))
            delay=int(np.argmin(lag_scores))/FPS
            late=evaluation&(t>=t[len(t)//2]+.15)
            record=dict(Q_multiplier=alpha,R_variance_multiplier=beta,samples=int(evaluation.sum()),
                prediction_rmse_3d_m=float(np.sqrt(np.mean(distance**2))),
                prediction_mean_3d_m=float(np.mean(distance)),prediction_p95_3d_m=float(np.percentile(distance,95)),
                prediction_max_3d_m=float(np.max(distance)),
                prediction_rmse_x_m=float(np.sqrt(np.mean(error[evaluation,0]**2))),
                filtered_observation_rms_3d_m=float(np.sqrt(np.mean(np.sum(displacement[obs_mask]**2,axis=1)))),
                filtered_x_second_difference_rms_m=float(np.sqrt(np.mean(second[roughmask]**2))),
                approximate_x_delay_s=delay,softened_frames=int(softened.sum()),
                second_half_prediction_rmse_3d_m=float(np.sqrt(np.mean(np.sum(error[late]**2,axis=1)))))
            records.append(record);cache[alpha,beta]=(filtered,forecast,error,record)
    columns=list(records[0]);write_csv(out/'experiment_metrics.csv',columns,[[r[k] for k in columns] for r in records])
    metadata=dict(base_config=BASE,Q_multipliers=SCALES,R_variance_multipliers=SCALES,
        dt_s=1/FPS,horizon_s=.25,startup_excluded_s=.15,samples=int(evaluation.sum()),
        input_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        future_observation='Mean of frames f+37 and f+38; both must be finite',
        evaluation='Comparison to future PnP observations, no independent ground truth',
        soft_gate='Current adaptive variance inflation is kept active and identical in all experiments',
        initial_state='Same zero-velocity initialization and fixed P0 for every run',
        roughness='RMS of second differences of filtered x; proxy includes true dynamics, not an accuracy metric',
        delay='Approximate 0..0.2s nonnegative shift minimizing filtered-x vs raw-observed-x MSE after 0.5s; not true sensor latency',
        parameter_search='Descriptive sensitivity on this recording; not an independently validated optimum')
    (out/'experiment_settings.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    colors=['#4363aa','#299d8f','#db9e22','#cd4b43','#8455a3']
    zoom=(t>=.6)&(t<=2.2)
    for which in ['Q','R']:
        fig,axes=plt.subplots(3,1,figsize=(12,10))
        axes[0].plot(t[zoom],z[zoom,0],'.',ms=3,color='.7',label='原始位置观测（不是独立真值）')
        axes[1].plot(t[zoom],future[zoom,0],color='black',lw=2,label='t+0.25秒的实际观测')
        axes[2].plot(t,z[:,2],'.',ms=2,color='.75',alpha=.5,label='原始深度观测')
        for scale,color in zip(SCALES,colors):
            filtered,forecast,_,record=cache[(scale,1.) if which=='Q' else (1.,scale)]
            label=f'{which} × {scale:g}'
            axes[0].plot(t[zoom],filtered[zoom,0],color=color,lw=1.3,label=label)
            axes[1].plot(t[zoom],forecast[zoom,0],color=color,lw=1.3,label=label)
            axes[2].plot(t,filtered[:,2],color=color,lw=1,label=label)
        axes[0].set(title='当前位置滤波：看平滑程度与跟随滞后',xlabel='当前时刻 t（秒）',ylabel='x（米）')
        axes[1].set(title='提前0.25秒预测：看转向附近的过冲',xlabel='发布预测的时刻 t（秒）',ylabel='未来位置 x（米）')
        axes[2].set(title='深度滤波：更平滑不代表更接近真实位置',xlabel='当前时刻 t（秒）',ylabel='z（米）')
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=3)
        fig.suptitle(f'固定同一批观测，只改变{which}'+('（R保持不变）' if which=='Q' else '（Q保持不变；倍率指方差）'))
        fig.tight_layout();fig.savefig(out/f'{which}_effect.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for which,color in [('Q','#2674b2'),('R','#d9792d')]:
        subset=[cache[(s,1.) if which=='Q' else (1.,s)][3] for s in SCALES]
        axes[0].plot(SCALES,[r['prediction_rmse_3d_m'] for r in subset],'o-',label=f'仅改变{which}',color=color)
        axes[1].plot(SCALES,[r['filtered_x_second_difference_rms_m']*1000 for r in subset],'o-',label=f'仅改变{which}',color=color)
    axes[0].set(xscale='log',xlabel='协方差倍率（对数轴）',ylabel='三维预测RMSE（米）',title='预测误差：并非参数越大或越小越好')
    axes[1].set(xscale='log',yscale='log',xlabel='协方差倍率（对数轴）',ylabel='滤波x二阶差分RMS（毫米）',title='粗糙程度代理：越小通常越平滑')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'QR_metrics.png',dpi=160);plt.close(fig)
    heat=np.array([[cache[q,r][3]['prediction_rmse_3d_m'] for r in SCALES] for q in SCALES])
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(heat,cmap='YlOrRd',aspect='auto')
    ax.set_xticks(range(5),[f'{v:g}' for v in SCALES]);ax.set_yticks(range(5),[f'{v:g}' for v in SCALES])
    ax.set(xlabel='R方差倍率（测量噪声）',ylabel='Q倍率（过程噪声）',title='Q/R共同作用：三维0.25秒预测RMSE（米）')
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center',color='black',fontsize=10)
    fig.colorbar(im,ax=ax,label='RMSE（米），越低越好');fig.tight_layout();fig.savefig(out/'QR_heatmap.png',dpi=160);plt.close(fig)
    # Export main comparison traces for inspection and reuse.
    for which in ['Q','R']:
        for s in SCALES:
            filtered,forecast,_,_=cache[(s,1.) if which=='Q' else (1.,s)]
            write_csv(out/f'{which}_x{s:g}_trace.csv',['t','obs_x','filtered_x','pred_x_t_plus_025','future_obs_x'],[[t[i],z[i,0],filtered[i,0],forecast[i,0],future[i,0]] for i in range(len(t))])
    print(json.dumps([r for r in records if r['Q_multiplier']==1 or r['R_variance_multiplier']==1],ensure_ascii=True,indent=2))


if __name__=='__main__':main()
