"""Paired IPPE results with real D vs zero D on exactly the same detected points."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from run_analysis import pose,source

def main():
    camera=json.loads(source('camera_info.json').read_text(encoding='utf-8-sig'))
    K=np.array(camera['camera_matrix']);D=np.array(camera['distortion_coefficients']);report={}
    keys=['lt_x','lt_y','lb_x','lb_y','rb_x','rb_y','rt_x','rt_y']
    for name in ['远距离平移','中距离陀螺','近距离陀螺']:
        measurements=[];counts=dict(with_D_valid=0,zero_D_valid=0,total_candidates=0)
        with (HERE/f'detections_{name}.csv').open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                points=np.array([float(row[key]) for key in keys]).reshape(4,2)
                a,b=pose(points,K,D),pose(points,K,np.zeros_like(D))
                counts['total_candidates']+=1;counts['with_D_valid']+=a is not None;counts['zero_D_valid']+=b is not None
                if a is None or b is None:continue
                measurements.append([np.linalg.norm(points.mean(axis=0)-K[:2,2]),np.linalg.norm(a[1]-b[1]),
                    abs(a[1][2]-b[1][2]),abs(np.arctan2(np.sin(a[2]-b[2]),np.cos(a[2]-b[2])))])
        values=np.array(measurements);q=np.quantile(values[:,0],[.25,.75])
        report[name]=dict(common_valid=len(values),**counts,
            position_difference_m_median=float(np.median(values[:,1])),
            position_difference_m_p95=float(np.quantile(values[:,1],.95)),
            z_difference_m_median=float(np.median(values[:,2])),
            yaw_difference_deg_median=float(np.median(values[:,3])*180/np.pi),
            radius_range_px=[float(values[:,0].min()),float(values[:,0].max())],radius_quartiles_px=q.tolist(),
            inner_quartile_position_difference_m_median=float(np.median(values[values[:,0]<=q[0],1])),
            outer_quartile_position_difference_m_median=float(np.median(values[values[:,0]>=q[1],1])))
    (HERE/'distortion_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Microsoft YaHei'
    plt.rcParams['axes.unicode_minus']=False
    names=list(report);x=np.arange(3);fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle('Distortion enabled vs zero distortion: paired PnP differences',fontsize=16)
    values=[report[n]['position_difference_m_median']*1000 for n in names]
    bars=axes[0].bar(x,values,color=['#3b82f6','#14b8a6','#f59e0b'])
    axes[0].bar_label(bars,fmt='%.2f',padding=4)
    axes[0].set(xticks=x,xticklabels=names,ylabel='Median position difference (mm)',title='All common valid candidates',ylim=(0,9))
    inner=[report[n]['inner_quartile_position_difference_m_median']*1000 for n in names]
    outer=[report[n]['outer_quartile_position_difference_m_median']*1000 for n in names]
    for offset,values,label in [(-.18,inner,'Inner radius quartile'),(.18,outer,'Outer radius quartile')]:
        bars=axes[1].bar(x+offset,values,width=.35,label=label);axes[1].bar_label(bars,fmt='%.2f',padding=4)
    axes[1].set(xticks=x,xticklabels=names,ylabel='Median position difference (mm)',title='Within-video observed coverage',ylim=(0,14));axes[1].legend()
    fig.text(.5,.025,'Differences are not ground-truth errors. Outer observed quartile is not the image boundary.',ha='center')
    fig.tight_layout(rect=[0,.06,1,.94]);(HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures'/'distortion_comparison_reproduced.png',dpi=150);plt.close(fig)
    print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
