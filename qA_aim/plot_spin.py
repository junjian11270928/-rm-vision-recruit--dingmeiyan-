"""Plot every plate observation without inventing persistent plate identities."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plot_cache'))
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    path=HERE/'detections_中距离陀螺.csv'
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    valid=[r for r in rows if all(r[k] for k in ['x','y','z','yaw'])]
    if not valid:
        raise ValueError('No valid poses in detections CSV')
    data=np.array([[float(r[k]) for k in ['frame','x','y','z','yaw']] for r in valid])
    t=data[:,0]/150.
    out=HERE/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for j,ax in enumerate(axes):
        ax.scatter(t,data[:,j+1],s=3,alpha=.6,color='#177ab4')
        ax.set_ylabel(['x (m)','y (m)','z (m)','yaw (rad)'][j])
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('Time (s), frame / 150')
    fig.suptitle('Middle-distance spin: all observed plates\nDots are separate detections; plate identities are not tracked')
    fig.tight_layout();fig.savefig(out/'spin_pose_time.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7))
    dots=ax.scatter(data[:,1],data[:,3],c=t,s=5,cmap='viridis',alpha=.7)
    ax.set(xlabel='x (m)',ylabel='z (m)',title='Spin: observed plate positions, not body center')
    ax.grid(alpha=.25);ax.set_aspect('equal',adjustable='datalim')
    fig.colorbar(dots,ax=ax,label='Time (s)');fig.tight_layout()
    fig.savefig(out/'spin_trajectory_xz.png',dpi=160);plt.close(fig)
    frames=len(set(int(r['frame']) for r in rows))
    summary=dict(candidate_rows=len(rows),valid_pose_rows=len(valid),failed_pose_rows=len(rows)-len(valid),frames_with_candidates=frames,depth_median_m=float(np.median(data[:,3])),depth_min_m=float(np.min(data[:,3])),depth_max_m=float(np.max(data[:,3])),note='Each dot is one plate detection. No plate ID tracking; do not interpret inter-dot changes as single-plate jumps. Yaw is a planar PnP estimate, not rotation speed.')
    (HERE/'spin_pose_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
