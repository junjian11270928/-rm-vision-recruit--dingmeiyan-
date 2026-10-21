from pathlib import Path
import os,sys,json,csv,hashlib
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.plot_cache'))
import numpy as np
p=Path(__file__).resolve().parent.parent/'qA_aim';sys.path.insert(0,str(p))
from estimate_spin import estimate,load_data,CONFIG,wrapped

def fixture(frames=900,omega=-17.,acceleration=.3,noise=True,moving=False):
 rng=np.random.default_rng(1729);groups={};truth=[]
 for frame in range(frames):
  t=frame/150;center=np.array([-.05+(.03*t if moving else 0),-.12,3.3]);phase=.17+omega*t+.5*acceleration*t*t
  observations=[]
  for k in range(4):
   yaw=float(wrapped(phase+k*np.pi/2,2*np.pi))
   if abs(yaw)>1.:continue
   position=center-.28*np.array([np.sin(yaw),0,np.cos(yaw)])
   if noise:position+=rng.normal(0,[.003,.003,.015]);yaw+=rng.normal(0,.025)
   observations.append([*position,yaw,.3])
  groups[frame]=observations;truth.append([*center,omega+acceleration*t])
 return groups,np.array(truth)

checks={}
groups,truth=fixture(noise=False,acceleration=0)
a,d,_,_=estimate(groups,len(truth))
valid=(np.arange(len(a))/150>=1.)&np.isfinite(a[:,1:4]).all(axis=1)
center_error=np.linalg.norm(a[valid,1:4]-truth[valid,:3],axis=1);omega_error=abs(a[valid,4]-truth[valid,3])
assert np.max(center_error)<.002 and np.max(omega_error)<.02
checks['exact_rotation']=dict(max_center_error_m=float(center_error.max()),max_omega_error_rad_s=float(omega_error.max()))
groups,truth=fixture();a,d,_,_=estimate(groups,len(truth));valid=(np.arange(len(a))/150>=1.)&np.isfinite(a[:,1:4]).all(axis=1)
center_error=np.linalg.norm(a[valid,1:4]-truth[valid,:3],axis=1);omega_error=abs(a[valid,4]-truth[valid,3])
assert np.median(center_error)<.03 and np.median(omega_error)<.5
checks['noisy_accelerating_rotation']=dict(median_center_error_m=float(np.median(center_error)),median_omega_error_rad_s=float(np.median(omega_error)))
for i in range(350,400):groups[i]=[]
gap,_,_,_=estimate(groups,len(truth));assert np.isnan(gap[381:400,1:]).all();assert np.isfinite(gap[430:,1:]).all()
checks['missing_observations']='PASS: output blank after 0.2s, recovers on valid observations'
groups,truth=fixture(noise=False,acceleration=0,moving=True);moving,_,_,_=estimate(groups,len(truth));ids=np.flatnonzero(np.isfinite(moving[:,1]))
assert moving[ids[-1],1]-moving[ids[0],1]>.12
checks['not_forced_fixed_center']='PASS: synthetic moving center produces changing center estimate'
for name in ['中距离陀螺','近距离陀螺']:
 groups,total,raw=load_data(name);full,diag,_,_=estimate(groups,total)
 cutoff=700;prefix,_,_,_=estimate({k:v for k,v in groups.items() if k<cutoff},cutoff)
 assert np.allclose(prefix,full[:cutoff],equal_nan=True),'Future data leakage'
 rows=list(csv.DictReader((p/f'pred_{name}.csv').open(encoding='utf-8-sig')))
 assert len(rows)==total and list(rows[0])==['t','cx','cy','cz','omega']
 for i,row in enumerate(rows):
  assert abs(float(row['t'])-i/150)<1e-8
  if i<90:assert all(not row[k] for k in ['cx','cy','cz','omega'])
  else:assert all(row[k] for k in ['cx','cy','cz','omega'])
 checks[name]=dict(prefix_causality='PASS',row_count=total,initialization_blank_frames=90,csv_format='PASS')
(p/'spin_estimation_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=True,indent=2))
