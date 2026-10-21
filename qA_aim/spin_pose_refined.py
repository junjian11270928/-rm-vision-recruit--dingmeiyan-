"""Experimental spin pose refinement; uses existing candidates and unchanged color thresholds."""
import cv2
import numpy as np
from run_analysis import OBJECT


def refine_points(frame, points, thresholds, method='row_center'):
    # Work locally around the original two light bars; pairing is unchanged.
    updated=[]
    for indices in ((0,1),(3,2)):
        old=points[list(indices)].astype(np.float64)
        x0=max(0,int(np.floor(old[:,0].min()))-15); x1=min(frame.shape[1],int(np.ceil(old[:,0].max()))+16)
        y0=max(0,int(np.floor(old[:,1].min()))-12); y1=min(frame.shape[0],int(np.ceil(old[:,1].max()))+13)
        roi=frame[y0:y1,x0:x1]
        b,g,r=cv2.split(roi)
        mask=((r.astype(np.int16)-b.astype(np.int16)>thresholds['red_blue']) & (r.astype(np.int16)-g.astype(np.int16)>thresholds['red_green']) & (r>thresholds['brightness'])).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        target=old.mean(axis=0)-[x0,y0]
        contours=[c for c in contours if cv2.contourArea(c)>=6]
        if not contours:
            updated.append(old);continue
        contour=min(contours,key=lambda c:np.linalg.norm(np.array(cv2.minAreaRect(c)[0])-target))
        if np.linalg.norm(np.array(cv2.minAreaRect(contour)[0])-target)>5:
            updated.append(old);continue
        filled=np.zeros_like(mask);cv2.drawContours(filled,[contour],-1,255,-1)
        yy,xx=np.nonzero(filled)
        if method=='pca':
            coords=np.c_[xx,yy].astype(np.float64)
            center=coords.mean(axis=0)
            _,vectors=np.linalg.eigh(np.cov(coords.T));axis=vectors[:,-1]
            projected=(coords-center)@axis
            ends=center+np.outer([projected.min(),projected.max()],axis)
        elif method=='row_center':
            # Row midpoints avoid allowing the width of a tilted bounding box
            # to determine the longitudinal endpoints.
            rows=np.unique(yy)
            centers=np.array([[(xx[yy==y].min()+xx[yy==y].max())/2,y] for y in rows],np.float32)
            vx,vy,cx,cy=cv2.fitLine(centers,cv2.DIST_HUBER,0,.01,.01).reshape(-1)
            if abs(vy)<.75:
                updated.append(old);continue
            ends_y=np.array([rows.min(),rows.max()],float)
            ends=np.c_[cx+(ends_y-cy)*vx/vy,ends_y]
        else:
            raise ValueError('Unknown endpoint method')
        ends=ends+np.array([x0,y0]);ends=ends[np.argsort(ends[:,1])]
        # Preserve the original estimate if component assignment looks wrong.
        if np.linalg.norm(ends-old,axis=1).max()>8:
            updated.append(old)
        else:
            updated.append(ends)
    left,right=updated
    return np.array([left[0],left[1],right[1],right[0]],np.float64)


def solve_refined(points,K,D):
    result=cv2.solvePnPGeneric(OBJECT,np.asarray(points,np.float64),K,D,flags=cv2.SOLVEPNP_IPPE)
    candidates=[]
    for initial_r,initial_t in zip(result[1],result[2]):
        rvec,tvec=cv2.solvePnPRefineLM(OBJECT,np.asarray(points,np.float64),K,D,initial_r.copy(),initial_t.copy())
        R,_=cv2.Rodrigues(rvec)
        camera_points=OBJECT@R.T+tvec.reshape(3)
        if not np.isfinite(camera_points).all() or np.min(camera_points[:,2])<=0:continue
        projection,_=cv2.projectPoints(OBJECT,rvec,tvec,K,D)
        error=float(np.sqrt(np.mean(np.sum((projection.reshape(4,2)-points)**2,axis=1))))
        if error>3:continue
        candidates.append((error,tvec.reshape(3),float(np.arctan2(R[0,2],R[2,2]))))
    return min(candidates,key=lambda item:item[0]) if candidates else None
