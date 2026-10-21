"""Blue armor light-core detector. No pose estimation or hidden-target extrapolation."""
import cv2
import numpy as np


def detect_blue(frame,blue_red=30,blue_min=180,green_min=100):
    b,g,r=cv2.split(frame)
    mask=((b.astype(np.int16)-r.astype(np.int16)>blue_red)&(b>blue_min)&(g>green_min)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lights=[]
    for contour in contours:
        if cv2.contourArea(contour)<3:continue
        box=cv2.boxPoints(cv2.minAreaRect(contour));edges=np.roll(box,-1,axis=0)-box
        sizes=np.linalg.norm(edges,axis=1);length=float(sizes.max());width=float(sizes.min())
        if width<.8 or length<5:continue
        if length/width<1.8:
            # Bloom can attach to one side of a vertical lamp. Keep the columns
            # with the longest vertical support, rather than using the full glow.
            x,y,w,h=cv2.boundingRect(contour)
            if h<1.2*w:continue
            local=np.zeros((h,w),np.uint8);cv2.drawContours(local,[contour-np.array([[[x,y]]])],-1,255,-1)
            support=np.count_nonzero(local,axis=0)
            good=support>=max(5,.85*support.max())
            local[:,~good]=0
            yy,xx=np.nonzero(local)
            if len(xx)<6:continue
            points=np.c_[xx+x,yy+y].astype(np.float32)
            box=cv2.boxPoints(cv2.minAreaRect(points));edges=np.roll(box,-1,axis=0)-box
            sizes=np.linalg.norm(edges,axis=1);length=float(sizes.max());width=float(sizes.min())
        if width<.8 or not 1.8<length/width<30:continue
        axis=edges[int(np.argmax(sizes))]/length
        if abs(axis[1])<.75:continue
        center=box.mean(axis=0)
        ends=np.array(sorted([center-axis*length/2,center+axis*length/2],key=lambda point:point[1]))
        lights.append((center,length,ends))
    lights.sort(key=lambda item:item[0][0]);candidates=[]
    for i,left in enumerate(lights):
        for j in range(i+1,len(lights)):
            right=lights[j];height=(left[1]+right[1])/2
            gap=(right[0][0]-left[0][0])/height;offset=abs(right[0][1]-left[0][1])/height
            ratio=min(left[1],right[1])/max(left[1],right[1])
            a=left[2][1]-left[2][0];b=right[2][1]-right[2][0]
            parallel=abs(float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)))
            if .75<gap<4 and offset<.6 and ratio>.55 and parallel>.94:
                # Avoid spanning over another lamp at the same height.
                blocked=any(left[0][0]<middle[0][0]<right[0][0] and abs(middle[0][1]-(left[0][1]+right[0][1])/2)<height*.6 for middle in lights)
                if blocked:continue
                score=abs(gap-135/55)+offset+(1-ratio)
                pts=np.array([left[2][0],left[2][1],right[2][1],right[2][0]])
                candidates.append((score,i,j,pts))
    used=set();armors=[]
    for _,i,j,points in sorted(candidates,key=lambda item:item[0]):
        if i not in used and j not in used:
            used.update((i,j));armors.append(points)
    return mask,lights,armors
