"""Export short, downsampled GIF previews; complete MP4 files remain authoritative."""
from pathlib import Path
import cv2
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent

def main():
    for folder in ['qA_aim','qA_yolo']:
        for name in ['远距离平移','中距离陀螺','近距离陀螺','装甲板_蓝']:
            video=ROOT/folder/'figures'/f'detected_{name}.mp4'
            cap=cv2.VideoCapture(str(video))
            if not cap.isOpened():raise RuntimeError(str(video))
            fps=cap.get(cv2.CAP_PROP_FPS);stride=max(1,round(fps/10));images=[]
            for frame_id in range(min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),int(fps*6))):
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Unexpected decode failure')
                if frame_id%stride:continue
                frame=cv2.resize(frame,(480,360))
                images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).quantize(colors=96))
            cap.release()
            output=video.with_name(f'preview_{name}.gif')
            images[0].save(output,save_all=True,append_images=images[1:],duration=round(1000*stride/fps),loop=0,optimize=False)
            print(folder,name,output.stat().st_size,flush=True)

if __name__=='__main__':main()
