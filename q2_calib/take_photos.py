"""Open a camera and save original photos with SPACE or S."""
import argparse
from datetime import datetime
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / 'photos'
    folder.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera. Try --camera 0 or --camera 2.')
        cv2.namedWindow('USB Camera', cv2.WINDOW_NORMAL)
        count = 0
        print('Click preview. SPACE or S: save photo. Q or Esc: quit.')
        print('Photos:', folder)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed. Check the USB connection.')
            preview = frame.copy()
            cv2.putText(preview, f'SPACE/S: photo | Q: quit | Saved: {count}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('USB Camera', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty('USB Camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord(' '), ord('s')):
                path = folder / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.png')
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError('Could not save photo: ' + str(path))
                count += 1
                print('Saved:', path)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
