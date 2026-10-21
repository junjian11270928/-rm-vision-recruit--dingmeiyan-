"""USB chessboard calibration. Python 3 + opencv-python + numpy."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class PreviewCorrection:
    """Smoothed gray-world white balance, followed by gamma brightening."""
    def __init__(self):
        self.gains = None

    def apply(self, frame, brightness=150, white_balance=True):
        small = cv2.resize(frame, (160, 120)).astype(np.float32)
        # Exclude nearly black pixels and saturated highlights from estimation.
        valid = (small.min(axis=2) > 15) & (small.max(axis=2) < 245)
        if white_balance and np.count_nonzero(valid) > 200:
            means = small[valid].mean(axis=0)
            target = np.clip(means.mean() / np.maximum(means, 1), 0.5, 2.0)
            self.gains = target if self.gains is None else 0.9 * self.gains + 0.1 * target
        gains = self.gains if white_balance and self.gains is not None else np.ones(3)
        balanced = np.clip(frame.astype(np.float32) * gains, 0, 255).astype(np.uint8)
        # Slider 100 is neutral; higher values lift dark tones.
        exponent = 100.0 / max(brightness, 100)
        lut = np.round(255 * (np.arange(256) / 255.0) ** exponent).astype(np.uint8)
        return cv2.LUT(balanced, lut)


def detect(gray, pattern):
    found, corners = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    # OpenCV versions may return (N, 2) or (N, 1, 2). Normalize once.
    if found:
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return found, corners


def corner_distance(first, second):
    first = np.asarray(first).reshape(-1, 2)
    second = np.asarray(second).reshape(-1, 2)
    return min(float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1)))),
               float(np.sqrt(np.mean(np.sum((first - second[::-1]) ** 2, axis=1)))))


def calibrate(points, size, args, folder):
    obj = np.zeros((args.cols * args.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    obj *= args.square_mm
    objects = [obj.copy() for _ in points]
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objects, points, size, None, None
    )
    errors = []
    total_sq = 0.0
    total_n = 0
    for i, (measured, rv, tv) in enumerate(zip(points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(obj, rv, tv, matrix, dist)
        squared = np.sum((measured.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1)
        total_sq += float(squared.sum())
        total_n += len(squared)
        errors.append({'image': f'{i+1:02d}.png',
                       'rmse_px': float(np.sqrt(squared.mean()))})
    global_rmse = float(np.sqrt(total_sq / total_n))
    result = {
        'image_size_wh': list(size), 'image_count': len(points),
        'inner_corners_cols_rows': [args.cols, args.rows],
        'square_size_mm': args.square_mm,
        'camera_matrix': matrix.tolist(),
        'distortion_order': ['k1', 'k2', 'p1', 'p2', 'k3'],
        'distortion_coefficients': dist.ravel().tolist(),
        'opencv_rms_px': float(rms), 'global_reprojection_rmse_px': global_rmse,
        'per_image_errors': errors,
        'rvecs': [v.ravel().tolist() for v in rvecs],
        'tvecs_mm': [v.ravel().tolist() for v in tvecs],
    }
    (folder / 'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez(folder / 'calibration.npz', camera_matrix=matrix, dist_coeffs=dist,
             image_size=np.array(size), rvecs=np.array(rvecs), tvecs=np.array(tvecs))
    with (folder / 'per_image_errors.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'rmse_px'])
        writer.writeheader()
        writer.writerows(errors)
    report = (f'Images: {len(points)}\nImage size: {size}\n'
              f'Camera matrix K (pixels):\n{matrix}\n'
              f'Distortion [k1 k2 p1 p2 k3]:\n{dist.ravel()}\n'
              f'OpenCV RMS: {rms:.6f} px\n'
              f'Global reprojection RMSE: {global_rmse:.6f} px\n\n'
              + '\n'.join(f"{e['image']}: {e['rmse_px']:.6f} px" for e in errors))
    (folder / 'result.txt').write_text(report, encoding='utf-8')
    print('\n' + report + '\n\nSaved to: ' + str(folder))
    raw = cv2.imread(str(folder / 'images' / '01.png'))
    corrected = cv2.undistort(raw, matrix, dist)
    cv2.imwrite(str(folder / 'undistorted_01.png'), corrected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=['msmf', 'dshow', 'auto'], default='msmf')
    parser.add_argument('--cols', type=int, default=8)
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--square-mm', type=float, required=True,
                        help='Measured physical square side in millimeters')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'calibration_runs')
    args = parser.parse_args()
    if not 15 <= args.count <= 25:
        parser.error('--count must be between 15 and 25')
    if args.cols < 3 or args.rows < 3 or args.square_mm <= 0:
        parser.error('Invalid board dimensions or square size')
    folder = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (folder / 'images').mkdir(parents=True)
    (folder / 'corners').mkdir()
    backend = {'msmf': cv2.CAP_MSMF, 'dshow': cv2.CAP_DSHOW, 'auto': cv2.CAP_ANY}[args.backend]
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera. Close other camera apps or try --camera 1 / 2.')
    # Driver support varies. Request auto exposure instead of forcing a fixed exposure.
    actual_backend = cap.getBackendName()
    exposure_mode = 0.75 if actual_backend == 'DSHOW' else 1.0
    accepted = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, exposure_mode)
    print(f'Camera {args.camera}, backend {actual_backend}, auto exposure request: {accepted}')
    print('If the preview stays black, try another --camera index or --backend dshow.')
    points = []
    size = None
    correction = PreviewCorrection()
    cv2.namedWindow('USB calibration', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Brightness 100=off', 'USB calibration', 150, 300, lambda _: None)
    cv2.createTrackbar('Auto white balance', 'USB calibration', 1, 1, lambda _: None)
    message = 'Move board; SPACE: save | C: calibrate (15+) | Q: quit'
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera frame read failed. Saved images remain on disk.')
            current_size = (frame.shape[1], frame.shape[0])
            if size is None:
                size = current_size
            if size != current_size:
                raise RuntimeError('Camera resolution changed; restart with a fixed resolution.')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = detect(gray, (args.cols, args.rows))
            display = correction.apply(
                frame,
                cv2.getTrackbarPos('Brightness 100=off', 'USB calibration'),
                bool(cv2.getTrackbarPos('Auto white balance', 'USB calibration')),
            )
            if found:
                cv2.drawChessboardCorners(display, (args.cols, args.rows), corners, found)
            cv2.putText(display, f'{len(points)}/{args.count}  Board: {"OK" if found else "NOT FOUND"}',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.putText(display, message, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 255), 1)
            cv2.imshow('USB calibration', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('USB calibration', cv2.WND_PROP_VISIBLE) < 1:
                print('Stopped. Images saved at:', folder)
                return
            if key == ord(' '):
                if not found:
                    message = 'Board not found: show the entire board clearly.'
                    continue
                # Reject near-identical image corner layouts, including reversed indexing.
                duplicate = any(corner_distance(corners, p) < 0.025 * min(size) for p in points)
                if duplicate:
                    message = 'Too similar: change position, distance or tilt.'
                    continue
                name = f'{len(points)+1:02d}.png'
                if not cv2.imwrite(str(folder / 'images' / name), frame):
                    raise RuntimeError('Failed to save image')
                cv2.imwrite(str(folder / 'corners' / name), display)
                points.append(corners.copy())
                message = 'Saved. Move board, hold still, then SPACE.'
                print(f'Saved {len(points)}/{args.count}: {name}')
            if len(points) == args.count or key == ord('c'):
                if len(points) < 15:
                    message = 'Need at least 15 valid images.'
                    continue
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    calibrate(points, size, args, folder)


if __name__ == '__main__':
    main()
