"""A4 pose/distance measurement with the calibrated USB camera."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

K = np.array([[679.84939735, 0, 313.56512996],
              [0, 679.9170018, 239.37191511], [0, 0, 1]], np.float64)
D = np.array([0.152834055, 0.185954830, 0.00324167352,
              -0.00191224742, -3.70450216], np.float64)
# Origin is the center of the paper; units are millimeters.
OBJECT = np.array([[-105, -148.5, 0], [105, -148.5, 0],
                   [105, 148.5, 0], [-105, 148.5, 0]], np.float64)


def save(path, frame):
    ok, data = cv2.imencode('.png', frame)
    if not ok:
        raise RuntimeError('Image encoding failed')
    data.tofile(str(path))


def plot_errors(rows, output, angle_test):
    if len(rows) < 2:
        return
    xkey = 'nominal_angle_deg' if angle_test else 'ruler_mm'
    xs = np.array([row[xkey] for row in rows], float)
    ys = np.array([row['absolute_error_mm'] for row in rows], float)
    xmax = max(float(xs.max()) * 1.1, 1)
    ymax = max(float(ys.max()) * 1.15, 1)
    chart = np.full((600, 900, 3), 255, np.uint8)
    cv2.putText(chart, 'Measured absolute distance error', (80, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for i in range(6):
        y = 500 - i * 80
        cv2.line(chart, (100, y), (830, y), (220, 220, 220), 1)
        cv2.putText(chart, f'{ymax*i/5:.1f}', (15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        x = 100 + i * 146
        cv2.putText(chart, f'{xmax*i/5:.0f}', (x-15, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.line(chart, (100, 100), (100, 500), (0,0,0), 2)
    cv2.line(chart, (100, 500), (830, 500), (0,0,0), 2)
    cv2.putText(chart, 'Error (mm)', (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    xlabel = 'Nominal angle (degrees)' if angle_test else 'Ruler distance (mm)'
    cv2.putText(chart, xlabel, (310, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1)
    # Show every trial and connect mean errors at each measured distance/angle.
    means = []
    for value in sorted(set(xs)):
        mean = float(ys[xs == value].mean())
        means.append([round(100+730*value/xmax), round(500-400*mean/ymax)])
    cv2.polylines(chart, [np.array(means, np.int32)], False, (200, 80, 0), 2)
    for x, y in zip(xs, ys):
        cv2.circle(chart, (round(100+730*x/xmax), round(500-400*y/ymax)), 5, (0,0,220), -1)
    save(output, chart)


def capture(index):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cv2.namedWindow('A4 capture', cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            if frame.shape[:2] != (480, 640):
                raise RuntimeError('Calibration requires 640x480 camera frames')
            preview = frame.copy()
            cv2.putText(preview, 'SPACE: capture | Q: quit', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow('A4 capture', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')) or cv2.getWindowProperty('A4 capture', cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord(' '):
                return frame
    finally:
        cap.release()
        cv2.destroyAllWindows()


def select_corners(frame):
    points = []
    # Exact 2x display mapping makes small distant targets easier to click.
    name = 'Click TL > TR > BR > BL | ENTER confirm | R reset | Q quit'
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / 2.0, y / 2.0))
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, click)
    try:
        while True:
            view = cv2.resize(frame, None, fx=2, fy=2)
            for i, point in enumerate(points):
                xy = tuple(np.round(np.array(point) * 2).astype(int))
                cv2.circle(view, xy, 5, (0, 0, 255), -1)
                cv2.putText(view, str(i+1), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(name, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q')) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord('r'):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                pts = np.array(points, np.float64)
                if not cv2.isContourConvex(pts.astype(np.float32).reshape(-1, 1, 2)) or abs(cv2.contourArea(pts.astype(np.float32))) < 50:
                    print('Invalid corner order or tiny target. Press R and select again.')
                    continue
                return pts
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--groups', type=int, default=5)
    parser.add_argument('--angle-test', action='store_true', help='Nine trials: 0/30/60 degrees, three each')
    args = parser.parse_args()
    if args.angle_test:
        args.groups = 9
    if args.groups < 5:
        parser.error('At least 5 groups required')
    out = Path(__file__).resolve().parent / 'a4_results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    rows = []
    fields = ['group', 'nominal_angle_deg', 'estimated_tilt_deg', 'ruler_mm', 'pnp_center_distance_mm', 'depth_z_mm',
              'signed_error_mm', 'absolute_error_mm', 'relative_error_percent', 'reprojection_rmse_px']
    print('Use flat A4 in portrait: width 210 mm, height 297 mm.')
    print('Measure ruler distance from camera optical center (approx. lens center) to paper CENTER.')
    print('Suggested distances: 500, 1000, 1500, 2000, 3000 mm. Keep focus/zoom unchanged.')
    print('Result folder:', out)
    while len(rows) < args.groups:
        angle = [0, 30, 60][len(rows)//3] if args.angle_test else 0
        print(f'Set paper tilt to {angle} degrees. Keep center fixed for angle test.')
        try:
            text = input(f'Group {len(rows)+1}: ruler distance in mm (Q to stop): ').strip()
            if text.lower() == 'q':
                break
            actual = float(text)
            if not np.isfinite(actual) or actual <= 0:
                raise ValueError
        except ValueError:
            print('Enter a positive number, e.g. 500.')
            continue
        frame = capture(args.camera)
        if frame is None:
            break
        points = select_corners(frame)
        if points is None:
            break
        ok, rvec, tvec = cv2.solvePnP(OBJECT, points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(tvec).all():
            print('Pose failed. Repeat this group.')
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        if np.any((OBJECT @ rotation.T + tvec.reshape(1, 3))[:, 2] <= 0):
            print('Invalid pose behind camera. Repeat this group.')
            continue
        projected, _ = cv2.projectPoints(OBJECT, rvec, tvec, K, D)
        error_px = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-points)**2, axis=1))))
        distance = float(np.linalg.norm(tvec))
        error = distance - actual
        number = len(rows)+1
        tilt = float(np.degrees(np.arccos(np.clip(abs(rotation[2, 2]), 0, 1))))
        rows.append(dict(zip(fields, [number, angle, tilt, actual, distance, float(tvec[2, 0]),
                                     error, abs(error), abs(error)/actual*100, error_px])))
        save(out / f'{number:02d}_original.png', frame)
        marked = frame.copy()
        cv2.polylines(marked, [np.round(points).astype(np.int32)], True, (0, 255, 0), 1)
        cv2.putText(marked, f'PnP {distance:.1f} mm | Ruler {actual:.1f} mm',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        save(out / f'{number:02d}_measured.png', marked)
        np.savez(out / f'{number:02d}_pose.npz', points=points, rvec=rvec, tvec=tvec, K=K, D=D)
        with (out / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        plot_errors(rows, out / 'error_curve.png', args.angle_test)
        print(f'PnP center distance: {distance:.2f} mm; ruler: {actual:.2f} mm; '
              f'error: {error:+.2f} mm; relative error: {abs(error)/actual*100:.2f}%; '
              f'reprojection RMSE: {error_px:.3f} px')
    print(f'Saved {len(rows)} groups at {out}')
    if len(rows) < args.groups:
        print('Incomplete measurement set.')


if __name__ == '__main__':
    main()
