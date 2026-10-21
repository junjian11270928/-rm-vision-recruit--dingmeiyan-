"""第一步：红色灯条与装甲板候选检测。按空格暂停，按 Q 退出。"""
from pathlib import Path
import argparse
import cv2
import numpy as np


def detect(frame, red_blue=45, red_green=25, brightness=110):
    b, g, r = cv2.split(frame)
    mask = ((r.astype(np.int16) - b.astype(np.int16) > red_blue) &
            (r.astype(np.int16) - g.astype(np.int16) > red_green) & (r > brightness)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lights = []
    for contour in contours:
        if len(contour) < 5 or cv2.contourArea(contour) < 6:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        edges = np.roll(box, -1, axis=0) - box
        sizes = np.linalg.norm(edges, axis=1)
        length, width = float(max(sizes)), float(min(sizes))
        if length < 6 or width < 1 or not 1.8 < length / width < 25:
            continue
        axis = edges[int(np.argmax(sizes))] / length
        if abs(axis[1]) < 0.75:
            continue
        center = box.mean(axis=0)
        ends = sorted([center - axis * length / 2, center + axis * length / 2], key=lambda p: p[1])
        lights.append((center, length, np.array(ends)))
    lights.sort(key=lambda item: item[0][0])
    candidates = []
    for i, left in enumerate(lights):
        for j in range(i + 1, len(lights)):
            right = lights[j]
            height = (left[1] + right[1]) / 2
            gap = (right[0][0] - left[0][0]) / height
            offset = abs(right[0][1] - left[0][1]) / height
            ratio = min(left[1], right[1]) / max(left[1], right[1])
            a = left[2][1] - left[2][0]
            b = right[2][1] - right[2][0]
            parallel = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
            if 0.8 < gap < 4 and offset < 0.6 and ratio > 0.55 and parallel > 0.94:
                score = abs(gap - 135 / 55) + offset + (1 - ratio)
                points = np.array([left[2][0], left[2][1], right[2][1], right[2][0]])
                candidates.append((score, i, j, points))
    used, armors = set(), []
    for _, i, j, points in sorted(candidates, key=lambda item: item[0]):
        if i not in used and j not in used:
            used.update((i, j))
            armors.append(points)
    return mask, lights, armors


def annotate(frame, lights, armors):
    output = frame.copy()
    for _, _, ends in lights:
        cv2.line(output, tuple(ends[0].astype(int)), tuple(ends[1].astype(int)), (0, 255, 255), 2)
    for points in armors:
        cv2.polylines(output, [points.astype(np.int32)], True, (0, 255, 0), 2)
        for label, point in zip(('LT', 'LB', 'RB', 'RT'), points):
            xy = tuple(point.astype(int))
            cv2.circle(output, xy, 3, (255, 255, 0), -1)
            cv2.putText(output, label, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default = next(Path(__file__).resolve().parents[1].rglob('远距离平移.mp4'))
    parser.add_argument('--video', default=str(default))
    parser.add_argument('--preview', action='store_true', help='save first frame without opening a window')
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, lights, armors = detect(frame)
        output = annotate(frame, lights, armors)
        if args.preview:
            cv2.imwrite(str(Path(__file__).with_name('first_frame.jpg')), output)
            print(f'First frame: {len(lights)} lights, {len(armors)} armor candidates')
            break
        cv2.imshow('Armor candidates', output)
        cv2.imshow('Red mask', mask)
        key = cv2.waitKey(20) & 0xff
        if key == 32:
            key = cv2.waitKey(0) & 0xff
        if key in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
