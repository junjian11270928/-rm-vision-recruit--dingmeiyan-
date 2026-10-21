"""Capture 20 checkerboard views from a USB camera for camera calibration."""
from datetime import datetime
from pathlib import Path
import argparse
import cv2


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = ROOT / "outputs" / "calibration_images"
PATTERN = (8, 6)  # inner corners: columns, rows
TARGET = 20
MINIMUM = 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1, help="USB camera index (default: 1)")
    args = parser.parse_args()

    session = datetime.now().strftime("session-%Y%m%d-%H%M%S")
    out = BASE_OUT / session
    out.mkdir(parents=True, exist_ok=False)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"无法打开相机索引 {args.camera}。请检查相机占用，或用 --camera 0/2 重试。")

    count = 0
    print("将棋盘格完整放入画面。按空格保存一张（需检测到全部 8×6 内角点）；按 q 在至少 15 张后结束。")
    print("请让棋盘出现在画面不同位置，并改变距离、倾斜方向和旋转角度；避免只在正中间拍正面。")
    window = "USB camera calibration | SPACE capture | q finish"
    while count < TARGET:
        ok, frame = cap.read()
        if not ok:
            print("相机读取失败。")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, PATTERN, corners, found)
        cv2.putText(view, f"Captured {count}/{TARGET} (minimum {MINIMUM})", (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(view, "SPACE: capture   q: finish after 15", (18, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if count >= MINIMUM:
                break
            print(f"目前只有 {count} 张，请至少采集 {MINIMUM} 张。")
        elif key == 32:
            if not found:
                print("未识别到完整棋盘格，请调整位置、距离或光线后再拍。")
                continue
            path = out / f"view_{count + 1:02d}.png"
            cv2.imwrite(str(path), frame)
            count += 1
            print(f"已保存 {count}/{TARGET}: {path}")
            cv2.waitKey(180)

    cap.release()
    cv2.destroyAllWindows()
    print(f"采集结束：{count} 张，目录：{out}")
    if count < MINIMUM:
        print(f"还需要至少 {MINIMUM - count} 张才可标定。")
    else:
        print("现在可以运行 work/calibrate.py 计算结果。")


if __name__ == "__main__":
    main()
