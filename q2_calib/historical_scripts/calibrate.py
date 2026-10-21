"""Calibrate a camera from the latest captured checkerboard session."""
from pathlib import Path
import json
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGES = ROOT / "outputs" / "calibration_images"
OUT = ROOT / "outputs"
PATTERN = (8, 6)  # inner corners: columns, rows


def main():
    sessions = sorted((p for p in BASE_IMAGES.glob("session-*") if p.is_dir()), reverse=True)
    if not sessions:
        raise SystemExit(f"没有找到采集会话，请先运行 work/capture_calibration.py：{BASE_IMAGES}")
    image_dir = sessions[0]
    image_paths = sorted(image_dir.glob("*.png"))
    if len(image_paths) < 15:
        raise SystemExit(f"最新会话至少需要 15 张图像；当前找到 {len(image_paths)} 张：{image_dir}")

    # Use square units here. Intrinsics, distortion, and pixel reprojection error
    # do not require the physical square size; translation scale would.
    grid = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    object_points, image_points, used = [], [], []
    image_size = None
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"跳过无法读取的图像：{path.name}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"未识别到 8×6 内角点，跳过：{path.name}")
            continue
        refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
        )
        object_points.append(grid.copy())
        image_points.append(refined)
        used.append(path.name)

    if len(used) < 15:
        raise SystemExit(f"成功识别的图像只有 {len(used)} 张；需至少 15 张。请补拍或检查棋盘格尺寸。")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view = []
    for name, obj, img, rvec, tvec in zip(used, object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(img, projected, cv2.NORM_L2) / len(projected) ** 0.5
        per_view.append({"image": name, "reprojection_error_px": float(error)})
    overall = float(np.sqrt(np.mean([item["reprojection_error_px"] ** 2 for item in per_view])))

    report = {
        "source_session": str(image_dir),
        "image_count_used": len(used),
        "image_size_px": {"width": image_size[0], "height": image_size[1]},
        "checkerboard_inner_corners": {"columns": PATTERN[0], "rows": PATTERN[1]},
        "square_size_unit": "one arbitrary square unit (physical square size was not supplied)",
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients_opencv_order": dist_coeffs.reshape(-1).tolist(),
        "opencv_rms_reprojection_error_px": float(rms),
        "overall_reprojection_error_px": overall,
        "per_image_reprojection_errors": per_view,
        "used_images": used,
    }
    (OUT / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fs = cv2.FileStorage(str(OUT / "calibration.yml"), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("checkerboard_inner_corners", np.array(PATTERN, np.int32))
    fs.write("opencv_rms_reprojection_error_px", float(rms))
    fs.release()

    preview = cv2.imread(str(image_dir / used[0]))
    h, w = preview.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(preview, camera_matrix, dist_coeffs, None, new_matrix)
    cv2.imwrite(str(OUT / "undistorted_preview.png"), undistorted)

    print(f"成功使用 {len(used)} 张图像（分辨率 {w}×{h}）")
    print("内参矩阵 K：\n", camera_matrix)
    print("畸变系数 [k1, k2, p1, p2, k3, ...]：\n", dist_coeffs.reshape(-1))
    print(f"OpenCV RMS 重投影误差：{rms:.6f} px")
    print(f"按每图误差合并的 RMS：{overall:.6f} px")
    print(f"报告：{OUT / 'calibration_report.json'}")
    print("说明：square size 未输入，因此外参平移尺度未知；K、畸变系数和像素误差仍可计算。")


if __name__ == "__main__":
    main()
