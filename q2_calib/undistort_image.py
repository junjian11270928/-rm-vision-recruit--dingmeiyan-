"""Undistort a 640x480 image using the measured USB camera calibration."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def save_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('Image encoding failed')
    encoded.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=Path(__file__).resolve().parent/'calibration_runs'/'20261016_192448_766416'/'images'/'09.png')
    args = parser.parse_args()
    original = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError('Cannot read image: ' + str(args.image))
    if original.shape[:2] != (480, 640):
        raise ValueError('This calibration requires an original 640x480 image.')
    matrix = np.array([[679.84939735, 0, 313.56512996],
                       [0, 679.9170018, 239.37191511],
                       [0, 0, 1]], dtype=np.float64)
    distortion = np.array([0.152834055, 0.185954830, 0.00324167352,
                           -0.00191224742, -3.70450216], dtype=np.float64)
    corrected = cv2.undistort(original, matrix, distortion, None, matrix)
    comparison = np.full((536, 1292, 3), 245, dtype=np.uint8)
    comparison[48:528, :640] = original
    comparison[48:528, 652:] = corrected
    for title, x in [('Before: original', 15), ('After: undistorted', 667)]:
        cv2.putText(comparison, title, (x, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (30, 30, 30), 2, cv2.LINE_AA)
    output = Path(__file__).resolve().parent / 'undistort_results'
    output.mkdir(exist_ok=True)
    corrected_path = output / (args.image.stem + '_undistorted.png')
    comparison_path = output / (args.image.stem + '_comparison.png')
    save_image(corrected_path, corrected)
    save_image(comparison_path, comparison)
    print('Saved:', corrected_path)
    print('Saved:', comparison_path)


if __name__ == '__main__':
    main()
