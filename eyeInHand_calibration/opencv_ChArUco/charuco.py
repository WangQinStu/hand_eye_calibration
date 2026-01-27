import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CHARUCO_CONFIG = {
    "squares_x": 4,
    "squares_y": 4,
    "square_length": 0.04,
    "marker_length": 0.03,
    "aruco_dict": "DICT_4X4_100",
}


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_intrinsics(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["distortion_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def load_poses(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        int(entry["index"]): (
            np.array(entry["rotation"], dtype=np.float64),
            np.array(entry["translation"], dtype=np.float64),
        )
        for entry in data
    }


def build_charuco_detector(board_obj, aruco_params, min_markers):
    if not hasattr(cv2.aruco, "CharucoDetector"):
        return None

    charuco_params = None
    if hasattr(cv2.aruco, "CharucoParameters"):
        charuco_params = cv2.aruco.CharucoParameters()
        if hasattr(charuco_params, "minMarkers"):
            # OpenCV 4.12 enforces minMarkers in [0, 2].
            charuco_params.minMarkers = max(0, min(min_markers, 2))

    if charuco_params is not None:
        return cv2.aruco.CharucoDetector(board_obj, charuco_params, aruco_params)

    try:
        return cv2.aruco.CharucoDetector(board_obj, aruco_params)
    except TypeError:
        return cv2.aruco.CharucoDetector(board_obj)


def interpolate_charuco_corners(
    marker_corners,
    marker_ids,
    gray_image,
    board_obj,
    camera_matrix,
    dist_coeffs,
    charuco_detector,
):
    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        return cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray_image,
            board_obj,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )

    if charuco_detector is not None:
        try:
            result = charuco_detector.detectBoard(
                gray_image, markerCorners=marker_corners, markerIds=marker_ids
            )
        except TypeError:
            try:
                result = charuco_detector.detectBoard(gray_image, marker_corners, marker_ids)
            except TypeError:
                result = charuco_detector.detectBoard(gray_image)

        if len(result) == 2:
            ch_corners, ch_ids = result
        else:
            ch_corners, ch_ids, _, _ = result
        ret = 0 if ch_ids is None else len(ch_ids)
        return ret, ch_corners, ch_ids

    raise AttributeError(
        "cv2.aruco lacks ChArUco interpolation APIs. Install opencv-contrib-python."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ChArUco-based hand-eye calibration")
    parser.add_argument("--images-dir", default="images", help="Directory containing img_XX.png")
    parser.add_argument("--poses-json", default="pose1.json", help="Pose JSON file")
    parser.add_argument("--intrinsics-json", default="camera_intrinsics.json", help="Camera intrinsics JSON")

    parser.add_argument("--squares-x", type=int, default=DEFAULT_CHARUCO_CONFIG["squares_x"])
    parser.add_argument("--squares-y", type=int, default=DEFAULT_CHARUCO_CONFIG["squares_y"])
    parser.add_argument("--square-length", type=float, default=DEFAULT_CHARUCO_CONFIG["square_length"])
    parser.add_argument("--marker-length", type=float, default=DEFAULT_CHARUCO_CONFIG["marker_length"])
    parser.add_argument("--aruco-dict", default=DEFAULT_CHARUCO_CONFIG["aruco_dict"])

    parser.add_argument("--min-markers", type=int, default=4, help="Minimum detected ArUco markers")
    parser.add_argument("--min-charuco-corners", type=int, default=4, help="Minimum ChArUco corners")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")

    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    images_dir = resolve_path(args.images_dir, base_dir)
    poses_path = resolve_path(args.poses_json, base_dir)
    intrinsics_path = resolve_path(args.intrinsics_json, base_dir)

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not poses_path.exists():
        raise SystemExit(f"Pose JSON not found: {poses_path}")
    if not intrinsics_path.exists():
        raise SystemExit(f"Intrinsics JSON not found: {intrinsics_path}")

    print(f"OpenCV: {cv2.__version__}\n")

    camera_matrix, dist_coeffs = load_intrinsics(intrinsics_path)
    print("使用已知相机内参:")
    print(
        f"  fx={camera_matrix[0, 0]:.1f}, fy={camera_matrix[1, 1]:.1f}, "
        f"cx={camera_matrix[0, 2]:.1f}, cy={camera_matrix[1, 2]:.1f}\n"
    )

    poses = load_poses(poses_path)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        aruco_dict,
    )

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    charuco_detector = build_charuco_detector(board, params, args.min_markers)

    visualize = not args.no_visualize

    gripper_Rs, gripper_ts = [], []
    target_Rs, target_ts = [], []
    charuco_obj_pts = []
    charuco_img_pts = []
    processed_count = 0

    print("处理图像...")
    print("=" * 60)

    for img_path in sorted(images_dir.glob("img_*.png")):
        idx = int(img_path.stem.split("_")[1])
        if idx not in poses:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)
        if hasattr(cv2.aruco, "refineDetectedMarkers"):
            corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                gray,
                board,
                corners,
                ids,
                rejected,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
            )

        ids_count = 0 if ids is None else len(ids)
        if ids is None or ids_count < args.min_markers:
            print(f"✗ [{idx:02d}] 标记不足: {ids_count}")
            continue

        ret, ch_corners, ch_ids = interpolate_charuco_corners(
            corners,
            ids,
            gray,
            board,
            camera_matrix,
            dist_coeffs,
            charuco_detector,
        )

        if ret is None or ret < args.min_charuco_corners:
            print(f"✗ [{idx:02d}] 角点不足: {ret if ret else 0}")
            continue

        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            print(f"✗ [{idx:02d}] 位姿估计失败")
            continue

        r_deg, t = poses[idx]
        angles = np.deg2rad(r_deg)
        rz = cv2.Rodrigues(np.array([0, 0, angles[0]]))[0]
        ry = cv2.Rodrigues(np.array([0, angles[1], 0]))[0]
        rx = cv2.Rodrigues(np.array([angles[2], 0, 0]))[0]
        r_gripper = rz @ ry @ rx

        gripper_Rs.append(r_gripper)
        gripper_ts.append(t.reshape(3, 1))
        target_Rs.append(cv2.Rodrigues(rvec)[0])
        target_ts.append(tvec)
        charuco_obj_pts.append(obj_pts)
        charuco_img_pts.append(img_pts)

        if visualize:
            vis_img = img.copy()
            cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis_img, ch_corners, ch_ids, (0, 255, 0))

            axis_length = args.square_length
            cv2.drawFrameAxes(vis_img, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

            info = f"[{idx:02d}] Markers: {ids_count}, Corners: {ret}"
            cv2.putText(
                vis_img,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Calibration", vis_img)
            cv2.waitKey(200)

        print(f"✓ [{idx:02d}] 标记={ids_count}, 角点={ret}")
        processed_count += 1

    if visualize:
        cv2.destroyAllWindows()

    print("=" * 60)
    print(f"有效图像: {processed_count}\n")

    if processed_count < 3:
        raise SystemExit("❌ 有效图像不足（需要 ≥3 张）")

    print("执行手眼标定...")

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        gripper_Rs,
        gripper_ts,
        target_Rs,
        target_ts,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    r_gripper2cam = r_cam2gripper.T
    t_gripper2cam = -r_cam2gripper.T @ t_cam2gripper

    total_error = 0.0
    num_points = 0

    for obj_pts, img_pts, rmat, tvec in zip(
        charuco_obj_pts, charuco_img_pts, target_Rs, target_ts
    ):
        projected, _ = cv2.projectPoints(
            obj_pts,
            cv2.Rodrigues(rmat)[0],
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        img_pts = img_pts.reshape(-1, 2)
        errors = np.linalg.norm(img_pts - projected, axis=1)
        total_error += np.sum(errors ** 2)
        num_points += len(errors)

    reprojection_error = np.sqrt(total_error / max(num_points, 1))

    result = {
        "opencv_version": cv2.__version__,
        "charuco_board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length": args.square_length,
            "marker_length": args.marker_length,
            "aruco_dict": args.aruco_dict,
        },
        "calibration": {
            "images_used": processed_count,
            "reprojection_error": float(reprojection_error),
        },
        "camera_matrix": camera_matrix.tolist(),
        "camera_params": {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        },
        "distortion": dist_coeffs.tolist(),
        "hand_eye": {
            "camera_in_gripper": {
                "R": r_cam2gripper.tolist(),
                "t_m": t_cam2gripper.ravel().tolist(),
                "t_mm": (t_cam2gripper.ravel() * 1000).tolist(),
            },
            "gripper_in_camera": {
                "R": r_gripper2cam.tolist(),
                "t_m": t_gripper2cam.ravel().tolist(),
                "t_mm": (t_gripper2cam.ravel() * 1000).tolist(),
            },
        },
    }

    with open("result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✓ 标定完成！")
    print("=" * 60)

    t_mm = t_cam2gripper.ravel() * 1000
    print("\n相机在末端坐标系 (mm):")
    print(f"  X = {t_mm[0]:7.2f}")
    print(f"  Y = {t_mm[1]:7.2f}")
    print(f"  Z = {t_mm[2]:7.2f}")

    print("\n旋转矩阵:")
    print(r_cam2gripper)

    print("\n结果已保存: result.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
