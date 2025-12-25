#!/usr/bin/env python3
"""
ROS 2 Tracked Person Publisher for Robot Dog Following.

This module provides a ROS 2 node that tracks a target person and publishes
their position for robot navigation.

Topic
-----
/tracked_person/status : std_msgs/String (JSON)
    Published tracking status containing:
    - is_tracked : bool - Whether target is being tracked
    - x : float - Lateral offset in meters (+ right, - left)
    - z : float - Distance forward in meters

Usage
-----
Basic usage:
    python tracked_person_publisher.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine

With display:
    python tracked_person_publisher.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine --display

Manual enrollment only:
    python tracked_person_publisher.py --yolo-det yolo11n.engine --yolo-seg yolo11s-seg.engine --no-auto-enroll

Examples
--------
>>> # Check published topic
>>> ros2 topic echo /tracked_person/status
>>> # Output: {"is_tracked": true, "x": 0.35, "z": 2.1}
"""

import argparse
import json
import logging
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    logger.warning("pyrealsense2 not available")


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - yolo_det : str
            Path to YOLO detection TensorRT engine.
        - yolo_seg : str
            Path to YOLO segmentation TensorRT engine.
        - clip_model : str
            OpenCLIP model name.
        - clip_pretrained : str
            OpenCLIP pretrained weights.
        - clothing_threshold : float
            Lab clothing similarity threshold.
        - clip_threshold : float
            OpenCLIP similarity threshold.
        - min_mask_coverage : float
            Minimum mask coverage percentage.
        - search_interval : float
            Search mode feature extraction interval in seconds.
        - tracker : str
            Tracker type ('botsort' or 'bytetrack').
        - camera : int or None
            Webcam index, or None for RealSense.
        - width : int
            Frame width.
        - height : int
            Frame height.
        - fps : int
            Frame rate.
        - no_display : bool
            Whether to disable display window.
        - save_video : str or None
            Path to save output video.
        - no_auto_enroll : bool
            Whether to disable auto-enrollment.
        - publish_hz : float
            ROS 2 publish rate in Hz.
    """
    p = argparse.ArgumentParser(description="Person Following with ROS 2 Publishing")

    p.add_argument('--yolo-det', type=str, required=True,
                   help='Path to YOLO detection TensorRT engine')
    p.add_argument('--yolo-seg', type=str, required=True,
                   help='Path to YOLO segmentation TensorRT engine')
    p.add_argument('--clip-model', type=str, default='ViT-B-16',
                   help='OpenCLIP model name')
    p.add_argument('--clip-pretrained', type=str, default='laion2b_s34b_b88k',
                   help='OpenCLIP pretrained weights')
    p.add_argument('--clothing-threshold', type=float, default=0.8,
                   help='Lab clothing similarity threshold')
    p.add_argument('--clip-threshold', type=float, default=0.8,
                   help='OpenCLIP similarity threshold')
    p.add_argument('--min-mask-coverage', type=float, default=35.0,
                   help='Minimum mask coverage percentage')
    p.add_argument('--search-interval', type=float, default=0.33,
                   help='Search mode feature extraction interval (seconds)')
    p.add_argument('--tracker', type=str, default='botsort',
                   choices=['botsort', 'bytetrack'], help='Tracker type')
    p.add_argument('--camera', type=int, default=None,
                   help='Use webcam instead of RealSense')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--no-display', action='store_true',
                   help='Disable display window')
    p.add_argument('--save-video', type=str, default=None,
                   help='Save output to video file')
    p.add_argument('--no-auto-enroll', action='store_true',
                   help='Disable auto-enrollment')
    p.add_argument('--publish-hz', type=float, default=15.0,
                   help='ROS 2 publish rate (Hz)')

    return p.parse_args()


class RealSenseCamera:
    """
    RealSense D400 series camera wrapper.

    Provides access to Intel RealSense D400 series depth cameras with
    aligned color and depth streams and camera intrinsics for 3D projection.

    Parameters
    ----------
    width : int, optional
        Frame width in pixels, by default 640.
    height : int, optional
        Frame height in pixels, by default 480.
    fps : int, optional
        Frame rate in frames per second, by default 30.

    Attributes
    ----------
    pipeline : rs.pipeline
        RealSense pipeline for streaming.
    config : rs.config
        RealSense configuration.
    profile : rs.pipeline_profile
        Active pipeline profile.
    depth_scale : float
        Depth scale factor to convert raw depth values to meters.
    align : rs.align
        Alignment object to align depth frames to color frames.
    fx : float
        Focal length in x direction from camera intrinsics (pixels).
    cx : float
        Principal point x coordinate from camera intrinsics (pixels).

    Raises
    ------
    RuntimeError
        If pyrealsense2 is not available.

    Examples
    --------
    >>> camera = RealSenseCamera(width=640, height=480, fps=30)
    >>> color, depth = camera.get_frames()
    >>> camera.stop()
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 not available")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx = float(intr.fx)
        self.cx = float(intr.ppx)

        for _ in range(30):
            self.pipeline.wait_for_frames()

        logger.info(f"RealSense started: {width}x{height}@{fps}fps")
        logger.info(f"  Intrinsics: fx={self.fx:.1f}, cx={self.cx:.1f}")

    def get_frames(self):
        """
        Get aligned color and depth frames from the RealSense camera.

        Retrieves the latest color and depth frames, aligns them, and
        converts depth values to meters.

        Returns
        -------
        color_image : numpy.ndarray or None
            BGR color image of shape (height, width, 3) with dtype uint8.
            Returns None if frame acquisition fails.
        depth_image : numpy.ndarray or None
            Depth image of shape (height, width) with dtype float64,
            containing depth values in meters.
            Returns None if frame acquisition fails.
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale

        return color_image, depth_image

    def stop(self):
        """
        Stop the RealSense camera pipeline.

        Releases all resources associated with the camera pipeline.
        """
        self.pipeline.stop()
        logger.info("RealSense stopped")


class WebCamera:
    """
    Webcam wrapper with dummy depth data.

    Provides a webcam interface with simulated depth data for testing
    without a depth camera. Uses a fixed assumed depth for all pixels.

    Parameters
    ----------
    camera_idx : int, optional
        Camera device index, by default 0.
    width : int, optional
        Frame width in pixels, by default 640.
    height : int, optional
        Frame height in pixels, by default 480.

    Attributes
    ----------
    cap : cv2.VideoCapture
        OpenCV video capture object.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    fx : float
        Simulated focal length in x direction (pixels).
    cx : float
        Simulated principal point x coordinate (pixels).

    Raises
    ------
    RuntimeError
        If the camera cannot be opened.

    Examples
    --------
    >>> camera = WebCamera(camera_idx=0, width=640, height=480)
    >>> color, depth = camera.get_frames()
    >>> camera.stop()
    """

    def __init__(self, camera_idx: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_idx}")

        self.width = width
        self.height = height
        self.fx = 600.0
        self.cx = width / 2.0

        logger.info(f"Webcam {camera_idx} started: {width}x{height}")

    def get_frames(self):
        """
        Get color frame with dummy depth data.

        Retrieves the latest color frame and generates a dummy depth
        image with a constant depth value of 2.0 meters.

        Returns
        -------
        color_image : numpy.ndarray or None
            BGR color image of shape (height, width, 3) with dtype uint8.
            Returns None if frame acquisition fails.
        depth_image : numpy.ndarray or None
            Dummy depth image of shape (height, width) with dtype float32,
            filled with 2.0 meter depth values.
            Returns None if frame acquisition fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        depth = np.full((self.height, self.width), 2.0, dtype=np.float32)
        return frame, depth

    def stop(self):
        """
        Release the webcam.

        Releases all resources associated with the video capture.
        """
        self.cap.release()
        logger.info("Webcam stopped")


class TrackedPersonPublisher(Node):
    """
    ROS 2 node for publishing tracked person status.

    Publishes JSON messages to /tracked_person/status containing
    tracking state and position information for robot navigation.

    Attributes
    ----------
    pub_status : rclpy.publisher.Publisher
        ROS 2 publisher for status messages.
    publish_count : int
        Total number of messages published.

    Examples
    --------
    >>> rclpy.init()
    >>> node = TrackedPersonPublisher()
    >>> node.publish_status(is_tracked=True, x=0.5, z=2.0)
    >>> node.destroy_node()
    >>> rclpy.shutdown()
    """

    def __init__(self):
        """
        Initialize the ROS 2 node.

        Creates the node with name 'tracked_person_publisher' and sets up
        the publisher for the /tracked_person/status topic.
        """
        super().__init__("tracked_person_publisher")
        self.pub_status = self.create_publisher(String, "/tracked_person/status", 10)
        self.publish_count = 0

    def publish_status(self, is_tracked: bool, x: float = 0.0, z: float = 0.0):
        """
        Publish tracking status as JSON message.

        Publishes a JSON-formatted message containing the current tracking
        state and target position.

        Parameters
        ----------
        is_tracked : bool
            Whether the target person is currently being tracked.
        x : float, optional
            Lateral offset in meters (positive = right, negative = left),
            by default 0.0.
        z : float, optional
            Forward distance in meters, by default 0.0.

        Notes
        -----
        The published JSON format is:
        {"is_tracked": bool, "x": float, "z": float}
        """
        data = {
            "is_tracked": is_tracked,
            "x": round(x, 3),
            "z": round(z, 3)
        }

        msg = String()
        msg.data = json.dumps(data)
        self.pub_status.publish(msg)
        self.publish_count += 1


def compute_lateral_offset(bbox, distance: float, fx: float, cx: float) -> float:
    """
    Compute lateral offset using pinhole camera model.

    Calculates the horizontal offset of the target from the camera center
    using the pinhole camera projection model.

    Parameters
    ----------
    bbox : tuple of int
        Bounding box coordinates (x1, y1, x2, y2) in pixels.
    distance : float
        Distance to person in meters (Z coordinate).
    fx : float
        Focal length in x direction from camera intrinsics (pixels).
    cx : float
        Principal point x coordinate from camera intrinsics (pixels).

    Returns
    -------
    float
        Lateral offset in meters. Positive values indicate the target
        is to the right of center, negative values to the left.

    Notes
    -----
    Uses the pinhole camera model inverse projection:
        X = (u - cx) * Z / fx

    where:
        - u is the pixel x-coordinate of the bounding box center
        - cx is the principal point x-coordinate
        - Z is the depth (distance)
        - fx is the focal length in x direction

    Examples
    --------
    >>> bbox = (300, 100, 400, 300)  # target slightly right of center
    >>> offset = compute_lateral_offset(bbox, distance=2.0, fx=600.0, cx=320.0)
    >>> print(f"Offset: {offset:.2f}m")
    """
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0
    return (u - cx) * distance / fx


def draw_visualization(frame: np.ndarray, result: dict, system,
                       is_tracked: bool, x_offset: float, distance: float,
                       publish_count: int) -> np.ndarray:
    """
    Draw tracking visualization overlay on frame.

    Renders bounding boxes, track IDs, target information, status bars,
    and ROS 2 publishing information onto the input frame.

    Parameters
    ----------
    frame : numpy.ndarray
        Input BGR color frame of shape (height, width, 3) with dtype uint8.
    result : dict
        Processing result dictionary from PersonFollowingSystem containing:
        - 'all_tracks' : list of dict
            List of tracked objects with 'bbox' and 'track_id' keys.
        - 'target_found' : bool
            Whether the target is currently tracked.
        - 'bbox' : tuple of int, optional
            Target bounding box (x1, y1, x2, y2) if target is found.
        - 'status' : str
            Current system status ('TRACKING_ACTIVE', 'SEARCHING', 'INACTIVE').
        - 'fps' : float
            Current processing frame rate.
        - 'time_lost' : float, optional
            Time since target was lost (during search mode).
    system : PersonFollowingSystem
        The person following system instance for accessing target info
        and candidate information.
    is_tracked : bool
        Whether the target is currently being tracked with valid position.
    x_offset : float
        Lateral offset of target in meters.
    distance : float
        Forward distance to target in meters.
    publish_count : int
        Number of ROS 2 messages published.

    Returns
    -------
    numpy.ndarray
        Annotated BGR frame of shape (height, width, 3) with dtype uint8,
        containing visualization overlays.

    Notes
    -----
    The visualization includes:
    - Green boxes for actively tracked targets
    - Orange boxes for targets being searched
    - Gray boxes for non-target tracks
    - Position information (X, Z coordinates)
    - Top status bar with tracking status, tracked state, FPS, and publish count
    - Bottom bar with keyboard controls
    - Search mode indicator when target is lost
    """
    display = frame.copy()
    H, W = display.shape[:2]

    # Draw all tracks
    for track in result.get('all_tracks', []):
        x1, y1, x2, y2 = track['bbox']
        track_id = track['track_id']

        # Color: green ONLY if target AND is_tracked
        if track_id == system.target.track_id and is_tracked:
            color = (0, 255, 0)  # Green - actively tracked
            thickness = 3
        elif track_id == system.target.track_id and not is_tracked:
            color = (0, 165, 255)  # Orange - target but lost/searching
            thickness = 2
        else:
            color = (128, 128, 128)  # Gray - not target
            thickness = 1

        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(display, f"ID:{track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw target info (only when actively tracked)
    if is_tracked and result.get('target_found') and 'bbox' in result:
        x1, y1, x2, y2 = result['bbox']

        # Position info text
        cv2.putText(display, f"Z:{distance:.2f}m X:{x_offset:+.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Center crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)

    # Draw candidates
    for cand in system.get_candidates_info():
        x1, y1, x2, y2 = cand['bbox']
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(display, f"L:{cand['clothing_sim']:.2f} C:{cand['clip_sim']:.2f}",
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # Status bar
    status = result.get('status', 'UNKNOWN')
    fps = result.get('fps', 0)

    status_color = {
        'TRACKING_ACTIVE': (0, 255, 0),
        'SEARCHING': (0, 165, 255),
        'INACTIVE': (128, 128, 128)
    }.get(status, (255, 255, 255))

    tracked_color = (0, 255, 0) if is_tracked else (0, 0, 255)

    cv2.rectangle(display, (0, 0), (W, 35), (0, 0, 0), -1)
    cv2.putText(display, f"Status: {status}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(display, f"Tracked: {'YES' if is_tracked else 'NO'}", (220, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracked_color, 1)
    cv2.putText(display, f"FPS:{fps:.0f} Pub:{publish_count}", (W - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if system.target.track_id is not None:
        info = f"Track: {system.target.track_id} | {system.target.get_quality_summary()}"
        cv2.putText(display, info, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.rectangle(display, (0, H - 30), (W, H), (0, 0, 0), -1)
    cv2.putText(display, "'e'=enroll | 'c'=clear | 's'=status | 'q'=quit", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if status == 'SEARCHING':
        time_lost = result.get('time_lost', 0)
        cv2.putText(display, f"SEARCHING... ({time_lost:.1f}s)", (W//2 - 80, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    return display


def main():
    """
    Main entry point for the ROS 2 tracked person publisher.

    Initializes the ROS 2 node, camera, and person following system,
    then runs the main processing loop with visualization, keyboard
    controls, and ROS 2 message publishing.

    The function handles:
    - ROS 2 node initialization and shutdown
    - Camera initialization (RealSense or webcam)
    - Person following system initialization
    - Main processing loop with frame processing
    - Auto-enrollment of nearest person (if enabled)
    - Rate-limited ROS 2 message publishing
    - Optional visualization with keyboard controls
    - Graceful shutdown on exit

    Raises
    ------
    SystemExit
        If RealSense camera is not available and no webcam specified.
    KeyboardInterrupt
        Gracefully handled when user interrupts execution.

    Notes
    -----
    Keyboard controls (when display is enabled):
    - 'e' : Manually enroll the nearest person as target
    - 'c' : Clear the current target
    - 's' : Print current system status
    - 'q' : Quit the application
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PERSON FOLLOWING SYSTEM - Lab + OpenCLIP + ROS 2")
    logger.info("=" * 60)

    # Initialize ROS 2
    rclpy.init()
    ros_node = TrackedPersonPublisher()
    logger.info("ROS 2 node initialized")
    logger.info("Publishing to: /tracked_person/status")

    # Initialize camera
    logger.info("Initializing camera...")
    if args.camera is not None:
        camera = WebCamera(args.camera, args.width, args.height)
    else:
        if not HAS_REALSENSE:
            logger.error("RealSense not available. Use --camera N for webcam.")
            rclpy.shutdown()
            sys.exit(1)
        camera = RealSenseCamera(args.width, args.height, args.fps)

    # Initialize system
    logger.info("Initializing person following system...")
    from person_following_system import PersonFollowingSystem

    system = PersonFollowingSystem(
        yolo_detection_engine=args.yolo_det,
        yolo_seg_engine=args.yolo_seg,
        device='cuda',
        tracker_type=args.tracker,
        use_clip=True,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clothing_threshold=args.clothing_threshold,
        clip_threshold=args.clip_threshold,
        min_mask_coverage=args.min_mask_coverage,
        search_interval=args.search_interval
    )

    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps,
                                       (args.width, args.height))
        logger.info(f"Saving video to: {args.save_video}")

    auto_enroll = not args.no_auto_enroll
    publish_period = 1.0 / max(args.publish_hz, 1.0)
    last_publish_time = 0.0

    logger.info("=" * 60)
    logger.info(f"Auto-enroll: {'ON' if auto_enroll else 'OFF'}")
    logger.info(f"Publish rate: {args.publish_hz} Hz")
    logger.info("Controls: 'e'=enroll, 'c'=clear, 's'=status, 'q'=quit")
    logger.info("=" * 60)

    try:
        while rclpy.ok():
            color_frame, depth_frame = camera.get_frames()
            if color_frame is None:
                rclpy.spin_once(ros_node, timeout_sec=0.0)
                continue

            result = system.process_frame(color_frame, depth_frame)
            status = result.get('status', 'UNKNOWN')

            # Check tracking
            is_tracked = (
                result.get('target_found', False) and
                result.get('bbox') is not None and
                result.get('distance') is not None
            )

            x_offset, distance = 0.0, 0.0
            if is_tracked:
                distance = result['distance']
                if np.isfinite(distance) and distance > 0.1:
                    x_offset = compute_lateral_offset(result['bbox'], distance,
                                                      camera.fx, camera.cx)
                else:
                    is_tracked = False

            # Auto-enroll
            if auto_enroll and status == "INACTIVE":
                try:
                    if system.enroll_target(color_frame, depth_frame):
                        logger.info("Auto-enrolled nearest person")
                except Exception:
                    pass

            # Publish
            current_time = time.time()
            if current_time - last_publish_time >= publish_period:
                ros_node.publish_status(is_tracked, x_offset, distance)
                last_publish_time = current_time

                if ros_node.publish_count % 30 == 0 and is_tracked:
                    logger.info(f"Target: x={x_offset:+.2f}m z={distance:.2f}m")

            # Visualization
            if not args.no_display:
                display = draw_visualization(color_frame, result, system,
                                            is_tracked, x_offset, distance,
                                            ros_node.publish_count)
                cv2.imshow("Person Following - ROS 2", display)

                if video_writer:
                    video_writer.write(display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('e'):
                    logger.info("Enrolling target...")
                    try:
                        if system.enroll_target(color_frame, depth_frame):
                            logger.info("Manually enrolled target")
                        else:
                            logger.warning("Enrollment failed")
                    except Exception as e:
                        logger.warning(f"Enrollment failed: {e}")
                elif key == ord('c'):
                    system.clear_target()
                    logger.info("Target cleared")
                elif key == ord('s'):
                    status_dict = system.get_status()
                    logger.info("=" * 40)
                    for k, v in status_dict.items():
                        logger.info(f"  {k}: {v}")
                    logger.info(f"  ROS publish count: {ros_node.publish_count}")
                    logger.info("=" * 40)

            rclpy.spin_once(ros_node, timeout_sec=0.0)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        camera.stop()
        if video_writer:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        logger.info(f"Published {ros_node.publish_count} messages")
        ros_node.destroy_node()
        rclpy.shutdown()
        logger.info("Done.")


if __name__ == "__main__":
    main()