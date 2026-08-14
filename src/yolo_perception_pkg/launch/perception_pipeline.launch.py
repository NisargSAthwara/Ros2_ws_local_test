"""
TH OWL — Perception Pipeline Launch File
=========================================
Launches both perception stages together:
  1. perception_carla_node  — Weather classification + image restoration
  2. yolo_detector_node     — Object detection on cleaned frames

All weight file paths are resolved relative to the current user's home
directory using os.path.expanduser('~'). No hardcoded usernames.

Override any path at launch time:
  ros2 launch yolo_perception_pkg perception_pipeline.launch.py

Or pass custom parameters:
  ros2 run yolo_perception_pkg perception_carla \\
    --ros-args -p classifier_weights_path:=/path/to/model.pt
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node

# -------------------------------------------------------------------------------------
#  Portable asset directory resolution
# -------------------------------------------------------------------------------------
_ASSETS = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'assets', 'pt_files')


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([

        # ── Stage 1: Perception Node ─────────────────────────────────────────────
        Node(
            package='yolo_perception_pkg',
            executable='perception_carla',
            name='perception_carla_node',
            output='screen',
            parameters=[{
                'restorer_weights_path':    os.path.join(_ASSETS, 'restorer.pt'),
                'classifier_weights_path':  os.path.join(_ASSETS, 'best_effi.pt'),
            }],
        ),

        # ── Stage 2: YOLO Detector Node ──────────────────────────────────────────
        Node(
            package='yolo_perception_pkg',
            executable='yolo_detector',
            name='yolo_detector_node',
            output='screen',
            parameters=[{
                'yolo_weights_path':        os.path.join(_ASSETS, 'yolo.pt'),
                'confidence_threshold':     0.5,
            }],
        ),

    ])
