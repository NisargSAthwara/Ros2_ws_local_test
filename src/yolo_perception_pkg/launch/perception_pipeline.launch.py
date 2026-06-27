import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_perception_pkg',
            executable='perception_carla',
            name='perception_carla_node',
            output='screen',
            parameters=[{
                'restorer_weights_path': '/home/dell_ubuntu/ros2_ws/assets/pt_files/restorer.pt',
                'classifier_weights_path': '/home/dell_ubuntu/ros2_ws/assets/pt_files/classifier.pt'
            }]
        ),
        Node(
            package='yolo_perception_pkg',
            executable='yolo_detector',
            name='yolo_detector_node',
            output='screen',
            parameters=[{
                'yolo_weights_path': '/home/dell_ubuntu/ros2_ws/assets/pt_files/yolo.pt',
                'confidence_threshold': 0.5
            }]
        )
    ])
