# Ros2_ws_local_test

 <!-- 

HOW TO RUN THIS CODE - 
TERMINAL 1 :
cd ~/ros2_ws
colcon build --symlink-install --packages-select yolo_perception_pkg

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run yolo_perception_pkg perception_offline --ros-args -p show_display:=false

TERMINAL 2 :
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run yolo_perception_pkg web_hud_streamer

 -->