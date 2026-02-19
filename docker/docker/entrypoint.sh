#!/usr/bin/env bash
set -e

# ROS2 env
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
fi

# If your workspace has an overlay, you can optionally source it:
# if [ -f "/workspace/starVLA/install/setup.bash" ]; then
#   source /workspace/starVLA/install/setup.bash
# fi

exec "$@"

