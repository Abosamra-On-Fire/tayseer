#!/usr/bin/env python3
"""
check_slam_env.py  ─  Run this BEFORE roslaunch to verify everything is ready.

  python3 check_slam_env.py
  python3 check_slam_env.py --scan-topic /limo/scan
"""

import subprocess
import sys
import argparse


RED   = "\033[91m"
GRN   = "\033[92m"
YLW   = "\033[93m"
RST   = "\033[0m"

def ok(msg):  print(f"  {GRN}✓{RST}  {msg}")
def warn(msg): print(f"  {YLW}⚠{RST}  {msg}")
def fail(msg): print(f"  {RED}✗{RST}  {msg}")


def shell(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-topic", default="/limo/scan")
    args = parser.parse_args()

    print("\n══════════  LIMO SLAM environment check  ══════════\n")

    # 1. ROS environment
    rc, out, _ = shell("echo $ROS_DISTRO")
    if "noetic" in out.lower():
        ok(f"ROS distro: {out}")
    elif out:
        warn(f"ROS distro is '{out}' – tested on noetic")
    else:
        fail("ROS_DISTRO not set – did you source /opt/ros/noetic/setup.bash?")

    # 2. roscore running?
    rc, _, _ = shell("rostopic list 2>/dev/null")
    if rc == 0:
        ok("roscore is running")
    else:
        fail("roscore is NOT running – start it with:  roscore")
        print("\n  Fix the above and re-run this script.\n")
        sys.exit(1)

    # 3. Scan topic present?
    rc, out, _ = shell(f"rostopic info {args.scan_topic} 2>/dev/null")
    if rc == 0:
        ok(f"Scan topic {args.scan_topic} is being published")
        # check message type
        for line in out.splitlines():
            if "Type:" in line:
                if "LaserScan" in line:
                    ok(f"  Type: {line.split(':',1)[1].strip()}")
                else:
                    fail(f"  Wrong type: {line}  (need sensor_msgs/LaserScan)")
    else:
        fail(f"Scan topic {args.scan_topic} not found")
        warn("  If using real hardware:  make sure the LIDAR driver is running")
        warn("  If replaying a bag  :  rosbag play --clock your_file.bag")

    # 4. Eigen3
    rc, _, _ = shell("dpkg -s libeigen3-dev 2>/dev/null | grep 'Status: install ok'")
    if rc == 0:
        ok("libeigen3-dev installed")
    else:
        fail("libeigen3-dev not found – run:  sudo apt install libeigen3-dev")

    # 5. Package built?
    rc, out, _ = shell("rospack find limo_slam 2>/dev/null")
    if rc == 0:
        ok(f"limo_slam package found at: {out}")
    else:
        fail("limo_slam package not found in ROS_PACKAGE_PATH")
        warn("  Build it:  cd ~/catkin_ws && catkin_make && source devel/setup.bash")

    # 6. Binary exists?
    rc, out, _ = shell("rosrun limo_slam pose_graph_node --help 2>&1 | head -1")
    if rc == 0 or "Usage" in out or "pose_graph" in out:
        ok("pose_graph_node binary found")
    else:
        rc2, _, _ = shell("find ~/catkin_ws/devel -name pose_graph_node 2>/dev/null | head -1")
        if rc2 == 0:
            ok("pose_graph_node binary found in devel/")
        else:
            fail("pose_graph_node binary not found – re-run catkin_make")

    print("\n══════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()