#pragma once

#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sensor_msgs/msg/laser_scan.hpp>

namespace graph_slam {

struct PoseNode {
    int id;
    double x;
    double y;
    double theta;

    sensor_msgs::msg::LaserScan scan;

    double timestamp;

    Eigen::Vector3d pose() const {
        return Eigen::Vector3d(x, y, theta);
    }
};

struct Edge {
    enum Type {
        ODOMETRY = 0,
        LOOP_CLOSURE = 1
    };

    int from_id;
    int to_id;
    Type type;

    double dx;
    double dy;
    double dtheta;

    Eigen::Matrix3d information;
};

struct Point2D {
    double x;
    double y;
};

using Cloud = std::vector<Point2D>;

struct ScanMatchResult {
    bool success;
    double dx;
    double dy;
    double dtheta;
    double fitness_score;
    Eigen::Matrix3d information;
};

inline double wrapAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

}  