#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <graph_slam/AddNode.h>
#include <graph_slam/AddEdge.h>
#include <graph_slam/ScanMatch.h>

#include "types.hpp"
#include <cmath>

using namespace std;
namespace graph_slam {

double scanDegeneracy(const sensor_msgs::LaserScan& scan)
{
    double mx = 0, my = 0; int N = 0;
    vector<pair<double,double>> pts;
    float angle = scan.angle_min;
    for (size_t i = 0; i < scan.ranges.size(); ++i, angle += scan.angle_increment) {
        float r = scan.ranges[i];
        if (!isfinite(r) || r <= scan.range_min || r >= scan.range_max) continue;
        pts.push_back({r*cos(angle), r*sin(angle)});
        mx += pts.back().first; my += pts.back().second; ++N;
    }
    if (N < 5) return 0.0;
    mx /= N; my /= N;
    double sxx = 0, sxy = 0, syy = 0;
    for (auto& p : pts) {
        double dx = p.first - mx, dy = p.second - my;
        sxx += dx*dx; sxy += dx*dy; syy += dy*dy;
    }
    double trace = sxx + syy;
    double det   = sxx*syy - sxy*sxy;
    double disc  = max(0.0, trace*trace - 4.0*det);
    double l1    = (trace + sqrt(disc)) / 2.0;
    double l2    = (trace - sqrt(disc)) / 2.0;
    return (l1 < 1e-9) ? 0.0 : l2 / l1;
}

class Pipeline {
public:
    Pipeline(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh), initialized_(false), last_node_id_(-1)
    {
        private_nh_.param("min_dist_m",                   min_dist_m_,          0.3);
        private_nh_.param("min_angle_deg",                min_angle_deg_,       30.0);
        private_nh_.param("use_scan_matching_from_odom",  use_icp_odom_,        true);
        private_nh_.param("fallback_to_odom_on_icp_fail", fallback_odom_,       true);
        private_nh_.param("corridor_degeneracy_thr",      corridor_degen_thr_,  0.05);
        private_nh_.param("icp_score_threshold",          icp_score_threshold_, 0.05);

        min_angle_rad_ = min_angle_deg_ * M_PI / 180.0;

        sub_odom_ = nh_.subscribe("/odom",     50, &Pipeline::odomCallback, this);
        sub_scan_ = nh_.subscribe("limo/scan", 10, &Pipeline::scanCallback, this);

        clt_add_node_   = nh_.serviceClient<graph_slam::AddNode>  ("graph_slam/add_node");
        clt_add_edge_   = nh_.serviceClient<graph_slam::AddEdge>  ("graph_slam/add_edge");
        clt_scan_match_ = nh_.serviceClient<graph_slam::ScanMatch>("graph_slam/scan_match");
        clt_add_node_.waitForExistence();
        clt_add_edge_.waitForExistence();
        clt_scan_match_.waitForExistence();

        ROS_INFO("[Pipeline] min_dist=%.2fm min_angle=%.1fdeg icp=%d fallback=%d corridor_thr=%.2f icp_score_thr=%.3f",
                 min_dist_m_, min_angle_deg_, use_icp_odom_, fallback_odom_,
                 corridor_degen_thr_, icp_score_threshold_);
    }

private:
    bool   initialized_;
    int    last_node_id_;
    double current_x_ = 0, current_y_ = 0, current_theta_ = 0;
    double last_x_    = 0, last_y_    = 0, last_theta_    = 0;

    sensor_msgs::LaserScan last_scan_, current_scan_;
    bool has_scan_ = false;

    double min_dist_m_, min_angle_deg_, min_angle_rad_;
    bool   use_icp_odom_, fallback_odom_;
    double corridor_degen_thr_;
    double icp_score_threshold_;
    double odom_cov_x_ = 0.0001, odom_cov_y_ = 0.0001, odom_cov_th_ = 0.001;

    ros::NodeHandle    nh_, private_nh_;
    ros::Subscriber    sub_odom_, sub_scan_;
    ros::ServiceClient clt_add_node_, clt_add_edge_, clt_scan_match_;

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    { current_scan_ = *msg; has_scan_ = true; }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                          msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        current_x_     = msg->pose.pose.position.x;
        current_y_     = msg->pose.pose.position.y;
        current_theta_ = yaw;

        double cx = msg->pose.covariance[0];
        double cy = msg->pose.covariance[7];
        double ct = msg->pose.covariance[35];
        odom_cov_x_  = (cx  > 1e-9) ? cx  : 0.0001;
        odom_cov_y_  = (cy  > 1e-9) ? cy  : 0.0001;
        odom_cov_th_ = (ct  > 1e-9) ? ct  : 0.001;

        if (!initialized_) { tryInitialize(); return; }

        double dx     = current_x_ - last_x_, dy = current_y_ - last_y_;
        double dist   = sqrt(dx*dx + dy*dy);
        double dtheta = fabs(wrapAngle(current_theta_ - last_theta_));

        if (dist >= min_dist_m_ || dtheta >= min_angle_rad_)
            createKeyframe();
    }

    void tryInitialize()
    {
        if (!has_scan_) return;
        graph_slam::AddNode srv;
        srv.request.x         = current_x_;
        srv.request.y         = current_y_;
        srv.request.theta     = current_theta_;
        srv.request.scan      = current_scan_;
        srv.request.timestamp = ros::Time::now().toSec();

        if (clt_add_node_.call(srv) && srv.response.success) {
            last_node_id_ = srv.response.id;
            last_x_ = current_x_; last_y_ = current_y_; last_theta_ = current_theta_;
            last_scan_ = current_scan_;
            initialized_ = true;
            ROS_INFO("[Pipeline] First node %d at (%.2f,%.2f,%.1fdeg)",
                     last_node_id_, current_x_, current_y_, current_theta_*180/M_PI);
        } else ROS_ERROR("[Pipeline] Failed to add first node");
    }

    void createKeyframe()
    {
        if (!has_scan_) return;

        graph_slam::AddNode node_srv;
        node_srv.request.x         = current_x_;
        node_srv.request.y         = current_y_;
        node_srv.request.theta     = current_theta_;
        node_srv.request.scan      = current_scan_;
        node_srv.request.timestamp = ros::Time::now().toSec();
        if (!clt_add_node_.call(node_srv) || !node_srv.response.success) {
            ROS_ERROR("[Pipeline] add_node failed"); return;
        }
        int new_id = node_srv.response.id;

        double odom_dx  = current_x_ - last_x_;
        double odom_dy  = current_y_ - last_y_;
        double odom_dth = wrapAngle(current_theta_ - last_theta_);

        double c = cos(last_theta_), s = sin(last_theta_);
        double kf_dx  =  c*odom_dx + s*odom_dy;
        double kf_dy  = -s*odom_dx + c*odom_dy;
        double kf_dth = odom_dth;

        Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
        bool icp_ok = false;

        if (use_icp_odom_) {
            graph_slam::ScanMatch sm;
            sm.request.reference_scan = last_scan_;
            sm.request.current_scan   = current_scan_;
            sm.request.init_dx        = kf_dx;
            sm.request.init_dy        = kf_dy;
            sm.request.init_dtheta    = kf_dth;

            if (clt_scan_match_.call(sm) && sm.response.success &&
                sm.response.score <= icp_score_threshold_)
            {
                double degen       = scanDegeneracy(current_scan_);
                bool   in_corridor = (degen < corridor_degen_thr_);

                if (in_corridor) {
                    kf_dx  = sm.response.dx;
                    kf_dth = sm.response.dtheta;
                    information(0,0) = 1.0 / max(odom_cov_x_, 1e-6);
                    information(1,1) = 1.0 / max(odom_cov_y_, 1e-6);
                    information(2,2) = min(1.0 / max(sm.response.score, 1e-6) * 0.3, 300.0);
                    ROS_DEBUG("[Pipeline] Corridor: ICP dx+rot, odom dy, degen=%.3f", degen);
                } else {
                    kf_dx  = sm.response.dx;
                    kf_dy  = sm.response.dy;
                    kf_dth = sm.response.dtheta;
                    for (int r = 0; r < 3; ++r)
                        for (int c2 = 0; c2 < 3; ++c2)
                            information(r,c2) = sm.response.information[r*3+c2];
                }
                icp_ok = true;
            } else {
                if (!fallback_odom_) { ROS_WARN("[Pipeline] ICP failed/poor, skipping"); return; }
                ROS_WARN("[Pipeline] ICP failed or poor score, using odometry");
                information(0,0) = 1.0 / max(odom_cov_x_, 1e-6);
                information(1,1) = 1.0 / max(odom_cov_y_, 1e-6);
                information(2,2) = 1.0 / max(odom_cov_th_, 1e-6);
            }
        } else {
            information(0,0) = 1.0 / max(odom_cov_x_, 1e-6);
            information(1,1) = 1.0 / max(odom_cov_y_, 1e-6);
            information(2,2) = 1.0 / max(odom_cov_th_, 1e-6);
        }

        graph_slam::AddEdge edge;
        edge.request.from_id  = last_node_id_;
        edge.request.to_id    = new_id;
        edge.request.type     = Edge::ODOMETRY;
        edge.request.dx       = kf_dx;
        edge.request.dy       = kf_dy;
        edge.request.dtheta   = kf_dth;
        for (int r = 0; r < 3; ++r)
            for (int c2 = 0; c2 < 3; ++c2)
                edge.request.information.push_back(information(r,c2));

        if (!clt_add_edge_.call(edge) || !edge.response.success) {
            ROS_ERROR("[Pipeline] add_edge failed"); return;
        }

        last_node_id_ = new_id;
        last_x_ = current_x_; last_y_ = current_y_; last_theta_ = current_theta_;
        last_scan_ = current_scan_;
        ROS_INFO("[Pipeline] Node %d<-%d odom(%.2f,%.2f,%.1fdeg) icp=%d",
                 new_id, edge.request.from_id, odom_dx, odom_dy, odom_dth*180/M_PI, icp_ok);
    }
};

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "graph_pipeline");
    ros::NodeHandle nh, pnh("~");
    graph_slam::Pipeline p(nh, pnh);
    ros::spin();
    return 0;
}