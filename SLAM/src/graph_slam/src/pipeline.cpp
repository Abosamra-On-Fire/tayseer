#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "graph_slam/srv/add_node.hpp"
#include "graph_slam/srv/add_edge.hpp"
#include "graph_slam/srv/scan_match.hpp"

#include "types.hpp"
#include <atomic>
#include <cmath>

using namespace std;
namespace graph_slam {

double scanDegeneracy(const sensor_msgs::msg::LaserScan& scan)
{
    double mx = 0, my = 0; 
    int N = 0;
    vector<pair<double,double>> pts;
    double angle = scan.angle_min;
    for (int i = 0; i < scan.ranges.size(); ++i) {
        double r = scan.ranges[i];
        if (!isfinite(r) || r <= scan.range_min || r >= scan.range_max) {
            angle += scan.angle_increment;
            continue;
        }
        pts.push_back({r*cos(angle), r*sin(angle)});
        mx += pts.back().first;
        my += pts.back().second;
        ++N;
        angle += scan.angle_increment;
    }
    if (N < 5) 
    {
        return 1.0;
    }
    mx /= N;
    my /= N;
    double sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < pts.size(); ++i) {
        double dx = pts[i].first - mx, dy = pts[i].second - my;
        sxx += dx*dx;
        sxy += dx*dy;
        syy += dy*dy;
    }
    double sum_of_diagonal = sxx + syy;
    double det = sxx*syy - sxy*sxy;
    double l1  = (sum_of_diagonal + sqrt(max(0.0, sum_of_diagonal*sum_of_diagonal - 4.0*det))) / 2.0;
    double l2  = (sum_of_diagonal - sqrt(max(0.0, sum_of_diagonal*sum_of_diagonal - 4.0*det))) / 2.0;
    return (l1 < 1e-9) ? 0.0 : l2 / l1;
}

class Pipeline : public rclcpp::Node
{
public:
    Pipeline()
    : Node("graph_pipeline")
    {
        min_dist_m_ = declare_parameter("min_dist_m", 0.3);
        min_angle_deg_ = declare_parameter("min_angle_deg", 30.0);
        fallback_odom_ = declare_parameter("fallback_to_odom_on_icp_fail", true);
        corridor_degen_thr_ = declare_parameter("corridor_degeneracy_threshold", 0.08);

        min_angle_rad_ = min_angle_deg_ * M_PI / 180.0;

        cb_group_sub_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_add_node_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_add_edge_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_scan_match_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = cb_group_sub_;

        sub_odom_ = create_subscription<nav_msgs::msg::Odometry>("/odom", 50,
            [this]( nav_msgs::msg::Odometry::SharedPtr msg) {
                odomCallback(msg);
            }, sub_opts);
        sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>("limo/scan", 10,
            [this]( sensor_msgs::msg::LaserScan::SharedPtr msg) {
                scanCallback(msg);
            }, sub_opts);

        clt_add_node_ = create_client<graph_slam::srv::AddNode>  ("graph_slam/add_node",   rmw_qos_profile_services_default, cb_group_add_node_);
        clt_add_edge_ = create_client<graph_slam::srv::AddEdge>  ("graph_slam/add_edge",   rmw_qos_profile_services_default, cb_group_add_edge_);
        clt_scan_match_ = create_client<graph_slam::srv::ScanMatch>("graph_slam/scan_match", rmw_qos_profile_services_default, cb_group_scan_match_);

        // watchdog_timer_ = create_wall_timer(
        //     std::chrono::seconds(30),
        //     [this]() {
        //         if (keyframe_in_flight_.load()) {
        //             RCLCPP_WARN(get_logger(), "[Pipeline] watchdog: resetting stale keyframe_in_flight_");
        //             keyframe_in_flight_.store(false);
        //         }
        //     }, cb_group_sub_);

        RCLCPP_INFO(this->get_logger(), "[Pipeline] min_dist=%.2fm min_angle=%.1fdeg fallback=%d corridor_thr=%.2f",
                 min_dist_m_, min_angle_deg_, fallback_odom_, corridor_degen_thr_);
    }

private:
    atomic<bool> initialized_{false};
    atomic<bool> initializing_{false};
    int    last_node_id_ = -1;
    double current_x_ = 0, current_y_ = 0, current_theta_ = 0;
    double last_x_    = 0, last_y_    = 0, last_theta_    = 0;

    sensor_msgs::msg::LaserScan last_scan_, current_scan_;
    bool has_scan_ = false;

    double min_dist_m_, min_angle_deg_, min_angle_rad_;
    bool fallback_odom_;
    double corridor_degen_thr_;
    double odom_cov_x_ = 0.0001, odom_cov_y_ = 0.0001, odom_cov_th_ = 0.001;

    rclcpp::CallbackGroup::SharedPtr cb_group_sub_;
    rclcpp::CallbackGroup::SharedPtr cb_group_add_node_;
    rclcpp::CallbackGroup::SharedPtr cb_group_add_edge_;
    rclcpp::CallbackGroup::SharedPtr cb_group_scan_match_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
    rclcpp::Client<graph_slam::srv::AddNode>::SharedPtr          clt_add_node_;
    rclcpp::Client<graph_slam::srv::AddEdge>::SharedPtr          clt_add_edge_;
    rclcpp::Client<graph_slam::srv::ScanMatch>::SharedPtr        clt_scan_match_;
    rclcpp::TimerBase::SharedPtr                                  watchdog_timer_;

    void scanCallback(sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        current_scan_ = *msg;
        has_scan_ = true;
    }

    void odomCallback(nav_msgs::msg::Odometry::SharedPtr msg)
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

        if (!initialized_) {
            tryInitialize();
            return;
        }

        double dx     = current_x_ - last_x_, dy = current_y_ - last_y_;
        double dist   = sqrt(dx*dx + dy*dy);
        double dtheta = fabs(wrapAngle(current_theta_ - last_theta_));

        if (dist >= min_dist_m_ || dtheta >= min_angle_rad_)
            createKeyframe();
    }

    void tryInitialize()
    {
        if (!has_scan_) 
        {
            return;
        }
        bool expected = false;
        if (!initializing_.compare_exchange_strong(expected, true)) 
        {
            return;
        }

        double init_x  = current_x_;
        double init_y  = current_y_;
        double init_th = current_theta_;
        sensor_msgs::msg::LaserScan init_scan = current_scan_;

        auto req       = std::make_shared<graph_slam::srv::AddNode::Request>();
        req->x         = init_x;
        req->y         = init_y;
        req->theta     = init_th;
        req->scan      = init_scan;
        req->timestamp = this->get_clock()->now().seconds();

        clt_add_node_->async_send_request(req,
            [this, init_x, init_y, init_th, init_scan]
            (rclcpp::Client<graph_slam::srv::AddNode>::SharedFuture f) {
            auto result = f.get();
            if (result && result->success) {
                last_node_id_ = result->id;
                last_x_       = init_x;
                last_y_       = init_y;
                last_theta_   = init_th;
                last_scan_    = init_scan;
                initialized_.store(true);
                RCLCPP_INFO(this->get_logger(), "[Pipeline] First node %d at (%.2f,%.2f,%.1fdeg)",
                         last_node_id_, init_x, init_y, init_th*180/M_PI);
            } else {
                RCLCPP_ERROR(this->get_logger(), "[Pipeline] Failed to add first node");
                initializing_.store(false);
            }
        });
    }

    void createKeyframe()
    {
        if (!has_scan_) 
        {
            return;
        }
        double kf_x = current_x_;
        double kf_y = current_y_;
        double kf_th = current_theta_;
        sensor_msgs::msg::LaserScan kf_scan = current_scan_;
        double prev_x = last_x_;
        double prev_y = last_y_;
        double prev_th = last_theta_;
        sensor_msgs::msg::LaserScan prev_scan = last_scan_;
        int    prev_id  = last_node_id_;
        double cov_x = odom_cov_x_;
        double cov_y = odom_cov_y_;
        double cov_th = odom_cov_th_;

        last_x_ = kf_x; 
        last_y_ = kf_y; 
        last_theta_ = kf_th;

        auto node_req      = std::make_shared<graph_slam::srv::AddNode::Request>();
        node_req->x         = kf_x;
        node_req->y         = kf_y;
        node_req->theta     = kf_th;
        node_req->scan      = kf_scan;
        node_req->timestamp = this->get_clock()->now().seconds();

        clt_add_node_->async_send_request(node_req,
            [this, kf_x, kf_y, kf_th, kf_scan, prev_x, prev_y, prev_th, prev_scan, prev_id, cov_x, cov_y, cov_th]
            (rclcpp::Client<graph_slam::srv::AddNode>::SharedFuture f) {
            auto node_res = f.get();
            if (!node_res || !node_res->success) {
                RCLCPP_ERROR(this->get_logger(), "[Pipeline] add_node failed");
                last_x_ = prev_x; 
                last_y_ = prev_y; 
                last_theta_ = prev_th;
                return;
            }
            int new_id = node_res->id;

            double odom_dx  = kf_x - prev_x;
            double odom_dy  = kf_y - prev_y;
            double odom_dth = wrapAngle(kf_th - prev_th);

            double c = cos(prev_th), s = sin(prev_th);
            double edge_dx  =  c*odom_dx + s*odom_dy;
            double edge_dy  = -s*odom_dx + c*odom_dy;
            double edge_dth = odom_dth;

            
                auto sm_req            = std::make_shared<graph_slam::srv::ScanMatch::Request>();
                sm_req->reference_scan = prev_scan;
                sm_req->current_scan   = kf_scan;
                sm_req->init_dx        = edge_dx;
                sm_req->init_dy        = edge_dy;
                sm_req->init_dtheta    = edge_dth;

                clt_scan_match_->async_send_request(sm_req,
                    [this, new_id, prev_id, odom_dx, odom_dy, odom_dth, edge_dx, edge_dy, edge_dth, kf_scan, cov_x, cov_y, cov_th]
                    (rclcpp::Client<graph_slam::srv::ScanMatch>::SharedFuture f2) {
                    auto sm_res = f2.get();
                    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
                    bool icp_ok = false;
                    double final_dx = edge_dx, final_dy = edge_dy, final_dth = edge_dth;

                    if (sm_res && sm_res->success) {
                        double degen = scanDegeneracy(kf_scan);

                        if (degen < corridor_degen_thr_) {
                            final_dth = sm_res->dtheta;
                            information(0,0) = 1.0 / cov_x;
                            information(1,1) = 1.0 / cov_y;
                            information(2,2) = 1.0 / max(sm_res->score, 1e-6);
                        } else {
                            final_dx  = sm_res->dx;
                            final_dy  = sm_res->dy;
                            final_dth = sm_res->dtheta;
                            for (int r = 0; r < 3; ++r)
                                for (int c2 = 0; c2 < 3; ++c2)
                                    information(r,c2) = sm_res->information[r*3+c2];
                        }
                        icp_ok = true;
                    } else {
                        RCLCPP_WARN(this->get_logger(), "[Pipeline] ICP failed, using odometry");
                        information(0,0) = 1.0 / cov_x;
                        information(1,1) = 1.0 / cov_y;
                        information(2,2) = 1.0 / cov_th;
                    }

                    sendEdge(prev_id, new_id, final_dx, final_dy, final_dth, information, odom_dx, odom_dy, odom_dth, icp_ok, kf_scan);
                });
           
        });
    }

    void sendEdge(int from_id, int to_id, double dx, double dy, double dth,
                   Eigen::Matrix3d& information,
                  double odom_dx, double odom_dy, double odom_dth, bool icp_ok,
                const sensor_msgs::msg::LaserScan& kf_scan)
    {
        auto edge_req      = std::make_shared<graph_slam::srv::AddEdge::Request>();
        edge_req->from_id  = from_id;
        edge_req->to_id    = to_id;
        edge_req->type     = Edge::ODOMETRY;
        edge_req->dx       = dx;
        edge_req->dy       = dy;
        edge_req->dtheta   = dth;
        for (int r = 0; r < 3; ++r)
            for (int c2 = 0; c2 < 3; ++c2)
                edge_req->information.push_back(information(r,c2));

        clt_add_edge_->async_send_request(edge_req,
            [this, to_id, from_id, odom_dx, odom_dy, odom_dth, icp_ok, kf_scan]
            (rclcpp::Client<graph_slam::srv::AddEdge>::SharedFuture f) {
            auto edge_res = f.get();
            if (!edge_res || !edge_res->success) {
                RCLCPP_ERROR(this->get_logger(), "[Pipeline] add_edge failed");
                return;
            }
            last_node_id_ = to_id;
            last_scan_    = kf_scan;
            RCLCPP_INFO(this->get_logger(), "[Pipeline] Node %d<-%d odom(%.2f,%.2f,%.1fdeg) icp=%d",
                     to_id, from_id, odom_dx, odom_dy, odom_dth*180/M_PI, icp_ok);
        });
    }
};

}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = make_shared<graph_slam::Pipeline>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
