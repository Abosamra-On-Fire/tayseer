#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "graph_slam/srv/add_node.hpp"
#include "graph_slam/srv/add_edge.hpp"
#include "graph_slam/srv/scan_match.hpp"

#include "types.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace std;
namespace graph_slam {

double scanDegeneracy(const sensor_msgs::msg::LaserScan& scan)
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

class Pipeline : public rclcpp::Node
{
public:
    explicit Pipeline()
    : Node("graph_pipeline"), initialized_(false), last_node_id_(-1)
    {
        min_dist_m_          = declare_parameter<double>("min_dist_m",                   0.3);
        min_angle_deg_       = declare_parameter<double>("min_angle_deg",                30.0);
        use_icp_odom_        = declare_parameter<bool>  ("use_scan_matching_from_odom",  true);
        fallback_odom_       = declare_parameter<bool>  ("fallback_to_odom_on_icp_fail", true);
        corridor_degen_thr_  = declare_parameter<double>("corridor_degeneracy_thr",      0.05);
        icp_score_threshold_ = declare_parameter<double>("icp_score_threshold",          0.05);

        min_angle_rad_ = min_angle_deg_ * M_PI / 180.0;

        sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 50,
            [this](nav_msgs::msg::Odometry::SharedPtr msg){ odomCallback(msg); });

        sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "limo/scan", 10,
            [this](sensor_msgs::msg::LaserScan::SharedPtr msg){ scanCallback(msg); });

        clt_add_node_   = create_client<graph_slam::srv::AddNode>  ("graph_slam/add_node");
        clt_add_edge_   = create_client<graph_slam::srv::AddEdge>  ("graph_slam/add_edge");
        clt_scan_match_ = create_client<graph_slam::srv::ScanMatch>("graph_slam/scan_match");

        clt_add_node_->wait_for_service();
        clt_add_edge_->wait_for_service();
        clt_scan_match_->wait_for_service();

        RCLCPP_INFO(get_logger(),
            "[Pipeline] min_dist=%.2fm min_angle=%.1fdeg icp=%d fallback=%d corridor_thr=%.2f icp_score_thr=%.3f",
            min_dist_m_, min_angle_deg_, use_icp_odom_, fallback_odom_,
            corridor_degen_thr_, icp_score_threshold_);
    }

private:
    bool   initialized_;
    int    last_node_id_;
    double current_x_ = 0, current_y_ = 0, current_theta_ = 0;
    double last_x_    = 0, last_y_    = 0, last_theta_    = 0;

    sensor_msgs::msg::LaserScan last_scan_, current_scan_;
    bool has_scan_ = false;

    double min_dist_m_, min_angle_deg_, min_angle_rad_;
    bool   use_icp_odom_, fallback_odom_;
    double corridor_degen_thr_;
    double icp_score_threshold_;
    double odom_cov_x_ = 0.0001, odom_cov_y_ = 0.0001, odom_cov_th_ = 0.001;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr    sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
    rclcpp::Client<graph_slam::srv::AddNode>::SharedPtr          clt_add_node_;
    rclcpp::Client<graph_slam::srv::AddEdge>::SharedPtr          clt_add_edge_;
    rclcpp::Client<graph_slam::srv::ScanMatch>::SharedPtr        clt_scan_match_;

    template<typename ClientT, typename RequestT>
    typename ClientT::element_type::Response::SharedPtr callSync(
        ClientT& client, typename RequestT::SharedPtr req)
    {
        auto future = client->async_send_request(req);
        if (rclcpp::spin_until_future_complete(
                this->get_node_base_interface(), future)
            != rclcpp::FutureReturnCode::SUCCESS)
            return nullptr;
        return future.get();
    }

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
        odom_cov_x_  = (cx > 1e-9) ? cx : 0.0001;
        odom_cov_y_  = (cy > 1e-9) ? cy : 0.0001;
        odom_cov_th_ = (ct > 1e-9) ? ct : 0.001;

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

        auto req = std::make_shared<graph_slam::srv::AddNode::Request>();
        req->x         = current_x_;
        req->y         = current_y_;
        req->theta     = current_theta_;
        req->scan      = current_scan_;
        req->timestamp = this->get_clock()->now().seconds();

        auto res = callSync<decltype(clt_add_node_),
                            graph_slam::srv::AddNode::Request>(clt_add_node_, req);

        if (res && res->success) {
            last_node_id_ = res->id;
            last_x_ = current_x_; last_y_ = current_y_; last_theta_ = current_theta_;
            last_scan_ = current_scan_;
            initialized_ = true;
            RCLCPP_INFO(get_logger(), "[Pipeline] First node %d at (%.2f,%.2f,%.1fdeg)",
                        last_node_id_, current_x_, current_y_, current_theta_*180/M_PI);
        } else {
            RCLCPP_ERROR(get_logger(), "[Pipeline] Failed to add first node");
        }
    }

    void createKeyframe()
    {
        if (!has_scan_) return;

        auto node_req = std::make_shared<graph_slam::srv::AddNode::Request>();
        node_req->x         = current_x_;
        node_req->y         = current_y_;
        node_req->theta     = current_theta_;
        node_req->scan      = current_scan_;
        node_req->timestamp = this->get_clock()->now().seconds();

        auto node_res = callSync<decltype(clt_add_node_),
                                 graph_slam::srv::AddNode::Request>(clt_add_node_, node_req);

        if (!node_res || !node_res->success) {
            RCLCPP_ERROR(get_logger(), "[Pipeline] add_node failed"); return;
        }
        int new_id = node_res->id;

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
            auto sm_req = std::make_shared<graph_slam::srv::ScanMatch::Request>();
            sm_req->reference_scan = last_scan_;
            sm_req->current_scan   = current_scan_;
            sm_req->init_dx        = kf_dx;
            sm_req->init_dy        = kf_dy;
            sm_req->init_dtheta    = kf_dth;

            auto sm_res = callSync<decltype(clt_scan_match_),
                                   graph_slam::srv::ScanMatch::Request>(clt_scan_match_, sm_req);

            if (sm_res && sm_res->success && sm_res->score <= icp_score_threshold_) {
                double degen       = scanDegeneracy(current_scan_);
                bool   in_corridor = (degen < corridor_degen_thr_);

                if (in_corridor) {
                    kf_dx  = sm_res->dx;
                    kf_dth = sm_res->dtheta;
                    information(0,0) = 1.0 / max(odom_cov_x_, 1e-6);
                    information(1,1) = 1.0 / max(odom_cov_y_, 1e-6);
                    information(2,2) = min(1.0 / max(sm_res->score, 1e-6) * 0.3, 300.0);
                    RCLCPP_DEBUG(get_logger(),
                        "[Pipeline] Corridor: ICP dx+rot, odom dy, degen=%.3f", degen);
                } else {
                    kf_dx  = sm_res->dx;
                    kf_dy  = sm_res->dy;
                    kf_dth = sm_res->dtheta;
                    for (int r = 0; r < 3; ++r)
                        for (int c2 = 0; c2 < 3; ++c2)
                            information(r,c2) = sm_res->information[r*3+c2];
                }
                icp_ok = true;
            } else {
                if (!fallback_odom_) {
                    RCLCPP_WARN(get_logger(), "[Pipeline] ICP failed/poor, skipping"); return;
                }
                RCLCPP_WARN(get_logger(), "[Pipeline] ICP failed or poor score, using odometry");
                information(0,0) = 1.0 / max(odom_cov_x_, 1e-6);
                information(1,1) = 1.0 / max(odom_cov_y_, 1e-6);
                information(2,2) = 1.0 / max(odom_cov_th_, 1e-6);
            }
        } else {
            information(0,0) = 1.0 / max(odom_cov_x_, 1e-6);
            information(1,1) = 1.0 / max(odom_cov_y_, 1e-6);
            information(2,2) = 1.0 / max(odom_cov_th_, 1e-6);
        }

        auto edge_req = std::make_shared<graph_slam::srv::AddEdge::Request>();
        edge_req->from_id = last_node_id_;
        edge_req->to_id   = new_id;
        edge_req->type    = Edge::ODOMETRY;
        edge_req->dx      = kf_dx;
        edge_req->dy      = kf_dy;
        edge_req->dtheta  = kf_dth;
        for (int r = 0; r < 3; ++r)
            for (int c2 = 0; c2 < 3; ++c2)
                edge_req->information.push_back(information(r,c2));

        auto edge_res = callSync<decltype(clt_add_edge_),
                                 graph_slam::srv::AddEdge::Request>(clt_add_edge_, edge_req);

        if (!edge_res || !edge_res->success) {
            RCLCPP_ERROR(get_logger(), "[Pipeline] add_edge failed"); return;
        }

        last_node_id_ = new_id;
        last_x_ = current_x_; last_y_ = current_y_; last_theta_ = current_theta_;
        last_scan_ = current_scan_;
        RCLCPP_INFO(get_logger(),
            "[Pipeline] Node %d<-%d odom(%.2f,%.2f,%.1fdeg) icp=%d",
            new_id, edge_req->from_id, odom_dx, odom_dy, odom_dth*180/M_PI, icp_ok);
    }
};

} // namespace graph_slam

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<graph_slam::Pipeline>());
    rclcpp::shutdown();
    return 0;
}