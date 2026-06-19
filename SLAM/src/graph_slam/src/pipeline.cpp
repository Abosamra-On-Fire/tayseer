#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "graph_slam/srv/add_node.hpp"
#include "graph_slam/srv/add_edge.hpp"
#include "graph_slam/srv/scan_match.hpp"

#include "types.hpp"
#include <cmath>

using namespace std;
namespace graph_slam {

double scanDegeneracy(sensor_msgs::msg::LaserScan& scan)
{
    double mx = 0, my = 0; int N = 0;
    vector<pair<double,double>> pts;
    double angle = scan.angle_min;
    for (size_t i = 0; i < scan.ranges.size(); ++i ) {
        float r = scan.ranges[i];
        if (!isfinite(r) || r <= scan.range_min || r >= scan.range_max) 
        {
             angle += scan.angle_increment;
            continue;
           
        }
        pts.push_back({r*cos(angle), r*sin(angle)});//in robot frame we bageb el cartesian coordinates 
        mx += pts.back().first; 
        my += pts.back().second; 
        ++N;
        angle += scan.angle_increment;
    }
    if (N < 5) 
        return 1.0;
    mx /= N; 
    my /= N;
    double sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < pts.size(); ++i) {
        double dx = pts[i].first - mx, dy = pts[i].second - my;
        sxx += dx*dx; 
        sxy += dx*dy; 
        syy += dy*dy;
    }
    double sum_of_digonal = sxx + syy;
    double det   = sxx*syy - sxy*sxy;
    double l1    = (sum_of_digonal + sqrt(max(0.0,sum_of_digonal*sum_of_digonal - 4.0*det))) / 2.0;
    double l2    = (sum_of_digonal - sqrt(max(0.0,sum_of_digonal*sum_of_digonal - 4.0*det))) / 2.0;
    return (l1 < 1e-9) ? 0.0 : l2 / l1;
}

class Pipeline : public rclcpp::Node
{
public:
    Pipeline()
    : Node("graph_pipeline"), initialized_(false), last_node_id_(-1)
    {
        min_dist_m_         = this->declare_parameter("min_dist_m",0.3);
        min_angle_deg_      = this->declare_parameter("min_angle_deg",30.0);
        use_icp_odom_       = this->declare_parameter("use_scan_matching_from_odom",true);
        fallback_odom_      = this->declare_parameter("fallback_to_odom_on_icp_fail",true);
        corridor_degen_thr_ = this->declare_parameter("corridor_degeneracy_threshold", 0.08);

        min_angle_rad_ = min_angle_deg_ * M_PI / 180.0;

        cb_group_reentrant_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        cb_group_add_node_  = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_add_edge_  = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_scan_match_= this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = cb_group_reentrant_;

        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>("/odom", 50,
              bind(&Pipeline::odomCallback, this,   placeholders::_1), sub_opts);
        sub_scan_ = this->create_subscription<sensor_msgs::msg::LaserScan>("limo/scan", 10,
              bind(&Pipeline::scanCallback, this,   placeholders::_1), sub_opts);

        clt_add_node_   = this->create_client<graph_slam::srv::AddNode>  ("graph_slam/add_node",   rmw_qos_profile_services_default, cb_group_add_node_);
        clt_add_edge_   = this->create_client<graph_slam::srv::AddEdge>  ("graph_slam/add_edge",   rmw_qos_profile_services_default, cb_group_add_edge_);
        clt_scan_match_ = this->create_client<graph_slam::srv::ScanMatch>("graph_slam/scan_match", rmw_qos_profile_services_default, cb_group_scan_match_);
        clt_add_node_->wait_for_service();
        clt_add_edge_->wait_for_service();
        clt_scan_match_->wait_for_service();

        RCLCPP_INFO(this->get_logger(), "[Pipeline] min_dist=%.2fm min_angle=%.1fdeg icp=%d fallback=%d corridor_thr=%.2f",
                 min_dist_m_, min_angle_deg_, use_icp_odom_, fallback_odom_, corridor_degen_thr_);
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
    double odom_cov_x_ = 0.0001, odom_cov_y_ = 0.0001, odom_cov_th_ = 0.001;

    rclcpp::CallbackGroup::SharedPtr cb_group_reentrant_;
    rclcpp::CallbackGroup::SharedPtr cb_group_add_node_;
    rclcpp::CallbackGroup::SharedPtr cb_group_add_edge_;
    rclcpp::CallbackGroup::SharedPtr cb_group_scan_match_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
    rclcpp::Client<graph_slam::srv::AddNode>::SharedPtr          clt_add_node_;
    rclcpp::Client<graph_slam::srv::AddEdge>::SharedPtr          clt_add_edge_;
    rclcpp::Client<graph_slam::srv::ScanMatch>::SharedPtr        clt_scan_match_;

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

        current_x_     = msg->pose.pose.position.x;//odometry frame 
        current_y_     = msg->pose.pose.position.y;
        current_theta_ = yaw;//rotation around z

        double cx = msg->pose.covariance[0];
        double cy = msg->pose.covariance[7];
        double ct = msg->pose.covariance[35];
        odom_cov_x_  = (cx  > 1e-9) ? cx  : 0.0001;
        odom_cov_y_  = (cy  > 1e-9) ? cy  : 0.0001;
        odom_cov_th_ = (ct  > 1e-9) ? ct  : 0.001;

        if (!initialized_) 
            { 
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
        auto req =   make_shared<graph_slam::srv::AddNode::Request>();
        req->x         = current_x_;
        req->y         = current_y_;
        req->theta     = current_theta_;
        req->scan      = current_scan_;
        req->timestamp = this->get_clock()->now().seconds();

        auto future = clt_add_node_->async_send_request(req);
        auto status = future.wait_for(  chrono::seconds(5));
        if (status !=   future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "[Pipeline] Failed to add first node");
            return;
        }
        auto result = future.get();

        if (result->success) {
            last_node_id_ = result->id;
            last_x_ = current_x_; 
            last_y_ = current_y_; 
            last_theta_ = current_theta_;
            last_scan_ = current_scan_;
            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "[Pipeline] First node %d at (%.2f,%.2f,%.1fdeg)",
                     last_node_id_, current_x_, current_y_, current_theta_*180/M_PI);
        } 
        else 
        {
            RCLCPP_ERROR(this->get_logger(), "[Pipeline] Failed to add first node");
        }
    }

    void createKeyframe()
    {
        if (!has_scan_) 
        {
            return;
        }

        auto node_req =   make_shared<graph_slam::srv::AddNode::Request>();
        node_req->x         = current_x_;
        node_req->y         = current_y_;
        node_req->theta     = current_theta_;
        node_req->scan      = current_scan_;
        node_req->timestamp = this->get_clock()->now().seconds();

        auto node_future = clt_add_node_->async_send_request(node_req);
        auto node_status = node_future.wait_for(chrono::seconds(5));
        if (node_status !=   future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "[Pipeline] add_node failed"); 
            return;
        }
        auto node_res = node_future.get();
        if (!node_res->success) {
            RCLCPP_ERROR(this->get_logger(), "[Pipeline] add_node failed"); 
            return;
        }
        int new_id = node_res->id;
        double odom_dx  = current_x_ - last_x_;
        double odom_dy  = current_y_ - last_y_;
        double odom_dth = wrapAngle(current_theta_ - last_theta_);
        //p_world=R(theta)*p_robot+T 34an at7awel men el robot frame lel world frame
        //hena ana 3awez at7awel men el odometry frame lel keyframe (ezay el last key frame 4ayef el odometry frame)
        //fa ba3mel el inverse lel rotation matrix
        double c = cos(last_theta_), s = sin(last_theta_);
        double kf_dx  =  c*odom_dx + s*odom_dy;
        double kf_dy  = -s*odom_dx + c*odom_dy;
        double kf_dth = odom_dth;

        Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
        bool icp_ok = false;

        if (use_icp_odom_) {
            auto sm_req =   make_shared<graph_slam::srv::ScanMatch::Request>();
            sm_req->reference_scan = last_scan_;
            sm_req->current_scan   = current_scan_;
            sm_req->init_dx        = kf_dx;
            sm_req->init_dy        = kf_dy;
            sm_req->init_dtheta    = kf_dth;

            auto sm_future = clt_scan_match_->async_send_request(sm_req);
            auto sm_status = sm_future.wait_for(  chrono::seconds(5));
              shared_ptr<graph_slam::srv::ScanMatch::Response> sm_res;
            if (sm_status ==   future_status::ready);
                sm_res = sm_future.get();

            if (sm_status ==   future_status::ready&& sm_res->success) {
                double degen       = scanDegeneracy(current_scan_);
                bool   in_corridor = (degen < corridor_degen_thr_);

                if (in_corridor) {
                    kf_dth = sm_res->dtheta;
                    information(0,0) = 1.0 / odom_cov_x_;
                    information(1,1) = 1.0 / odom_cov_y_;
                    information(2,2) = 1.0 / max(sm_res->score, 1e-6);
                    RCLCPP_DEBUG(this->get_logger(), "[Pipeline] Corridor: odom trans + ICP rot, degen=%.3f", degen);
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
                if (!fallback_odom_) 
                { RCLCPP_WARN(this->get_logger(), "[Pipeline] ICP failed, skipping"); 
                  return; 
                }
                RCLCPP_WARN(this->get_logger(), "[Pipeline] ICP failed, using odometry");
                information(0,0) = 1.0 / odom_cov_x_;
                information(1,1) = 1.0 / odom_cov_y_;
                information(2,2) = 1.0 / odom_cov_th_;
            }
        } else {
            information(0,0) = 1.0 / odom_cov_x_;
            information(1,1) = 1.0 / odom_cov_y_;
            information(2,2) = 1.0 / odom_cov_th_;
        }

        auto edge_req =   make_shared<graph_slam::srv::AddEdge::Request>();
        edge_req->from_id  = last_node_id_;
        edge_req->to_id    = new_id;
        edge_req->type     = Edge::ODOMETRY;
        edge_req->dx       = kf_dx;
        edge_req->dy       = kf_dy;
        edge_req->dtheta   = kf_dth;
        for (int r = 0; r < 3; ++r)
            for (int c2 = 0; c2 < 3; ++c2)
                edge_req->information.push_back(information(r,c2));

        auto edge_future = clt_add_edge_->async_send_request(edge_req);
        auto edge_status = edge_future.wait_for(  chrono::seconds(5));
        if (edge_status !=   future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "[Pipeline] add_edge failed"); return;
        }
        auto edge_res = edge_future.get();
        if (!edge_res->success) {
            RCLCPP_ERROR(this->get_logger(), "[Pipeline] add_edge failed"); return;
        }

        last_node_id_ = new_id;
        last_x_ = current_x_; last_y_ = current_y_; last_theta_ = current_theta_;
        last_scan_ = current_scan_;
        RCLCPP_INFO(this->get_logger(), "[Pipeline] Node %d<-%d odom(%.2f,%.2f,%.1fdeg) icp=%d",
                 new_id, edge_req->from_id, odom_dx, odom_dy, odom_dth*180/M_PI, icp_ok);
    }
};

}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node =   make_shared<graph_slam::Pipeline>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}