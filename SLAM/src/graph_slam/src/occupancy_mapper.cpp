#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "graph_slam/srv/get_graph.hpp"
#include "types.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>

using namespace std;
namespace graph_slam {

const double LOGODD_HIT  =  0.85;
const double LOGODD_MISS = -0.30;
const double LOGODD_MIN  = -5.0;
const double LOGODD_MAX  =  5.0;

struct Particle {
    double x, y, theta, weight;
};

class OccupancyMapper : public rclcpp::Node
{
public:
    explicit OccupancyMapper()
    : Node("occupancy_mapper"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_, this),
      rng_(std::random_device{}())
    {
        map_resolution_       = declare_parameter<double>("map_resolution",       0.05);
        map_width_m_          = declare_parameter<double>("map_width_m",          40.0);
        map_height_m_         = declare_parameter<double>("map_height_m",         40.0);
        publish_period_sec_   = declare_parameter<double>("publish_period_sec",   0.05);
        map_angle_step_deg_   = declare_parameter<double>("map_angle_step_deg",   0.5);
        fov_half_deg_         = declare_parameter<double>("fov_half_deg",         60.0);
        localization_enabled_ = declare_parameter<bool>  ("localization_enabled", true);
        num_particles_        = declare_parameter<int>   ("num_particles",        500);
        particle_sigma_x_     = declare_parameter<double>("particle_sigma_x",     0.1);
        particle_sigma_y_     = declare_parameter<double>("particle_sigma_y",     0.1);
        particle_sigma_theta_ = declare_parameter<double>("particle_sigma_theta", 0.05);
        motion_sigma_x_       = declare_parameter<double>("motion_sigma_x",       0.02);
        motion_sigma_y_       = declare_parameter<double>("motion_sigma_y",       0.02);
        motion_sigma_theta_   = declare_parameter<double>("motion_sigma_theta",   0.01);
        resample_threshold_   = declare_parameter<double>("resample_threshold",   0.5);
        scan_subsample_       = declare_parameter<int>   ("scan_subsample",       8);
        odom_frame_           = declare_parameter<string>("odom_frame",           string("odom"));
        base_frame_           = declare_parameter<string>("base_frame",           string("base_link"));

        width_cells_  = static_cast<int>(map_width_m_  / map_resolution_);
        height_cells_ = static_cast<int>(map_height_m_ / map_resolution_);
        origin_x_ = -map_width_m_  / 2.0;
        origin_y_ = -map_height_m_ / 2.0;

        log_odds_.assign(width_cells_ * height_cells_, 0.0);

        localization_initialized_ = false;
        last_odom_x_ = 0.0; last_odom_y_ = 0.0; last_odom_theta_ = 0.0;
        map_odom_x_  = 0.0; map_odom_y_  = 0.0; map_odom_theta_  = 0.0;
        odom_received_ = false;
        map_ready_     = false;
        busy_          = false;

        auto latched = rclcpp::QoS(1).transient_local();
        pub_map_            = create_publisher<nav_msgs::msg::OccupancyGrid>("/map", latched);
        pub_particle_cloud_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/particle_cloud_debug", 1);

        sub_trigger_ = create_subscription<std_msgs::msg::Bool>(
            "/graph_slam/trigger_remap", 1,
            [this](std_msgs::msg::Bool::SharedPtr msg){ triggerCallback(msg); });

        if (localization_enabled_) {
            sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
                "/scan", 10,
                [this](sensor_msgs::msg::LaserScan::SharedPtr msg){ scanCallback(msg); });
            sub_initial_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                "/initialpose", 1,
                [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg){ initialPoseCallback(msg); });
        }

        clt_get_graph_ = create_client<graph_slam::srv::GetGraph>("/graph_slam/get_graph");
        clt_get_graph_->wait_for_service();

        timer_ = create_wall_timer(
            std::chrono::duration<double>(publish_period_sec_),
            [this](){ timerCallback(); });

        RCLCPP_INFO(get_logger(),
            "[OccupancyMapper] %.0fx%.0f m @ %.3f m/cell = %dx%d cells  fov=+/-%.0fdeg  localization=%s",
            map_width_m_, map_height_m_, map_resolution_,
            width_cells_, height_cells_, fov_half_deg_,
            localization_enabled_ ? "ON" : "OFF");
    }

private:
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr                           sub_trigger_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr                   sub_scan_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_initial_pose_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr                     pub_map_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr                     pub_particle_cloud_;
    rclcpp::Client<graph_slam::srv::GetGraph>::SharedPtr                           clt_get_graph_;
    rclcpp::TimerBase::SharedPtr                                                   timer_;
    tf2_ros::TransformBroadcaster                                                  tf_broadcaster_{this};
    tf2_ros::Buffer                                                                tf_buffer_;
    tf2_ros::TransformListener                                                     tf_listener_;
    std::mt19937                                                                   rng_;

    double map_resolution_;
    double map_width_m_, map_height_m_;
    int    width_cells_, height_cells_;
    double origin_x_, origin_y_;
    vector<double> log_odds_;
    double publish_period_sec_;
    double map_angle_step_deg_;
    double fov_half_deg_;

    bool   localization_enabled_;
    int    num_particles_;
    double particle_sigma_x_, particle_sigma_y_, particle_sigma_theta_;
    double motion_sigma_x_,   motion_sigma_y_,   motion_sigma_theta_;
    double resample_threshold_;
    int    scan_subsample_;
    string odom_frame_, base_frame_;

    vector<Particle> particles_;
    bool   localization_initialized_;
    double last_odom_x_, last_odom_y_, last_odom_theta_;
    double map_odom_x_,  map_odom_y_,  map_odom_theta_;
    bool   odom_received_;
    bool   map_ready_;
    bool   busy_;

    int last_mapped_node_ = -1;

    void triggerCallback(std_msgs::msg::Bool::SharedPtr msg)
    {
        if (msg->data) {
            RCLCPP_INFO(get_logger(), "[OccupancyMapper] Full rebuild triggered");
            fullRebuild();
        }
    }

    void timerCallback()
    {
        incrementalUpdate();
        broadcastMapTf();
    }

    void initialPoseCallback(geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        double ix = msg->pose.pose.position.x;
        double iy = msg->pose.pose.position.y;
        tf2::Quaternion q;
        tf2::fromMsg(msg->pose.pose.orientation, q);
        double itheta = atan2(2.0*(q.w()*q.z()+q.x()*q.y()),
                              1.0-2.0*(q.y()*q.y()+q.z()*q.z()));
        initParticles(ix, iy, itheta);
        RCLCPP_INFO(get_logger(),
            "[OccupancyMapper] Particles initialized from /initialpose (%.2f,%.2f,%.3f)",
            ix, iy, itheta);
    }

    void scanCallback(sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        if (!map_ready_) return;

        geometry_msgs::msg::TransformStamped odom_tf;
        try {
            odom_tf = tf_buffer_.lookupTransform(odom_frame_, base_frame_,
                                                 tf2::TimePointZero,
                                                 tf2::durationFromSec(0.05));
        } catch (tf2::TransformException& ex) {
            return;
        }

        double ox = odom_tf.transform.translation.x;
        double oy = odom_tf.transform.translation.y;
        tf2::Quaternion oq;
        tf2::fromMsg(odom_tf.transform.rotation, oq);
        double otheta = atan2(2.0*(oq.w()*oq.z()+oq.x()*oq.y()),
                              1.0-2.0*(oq.y()*oq.y()+oq.z()*oq.z()));

        if (!localization_initialized_) {
            double wx = map_odom_x_ + ox*cos(map_odom_theta_) - oy*sin(map_odom_theta_);
            double wy = map_odom_y_ + ox*sin(map_odom_theta_) + oy*cos(map_odom_theta_);
            double wt = map_odom_theta_ + otheta;
            initParticles(wx, wy, wt);
            last_odom_x_ = ox; last_odom_y_ = oy; last_odom_theta_ = otheta;
            odom_received_ = true;
            localization_initialized_ = true;
            return;
        }

        double dx     = ox - last_odom_x_;
        double dy     = oy - last_odom_y_;
        double dtheta = otheta - last_odom_theta_;
        while (dtheta >  M_PI) dtheta -= 2.0*M_PI;
        while (dtheta < -M_PI) dtheta += 2.0*M_PI;
        last_odom_x_ = ox; last_odom_y_ = oy; last_odom_theta_ = otheta;

        motionUpdate(dx, dy, dtheta);
        measurementUpdate(*scan);
        double n_eff = computeNeff();
        if (n_eff < resample_threshold_ * num_particles_)
            resample();
        estimatePoseAndUpdateTf(ox, oy, otheta);
    }

    void initParticles(double cx, double cy, double ctheta)
    {
        particles_.resize(num_particles_);
        std::normal_distribution<double> nx(cx,     particle_sigma_x_);
        std::normal_distribution<double> ny(cy,     particle_sigma_y_);
        std::normal_distribution<double> nt(ctheta, particle_sigma_theta_);
        for (auto& p : particles_) {
            p.x = nx(rng_); p.y = ny(rng_); p.theta = nt(rng_);
            p.weight = 1.0 / num_particles_;
        }
    }

    void motionUpdate(double dx, double dy, double dtheta)
    {
        std::normal_distribution<double> nx(0.0, motion_sigma_x_    + fabs(dx)*0.1);
        std::normal_distribution<double> ny(0.0, motion_sigma_y_    + fabs(dy)*0.1);
        std::normal_distribution<double> nt(0.0, motion_sigma_theta_ + fabs(dtheta)*0.05);
        for (auto& p : particles_) {
            double s = sin(p.theta), c = cos(p.theta);
            p.x     += c*dx - s*dy + nx(rng_);
            p.y     += s*dx + c*dy + ny(rng_);
            p.theta += dtheta + nt(rng_);
            while (p.theta >  M_PI) p.theta -= 2.0*M_PI;
            while (p.theta < -M_PI) p.theta += 2.0*M_PI;
        }
    }

    double likelihoodFieldAt(double wx, double wy)
    {
        int cx = static_cast<int>((wx - origin_x_) / map_resolution_);
        int cy = static_cast<int>((wy - origin_y_) / map_resolution_);
        const int R = 3;
        double best = -1e9;
        for (int dy = -R; dy <= R; ++dy) {
            for (int dx = -R; dx <= R; ++dx) {
                int nx = cx+dx, ny = cy+dy;
                if (nx < 0 || nx >= width_cells_ || ny < 0 || ny >= height_cells_) continue;
                best = max(best, log_odds_[ny*width_cells_+nx]);
            }
        }
        return (best > 0.5) ? 1.0 : ((best < -0.5) ? 0.0 : 0.5);
    }

    void measurementUpdate(const sensor_msgs::msg::LaserScan& scan)
    {
        int N = static_cast<int>(scan.ranges.size());
        for (auto& p : particles_) {
            double log_w = 0.0;
            for (int i = 0; i < N; i += scan_subsample_) {
                float r = scan.ranges[i];
                if (!isfinite(r) || r < scan.range_min || r >= scan.range_max) continue;
                double angle = p.theta + scan.angle_min + i * scan.angle_increment;
                double hx = p.x + r * cos(angle);
                double hy = p.y + r * sin(angle);
                double lk = likelihoodFieldAt(hx, hy);
                lk = max(lk, 1e-6);
                log_w += log(lk);
            }
            p.weight *= exp(log_w);
        }
        double sum = 0.0;
        for (auto& p : particles_) sum += p.weight;
        if (sum < 1e-300) {
            for (auto& p : particles_) p.weight = 1.0 / num_particles_;
        } else {
            for (auto& p : particles_) p.weight /= sum;
        }
    }

    double computeNeff()
    {
        double sum_sq = 0.0;
        for (auto& p : particles_) sum_sq += p.weight * p.weight;
        return (sum_sq > 0.0) ? 1.0 / sum_sq : 0.0;
    }

    void resample()
    {
        vector<double> cdf(num_particles_);
        cdf[0] = particles_[0].weight;
        for (int i = 1; i < num_particles_; ++i)
            cdf[i] = cdf[i-1] + particles_[i].weight;

        std::uniform_real_distribution<double> ud(0.0, 1.0/num_particles_);
        double u = ud(rng_);
        int j = 0;
        vector<Particle> new_p(num_particles_);
        for (int i = 0; i < num_particles_; ++i) {
            double target = u + static_cast<double>(i) / num_particles_;
            while (j < num_particles_-1 && cdf[j] < target) ++j;
            new_p[i] = particles_[j];
            new_p[i].weight = 1.0 / num_particles_;
        }
        particles_ = new_p;
    }

    void estimatePoseAndUpdateTf(double odom_x, double odom_y, double odom_theta)
    {
        double wx = 0.0, wy = 0.0, ws = 0.0, wc = 0.0;
        for (auto& p : particles_) {
            wx += p.weight * p.x;
            wy += p.weight * p.y;
            ws += p.weight * sin(p.theta);
            wc += p.weight * cos(p.theta);
        }
        double best_theta = atan2(ws, wc);

        tf2::Transform T_map_base;
        T_map_base.setOrigin(tf2::Vector3(wx, wy, 0.0));
        tf2::Quaternion q; q.setRPY(0.0, 0.0, best_theta);
        T_map_base.setRotation(q);

        tf2::Transform T_odom_base;
        T_odom_base.setOrigin(tf2::Vector3(odom_x, odom_y, 0.0));
        tf2::Quaternion qo; qo.setRPY(0.0, 0.0, odom_theta);
        T_odom_base.setRotation(qo);

        tf2::Transform T_map_odom = T_map_base * T_odom_base.inverse();

        map_odom_x_     = T_map_odom.getOrigin().x();
        map_odom_y_     = T_map_odom.getOrigin().y();
        map_odom_theta_ = atan2(2.0*(T_map_odom.getRotation().w()*T_map_odom.getRotation().z()
                                     +T_map_odom.getRotation().x()*T_map_odom.getRotation().y()),
                                1.0-2.0*(T_map_odom.getRotation().y()*T_map_odom.getRotation().y()
                                        +T_map_odom.getRotation().z()*T_map_odom.getRotation().z()));
    }

    void fullRebuild()
    {
        if (busy_) return;
        busy_ = true;

        auto req = std::make_shared<graph_slam::srv::GetGraph::Request>();
        clt_get_graph_->async_send_request(req,
            [this](rclcpp::Client<graph_slam::srv::GetGraph>::SharedFuture future)
            {
                auto resp = *(future.get());
                int N = static_cast<int>(resp.ids.size());
                if (N == 0) {
                    RCLCPP_WARN(get_logger(), "[OccupancyMapper] Graph empty");
                    busy_ = false;
                    return;
                }

                fill(log_odds_.begin(), log_odds_.end(), 0.0);
                for (int i = 0; i < N; ++i)
                    insertScan(resp.xs[i], resp.ys[i], resp.thetas[i], resp.scans[i]);

                last_mapped_node_ = N - 1;
                map_ready_ = true;
                publishMap();
                RCLCPP_INFO(get_logger(), "[OccupancyMapper] Full rebuild: %d nodes", N);
                busy_ = false;
            });
    }

    void incrementalUpdate()
    {
        if (busy_) return;
        busy_ = true;

        auto req = std::make_shared<graph_slam::srv::GetGraph::Request>();
        clt_get_graph_->async_send_request(req,
            [this](rclcpp::Client<graph_slam::srv::GetGraph>::SharedFuture future)
            {
                auto resp = *(future.get());
                int N = static_cast<int>(resp.ids.size());
                bool changed = false;
                for (int i = last_mapped_node_ + 1; i < N; ++i) {
                    insertScan(resp.xs[i], resp.ys[i], resp.thetas[i], resp.scans[i]);
                    changed = true;
                }
                if (changed) {
                    publishMap();
                    last_mapped_node_ = N - 1;
                    map_ready_ = true;
                }
                busy_ = false;
            });
    }

    void insertScan(double x, double y, double theta,
                    const sensor_msgs::msg::LaserScan& scan)
    {
        if (scan.ranges.empty()) return;

        int rx = worldToCell(x, true);
        int ry = worldToCell(y, false);

        double step_rad = map_angle_step_deg_ * M_PI / 180.0;
        int    step     = max(1, static_cast<int>(step_rad / scan.angle_increment));
        double fov_rad  = fov_half_deg_ * M_PI / 180.0;

        int num_rays = static_cast<int>(scan.ranges.size());
        for (int i = 0; i < num_rays; i += step) {
            double angle  = scan.angle_min + i * scan.angle_increment;
            double a_norm = fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
            if (fabs(a_norm) > fov_rad) continue;

            float r   = scan.ranges[i];
            bool  hit = isfinite(r) && r > scan.range_min && r < scan.range_max;

            double global_angle = theta + angle;

            if (hit) {
                int ex = worldToCell(x + r * cos(global_angle), true);
                int ey = worldToCell(y + r * sin(global_angle), false);
                bresenham(rx, ry, ex, ey, true);
            } else {
                int ex = worldToCell(x + scan.range_max * cos(global_angle), true);
                int ey = worldToCell(y + scan.range_max * sin(global_angle), false);
                bresenham(rx, ry, ex, ey, false);
            }
        }
    }

    void bresenham(int x0, int y0, int x1, int y1, bool mark_hit)
    {
        int dx = abs(x1 - x0);
        int dy = abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int f  = dx - dy;
        int x  = x0, y = y0;

        while (true) {
            if (x < 0 || x >= width_cells_ || y < 0 || y >= height_cells_) break;
            if (x == x1 && y == y1) { updateCell(x, y, mark_hit ? LOGODD_HIT : LOGODD_MISS); break; }
            updateCell(x, y, LOGODD_MISS);
            int f2 = 2 * f;
            if (f2 > -dy) { f -= dy; x += sx; }
            if (f2 <  dx) { f += dx; y += sy; }
        }
    }

    void updateCell(int x, int y, double delta)
    {
        int idx = y * width_cells_ + x;
        log_odds_[idx] = min(max(log_odds_[idx] + delta, LOGODD_MIN), LOGODD_MAX);
    }

    int worldToCell(double coord, bool is_x)
    {
        double origin = is_x ? origin_x_ : origin_y_;
        return static_cast<int>((coord - origin) / map_resolution_);
    }

    void publishMap()
    {
        nav_msgs::msg::OccupancyGrid grid;
        grid.header.frame_id           = "map";
        grid.header.stamp              = this->get_clock()->now();
        grid.info.resolution           = map_resolution_;
        grid.info.width                = width_cells_;
        grid.info.height               = height_cells_;
        grid.info.origin.position.x    = origin_x_;
        grid.info.origin.position.y    = origin_y_;
        grid.info.origin.orientation.w = 1.0;

        grid.data.resize(width_cells_ * height_cells_);
        for (size_t i = 0; i < log_odds_.size(); ++i) {
            if      (log_odds_[i] >  0.5) grid.data[i] = 100;
            else if (log_odds_[i] < -0.5) grid.data[i] = 0;
            else                           grid.data[i] = -1;
        }
        pub_map_->publish(grid);
    }

    void broadcastMapTf()
    {
        geometry_msgs::msg::TransformStamped ts;
        ts.header.stamp    = this->get_clock()->now();
        ts.header.frame_id = "map";
        ts.child_frame_id  = odom_frame_;
        ts.transform.translation.x = map_odom_x_;
        ts.transform.translation.y = map_odom_y_;
        ts.transform.translation.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, map_odom_theta_);
        ts.transform.rotation = tf2::toMsg(q);
        tf_broadcaster_.sendTransform(ts);
    }
};

}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<graph_slam::OccupancyMapper>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}