#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>

#include <graph_slam/GetGraph.h>
#include "types.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

using namespace std;
namespace graph_slam {

const double LOGODD_HIT  =  0.85;
const double LOGODD_MISS = -0.30;
const double LOGODD_MIN  = -5.0;
const double LOGODD_MAX  =  5.0;

struct Particle {
    double x, y, theta, weight;
};

class OccupancyMapper {
public:
    OccupancyMapper(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh), tf_buffer_(), tf_listener_(tf_buffer_), rng_(std::random_device{}())
    {
        private_nh_.param("map_resolution",    map_resolution_,    0.05);
        private_nh_.param("map_width_m",        map_width_m_,       40.0);
        private_nh_.param("map_height_m",       map_height_m_,      40.0);
        private_nh_.param("publish_period_sec", publish_period_sec_, 0.05);
        private_nh_.param("map_angle_step_deg", map_angle_step_deg_, 0.5);
        private_nh_.param("fov_half_deg",       fov_half_deg_,    60.0);

        private_nh_.param("localization_enabled", localization_enabled_, true);
        private_nh_.param("num_particles",         num_particles_,        500);
        private_nh_.param("particle_sigma_x",      particle_sigma_x_,     0.1);
        private_nh_.param("particle_sigma_y",      particle_sigma_y_,     0.1);
        private_nh_.param("particle_sigma_theta",  particle_sigma_theta_,  0.05);
        private_nh_.param("motion_sigma_x",        motion_sigma_x_,       0.02);
        private_nh_.param("motion_sigma_y",        motion_sigma_y_,       0.02);
        private_nh_.param("motion_sigma_theta",    motion_sigma_theta_,   0.01);
        private_nh_.param("resample_threshold",    resample_threshold_,   0.5);
        private_nh_.param("scan_subsample",        scan_subsample_,       8);
        private_nh_.param("odom_frame",            odom_frame_,           string("odom"));
        private_nh_.param("base_frame",            base_frame_,           string("base_link"));

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

        pub_map_       = nh_.advertise<nav_msgs::OccupancyGrid>("/map", 1, true);
        pub_particle_cloud_ = nh_.advertise<nav_msgs::OccupancyGrid>("/particle_cloud_debug", 1, false);
        sub_trigger_   = nh_.subscribe("/graph_slam/trigger_remap", 1,
                                       &OccupancyMapper::triggerCallback, this);

        if (localization_enabled_) {
            sub_scan_  = nh_.subscribe("/scan", 10, &OccupancyMapper::scanCallback,  this);
            sub_initial_pose_ = nh_.subscribe("/initialpose", 1,
                                              &OccupancyMapper::initialPoseCallback, this);
        }

        clt_get_graph_ = nh_.serviceClient<graph_slam::GetGraph>("/graph_slam/get_graph");
        clt_get_graph_.waitForExistence();

        timer_ = nh_.createTimer(ros::Duration(publish_period_sec_),
                                 &OccupancyMapper::timerCallback, this);

        ROS_INFO("[OccupancyMapper] %.0fx%.0f m @ %.3f m/cell = %dx%d cells  fov=+/-%.0fdeg  localization=%s",
                 map_width_m_, map_height_m_, map_resolution_,
                 width_cells_, height_cells_, fov_half_deg_,
                 localization_enabled_ ? "ON" : "OFF");
    }

private:
    ros::NodeHandle    nh_, private_nh_;
    ros::Subscriber    sub_trigger_;
    ros::Subscriber    sub_scan_;
    ros::Subscriber    sub_initial_pose_;
    ros::Publisher     pub_map_;
    ros::Publisher     pub_particle_cloud_;
    ros::ServiceClient clt_get_graph_;
    ros::Timer         timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::Buffer               tf_buffer_;
    tf2_ros::TransformListener    tf_listener_;
    std::mt19937                  rng_;

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

    int last_mapped_node_ = -1;

    void triggerCallback(const std_msgs::Bool::ConstPtr& msg)
    {
        if (msg->data) {
            ROS_INFO("[OccupancyMapper] Full rebuild triggered");
            fullRebuild();
        }
    }

    void timerCallback(const ros::TimerEvent&)
    {
        incrementalUpdate();
        broadcastMapTf();
    }

    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
    {
        double ix = msg->pose.pose.position.x;
        double iy = msg->pose.pose.position.y;
        tf2::Quaternion q;
        tf2::fromMsg(msg->pose.pose.orientation, q);
        double itheta = atan2(2.0*(q.w()*q.z()+q.x()*q.y()),
                              1.0-2.0*(q.y()*q.y()+q.z()*q.z()));
        initParticles(ix, iy, itheta);
        ROS_INFO("[OccupancyMapper] Particles initialized from /initialpose (%.2f,%.2f,%.3f)",
                 ix, iy, itheta);
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
    {
        if (!map_ready_) return;

        geometry_msgs::TransformStamped odom_tf;
        try {
            odom_tf = tf_buffer_.lookupTransform(odom_frame_, base_frame_,
                                                 ros::Time(0), ros::Duration(0.05));
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
        std::normal_distribution<double> nx(cx,    particle_sigma_x_);
        std::normal_distribution<double> ny(cy,    particle_sigma_y_);
        std::normal_distribution<double> nt(ctheta,particle_sigma_theta_);
        for (auto& p : particles_) {
            p.x = nx(rng_); p.y = ny(rng_); p.theta = nt(rng_);
            p.weight = 1.0 / num_particles_;
        }
    }

    void motionUpdate(double dx, double dy, double dtheta)
    {
        std::normal_distribution<double> nx(0.0, motion_sigma_x_   + fabs(dx)*0.1);
        std::normal_distribution<double> ny(0.0, motion_sigma_y_   + fabs(dy)*0.1);
        std::normal_distribution<double> nt(0.0, motion_sigma_theta_+ fabs(dtheta)*0.05);
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

    void measurementUpdate(const sensor_msgs::LaserScan& scan)
    {
        int N = (int)scan.ranges.size();
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
            double target = u + (double)i / num_particles_;
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
        graph_slam::GetGraph srv;
        if (!clt_get_graph_.call(srv)) {
            ROS_ERROR("[OccupancyMapper] get_graph failed"); return;
        }
        int N = (int)srv.response.ids.size();
        if (N == 0) { ROS_WARN("[OccupancyMapper] Graph empty"); return; }

        fill(log_odds_.begin(), log_odds_.end(), 0.0);

        for (int i = 0; i < N; ++i)
            insertScan(srv.response.xs[i], srv.response.ys[i],
                       srv.response.thetas[i], srv.response.scans[i]);

        last_mapped_node_ = N - 1;
        map_ready_ = true;
        publishMap();
        ROS_INFO("[OccupancyMapper] Full rebuild: %d nodes", N);
    }

    void incrementalUpdate()
    {
        graph_slam::GetGraph srv;
        if (!clt_get_graph_.call(srv)) return;

        int N = (int)srv.response.ids.size();
        bool changed = false;
        for (int i = last_mapped_node_ + 1; i < N; ++i) {
            insertScan(srv.response.xs[i], srv.response.ys[i],
                       srv.response.thetas[i], srv.response.scans[i]);
            changed = true;
        }
        if (changed) {
            publishMap();
            last_mapped_node_ = N - 1;
            map_ready_ = true;
        }
    }

    void insertScan(double x, double y, double theta,
                    const sensor_msgs::LaserScan& scan)
    {
        if (scan.ranges.empty()) return;

        int rx = worldToCell(x, true);
        int ry = worldToCell(y, false);

        double step_rad = map_angle_step_deg_ * M_PI / 180.0;
        int    step     = max(1, static_cast<int>(step_rad / scan.angle_increment));

        double fov_rad = fov_half_deg_ * M_PI / 180.0;

        int num_rays = (int)scan.ranges.size();
        for (int i = 0; i < num_rays; i += step)
        {
            double angle = scan.angle_min + i * scan.angle_increment;

            double a_norm = fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
            if (fabs(a_norm) > fov_rad) continue;

            float r   = scan.ranges[i];
            bool  hit = isfinite(r) && r > scan.range_min && r < scan.range_max;

            double global_angle = theta + angle;

            if (hit) {
                double wx = x + r * cos(global_angle);
                double wy = y + r * sin(global_angle);
                int ex = worldToCell(wx, true);
                int ey = worldToCell(wy, false);
                bresenham(rx, ry, ex, ey, true);
            } else {
                double wx = x + scan.range_max * cos(global_angle);
                double wy = y + scan.range_max * sin(global_angle);
                int ex = worldToCell(wx, true);
                int ey = worldToCell(wy, false);
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

        while (true)
        {
            if (x < 0 || x >= width_cells_ || y < 0 || y >= height_cells_) break;

            if (x == x1 && y == y1) {
                updateCell(x, y, mark_hit ? LOGODD_HIT : LOGODD_MISS);
                break;
            }

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
        nav_msgs::OccupancyGrid grid;
        grid.header.frame_id           = "map";
        grid.header.stamp              = ros::Time::now();
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
        pub_map_.publish(grid);
    }

    void broadcastMapTf()
    {
        geometry_msgs::TransformStamped ts;
        ts.header.stamp    = ros::Time::now();
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
    ros::init(argc, argv, "occupancy_mapper");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    graph_slam::OccupancyMapper mapper(nh, private_nh);
    ros::spin();
    return 0;
}