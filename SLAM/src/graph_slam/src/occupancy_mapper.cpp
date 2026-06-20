#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>

#include "graph_slam/srv/get_graph.hpp"
#include "types.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;
namespace graph_slam {

const double LOGODD_HIT  =  1.0;
const double LOGODD_MISS = -0.25;
const double LOGODD_MIN  = -2.0;
const double LOGODD_MAX  =  4.0;

class OccupancyMapper : public rclcpp::Node
{
public:
    explicit OccupancyMapper()
    : Node("occupancy_mapper")
    {
        map_resolution_     = declare_parameter<double>("map_resolution",     0.05);
        map_width_m_        = declare_parameter<double>("map_width_m",        40.0);
        map_height_m_       = declare_parameter<double>("map_height_m",       40.0);
        publish_period_sec_ = declare_parameter<double>("publish_period_sec", 0.05);
        map_angle_step_deg_ = declare_parameter<double>("map_angle_step_deg", 0.5);
        fov_half_deg_       = declare_parameter<double>("fov_half_deg",       60.0);

        width_cells_  = map_width_m_  / map_resolution_;
        height_cells_ = map_height_m_ / map_resolution_;
        origin_x_ = -map_width_m_  / 2.0;
        origin_y_ = -map_height_m_ / 2.0;

        log_odds_.assign(width_cells_ * height_cells_, 0.0);

        cb_group_reentrant_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        cb_group_get_graph_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = cb_group_reentrant_;

        pub_map_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(1).transient_local());

        sub_trigger_ = create_subscription<std_msgs::msg::Bool>(
            "/graph_slam/trigger_remap", 1,
            [this](std_msgs::msg::Bool::SharedPtr msg){ triggerCallback(msg); },
            sub_opts);

        clt_get_graph_ = create_client<graph_slam::srv::GetGraph>(
            "graph_slam/get_graph",
            rmw_qos_profile_services_default,
            cb_group_get_graph_);
        clt_get_graph_->wait_for_service();

        timer_ = create_wall_timer(
            std::chrono::duration<double>(publish_period_sec_),
            [this](){ incrementalUpdate(); },
            cb_group_reentrant_);

        RCLCPP_INFO(get_logger(),
            "[OccupancyMapper] %.0fx%.0f m @ %.3f m/cell = %dx%d cells  fov=+/-%.0f deg",
            map_width_m_, map_height_m_, map_resolution_,
            width_cells_, height_cells_, fov_half_deg_);
    }

private:
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr       sub_trigger_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_map_;
    rclcpp::Client<graph_slam::srv::GetGraph>::SharedPtr        clt_get_graph_;
    rclcpp::TimerBase::SharedPtr                                timer_;

    rclcpp::CallbackGroup::SharedPtr cb_group_reentrant_;
    rclcpp::CallbackGroup::SharedPtr cb_group_get_graph_;

    double map_resolution_;
    double map_width_m_, map_height_m_;
    double publish_period_sec_;
    double map_angle_step_deg_;
    double fov_half_deg_;

    int    width_cells_, height_cells_;
    double origin_x_,    origin_y_;
    vector<double> log_odds_;

    int  last_mapped_node_ = -1;



    void triggerCallback(std_msgs::msg::Bool::SharedPtr msg)
    {
        if (!msg->data) return;
        RCLCPP_INFO(get_logger(), "[OccupancyMapper] Full rebuild triggered");

        fill(log_odds_.begin(), log_odds_.end(), 0.0);
        last_mapped_node_ = -1;

        auto resp = callGetGraph();
        if (!resp) return;

        int N = static_cast<int>(resp->ids.size());
        if (N == 0) { RCLCPP_WARN(get_logger(), "[OccupancyMapper] Graph empty"); return; }

        for (int i = 0; i < N; ++i)
            insertScan(resp->xs[i], resp->ys[i], resp->thetas[i], resp->scans[i]);

        last_mapped_node_ = N - 1;
        publishMap();
        RCLCPP_INFO(get_logger(), "[OccupancyMapper] Full rebuild done: %d nodes", N);
    }


    void incrementalUpdate()
    {
        auto resp = callGetGraph();
        if (!resp) return;

        int N = static_cast<int>(resp->ids.size());
        bool changed = false;
        for (int i = last_mapped_node_ + 1; i < N; ++i) {
            insertScan(resp->xs[i], resp->ys[i], resp->thetas[i], resp->scans[i]);
            changed = true;
        }
        if (changed) {
            last_mapped_node_ = N - 1;
            publishMap();
        }
    }


    graph_slam::srv::GetGraph::Response::SharedPtr callGetGraph()
    {
        auto req    = std::make_shared<graph_slam::srv::GetGraph::Request>();
        auto future = clt_get_graph_->async_send_request(req);
        if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "[OccupancyMapper] get_graph timed out");
            return nullptr;
        }
        return future.get();
    }


    void insertScan(double x, double y, double theta,
                    sensor_msgs::msg::LaserScan& scan)
    {
        if (scan.ranges.empty()) return;

        int rx = worldToCell(x, true);
        int ry = worldToCell(y, false);

        double step_rad = map_angle_step_deg_ * M_PI / 180.0;
        int    step     = max(1, static_cast<int>(step_rad / scan.angle_increment));
        double fov_rad  = fov_half_deg_ * M_PI / 180.0;

        double angle = scan.angle_min;
        for (size_t i = 0; i < scan.ranges.size();
             i += step, angle += step * scan.angle_increment)
        {
            double a_norm = fmod(angle + M_PI, 2.0*M_PI) - M_PI;
            if (fabs(a_norm) > fov_rad) continue;

            float r   = scan.ranges[i];
            bool  hit = isfinite(r) && r > scan.range_min && r < scan.range_max;

            double end_r = hit ? static_cast<double>(r) : scan.range_max;
            double wx    = x + end_r * cos(theta + angle);
            double wy    = y + end_r * sin(theta + angle);

            int ex = worldToCell(wx, true);
            int ey = worldToCell(wy, false);

            bresenham(rx, ry, ex, ey, hit);
        }
    }

    void bresenham(int x0, int y0, int x1, int y1, bool mark_hit)
    {
        int dx = abs(x1 - x0), dy = abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int f  = dx - dy;
        int x  = x0, y = y0;

        while (true) {
            if (x < 0 || x >= width_cells_ || y < 0 || y >= height_cells_) break;

            if (x == x1 && y == y1) {
                if (mark_hit) {
                    updateCell(x, y, LOGODD_HIT);
                    smearOccupied(x, y, 2, LOGODD_HIT / 2.0);
                }
                break;
            }

            updateCell(x, y, LOGODD_MISS);

            int f2 = 2 * f;
            if (f2 > -dy) { f -= dy; x += sx; }
            if (f2 <  dx) { f += dx; y += sy; }
        }
    }

    void smearOccupied(int cx, int cy, int radius_cells, double delta)
    {
        for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
            for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
                if (dx*dx + dy*dy > radius_cells*radius_cells) continue;
                int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || nx >= width_cells_ || ny < 0 || ny >= height_cells_) continue;
                updateCell(nx, ny, delta);
            }
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
            if      (log_odds_[i] >  1.0) grid.data[i] = 100;
            else if (log_odds_[i] < -0.5) grid.data[i] = 0;
            else                           grid.data[i] = -1;
        }
        pub_map_->publish(grid);
    }
};

} 
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<graph_slam::OccupancyMapper>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}