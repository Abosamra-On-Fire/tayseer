#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include "graph_slam/srv/add_node.hpp"
#include "graph_slam/srv/add_edge.hpp"
#include "graph_slam/srv/get_graph.hpp"
#include "graph_slam/srv/update_poses.hpp"
#include "graph_slam/srv/remove_edge.hpp"

#include "types.hpp"

#include <mutex>
#include <vector>
#include <algorithm>
using namespace std;

namespace graph_slam {

    class GraphManager : public rclcpp::Node {
    public:
        GraphManager()
        : Node("graph_manager")
        {
            node_scale_ = this->declare_parameter("node_scale", 0.05);
            edge_width_ = this->declare_parameter("edge_width", 0.02);

            cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

            add_node_srv_ = this->create_service<graph_slam::srv::AddNode>("/graph_slam/add_node",
                bind(&GraphManager::addNodeCallback, this, placeholders::_1, placeholders::_2),
                rmw_qos_profile_services_default, cb_group_);

            add_edge_srv_ = this->create_service<graph_slam::srv::AddEdge>("/graph_slam/add_edge",
                bind(&GraphManager::addEdgeCallback, this, placeholders::_1, placeholders::_2),
                rmw_qos_profile_services_default, cb_group_);

            get_graph_srv_ = this->create_service<graph_slam::srv::GetGraph>("/graph_slam/get_graph",
                bind(&GraphManager::getGraphCallback, this, placeholders::_1, placeholders::_2),
                rmw_qos_profile_services_default, cb_group_);

            update_poses_srv_ = this->create_service<graph_slam::srv::UpdatePoses>("/graph_slam/update_poses",
                bind(&GraphManager::updatePosesCallback, this, placeholders::_1, placeholders::_2),
                rmw_qos_profile_services_default, cb_group_);

            remove_edge_srv_ = this->create_service<graph_slam::srv::RemoveEdge>("/graph_slam/remove_edge",
                bind(&GraphManager::removeEdgeCallback, this, placeholders::_1, placeholders::_2),
                rmw_qos_profile_services_default, cb_group_);

            rclcpp::QoS latched_qos(1);
            latched_qos.transient_local();

            publish_markers  = this->create_publisher<visualization_msgs::msg::MarkerArray>("graph_markers", latched_qos);
            publish_raw_path = this->create_publisher<nav_msgs::msg::Path>("raw_path", latched_qos);

            RCLCPP_INFO(this->get_logger(), "GraphManager initialized with node_scale: %f, edge_width: %f", node_scale_, edge_width_);
        }

    private:
        vector<Edge>     edges_;
        vector<PoseNode> nodes_;
        mutex            mutex_;

        rclcpp::CallbackGroup::SharedPtr cb_group_;

        rclcpp::Service<graph_slam::srv::AddNode>::SharedPtr      add_node_srv_;
        rclcpp::Service<graph_slam::srv::AddEdge>::SharedPtr      add_edge_srv_;
        rclcpp::Service<graph_slam::srv::GetGraph>::SharedPtr     get_graph_srv_;
        rclcpp::Service<graph_slam::srv::UpdatePoses>::SharedPtr  update_poses_srv_;
        rclcpp::Service<graph_slam::srv::RemoveEdge>::SharedPtr   remove_edge_srv_;

        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publish_markers;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  publish_raw_path;

        double node_scale_;
        double edge_width_;

        void addNodeCallback(const shared_ptr<graph_slam::srv::AddNode::Request> req, shared_ptr<graph_slam::srv::AddNode::Response> res)
        {
            lock_guard<mutex> lock(mutex_);
            PoseNode n;
            n.id        = nodes_.size();
            n.x         = req->x;
            n.y         = req->y;
            n.theta     = req->theta;
            n.scan      = req->scan;
            n.timestamp = req->timestamp;

            nodes_.push_back(n);
            res->id      = n.id;
            res->success = true;
            publishAll();
        }

        void addEdgeCallback(const shared_ptr<graph_slam::srv::AddEdge::Request> req, shared_ptr<graph_slam::srv::AddEdge::Response> res)
        {
            lock_guard<mutex> lock(mutex_);
            if (req->from_id >= (int)nodes_.size() || req->to_id >= (int)nodes_.size()) {
                res->success = false;
                return;
            }

            for (const auto& e : edges_) {
                if (e.from_id == req->from_id && e.to_id == req->to_id &&
                    e.type == static_cast<Edge::Type>(req->type)) {
                    res->success = false;
                    return;
                }
            }

            Edge e;
            e.from_id  = req->from_id;
            e.to_id    = req->to_id;
            e.type     = static_cast<Edge::Type>(req->type);
            e.dx       = req->dx;
            e.dy       = req->dy;
            e.dtheta   = req->dtheta;

            if (req->information.size() >= 9) {
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        e.information(r, c) = req->information[r * 3 + c];
            } else {
                e.information = Eigen::Matrix3d::Identity() * 100.0;
            }

            edges_.push_back(e);
            res->success = true;
            publishAll();
        }

        void removeEdgeCallback(const shared_ptr<graph_slam::srv::RemoveEdge::Request> req, shared_ptr<graph_slam::srv::RemoveEdge::Response> res)
        {
            lock_guard<mutex> lock(mutex_);
            auto it = find_if(edges_.begin(), edges_.end(), [&](const Edge& e) {
                return e.from_id == req->from_id && e.to_id == req->to_id &&
                       e.type == Edge::LOOP_CLOSURE;
            });

            if (it == edges_.end()) {
                res->success = false;
                return;
            }

            edges_.erase(it);
            res->success = true;
            publishAll();
        }

        void getGraphCallback(const shared_ptr<graph_slam::srv::GetGraph::Request> req, shared_ptr<graph_slam::srv::GetGraph::Response> res)
        {
            lock_guard<mutex> lock(mutex_);
            for (size_t i = 0; i < nodes_.size(); ++i) {
                res->ids.push_back(nodes_[i].id);
                res->xs.push_back(nodes_[i].x);
                res->ys.push_back(nodes_[i].y);
                res->thetas.push_back(nodes_[i].theta);
                res->timestamps.push_back(nodes_[i].timestamp);
                res->scans.push_back(nodes_[i].scan);
            }

            for (size_t ei = 0; ei < edges_.size(); ++ei) {
                res->edges_from_id.push_back(edges_[ei].from_id);
                res->edges_to_id.push_back(edges_[ei].to_id);
                res->edges_type.push_back(edges_[ei].type);
                res->edges_dx.push_back(edges_[ei].dx);
                res->edges_dy.push_back(edges_[ei].dy);
                res->edges_dtheta.push_back(edges_[ei].dtheta);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        res->edges_information.push_back(edges_[ei].information(r, c));
            }
        }

        void updatePosesCallback(const shared_ptr<graph_slam::srv::UpdatePoses::Request> req, shared_ptr<graph_slam::srv::UpdatePoses::Response> res)
        {
            lock_guard<mutex> lock(mutex_);
            if (req->ids.size() != req->xs.size() || req->ids.size() != req->ys.size() || req->ids.size() != req->thetas.size()) {
                res->success = false;
                return;
            }

            for (size_t i = 0; i < req->ids.size(); ++i) {
                int id = req->ids[i];
                if (id >= (int)nodes_.size()) {
                    res->success = false;
                    return;
                }
                nodes_[id].x     = req->xs[i];
                nodes_[id].y     = req->ys[i];
                nodes_[id].theta = req->thetas[i];
            }

            res->success = true;
            publishAll();
        }

        void publishAll() {
            publishMarkers();
            publishRawPath();
        }

        void publishMarkers() {
            visualization_msgs::msg::MarkerArray marker_array;
            rclcpp::Time now = this->get_clock()->now();

            visualization_msgs::msg::Marker delete_marker;
            delete_marker.header.frame_id = "map";
            delete_marker.header.stamp    = now;
            delete_marker.ns              = "delete_all";
            delete_marker.id              = 0;
            delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(delete_marker);

            for (size_t i = 0; i < nodes_.size(); ++i) {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp    = now;
                marker.ns              = "nodes";
                marker.id              = nodes_[i].id;
                marker.type            = visualization_msgs::msg::Marker::SPHERE;
                marker.action          = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = nodes_[i].x;
                marker.pose.position.y = nodes_[i].y;
                marker.pose.position.z = 0.0;
                marker.pose.orientation.w = 1.0;
                marker.scale.x = node_scale_;
                marker.scale.y = node_scale_;
                marker.scale.z = node_scale_;
                marker.color.a = 1.0;
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;

                visualization_msgs::msg::Marker text_marker;
                text_marker.header.frame_id = "map";
                text_marker.header.stamp    = now;
                text_marker.ns              = "node_ids";
                text_marker.id              = nodes_[i].id;
                text_marker.type            = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.action          = visualization_msgs::msg::Marker::ADD;
                text_marker.pose            = marker.pose;
                text_marker.pose.position.z += 0.1;
                text_marker.scale.z = 0.1;
                text_marker.color.a = 1.0;
                text_marker.color.r = 1.0;
                text_marker.color.g = 1.0;
                text_marker.color.b = 1.0;
                text_marker.text    = to_string(nodes_[i].id);

                marker_array.markers.push_back(marker);
                marker_array.markers.push_back(text_marker);
            }

            int edge_id = 0;
            for (size_t i = 0; i < edges_.size(); ++i) {
                if (edges_[i].from_id >= (int)nodes_.size() || edges_[i].to_id >= (int)nodes_.size()) continue;

                auto& from_node = nodes_[edges_[i].from_id];
                auto& to_node   = nodes_[edges_[i].to_id];

                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp    = now;
                marker.ns    = (edges_[i].type == Edge::ODOMETRY) ? "odometry_edges" : "loop_closure_edges";
                marker.id    = edge_id++;
                marker.type  = visualization_msgs::msg::Marker::LINE_LIST;
                marker.action= visualization_msgs::msg::Marker::ADD;
                marker.pose.orientation.w = 1.0;
                marker.color.a = 1.0;

                if (edges_[i].type == Edge::ODOMETRY) {
                    marker.scale.x = edge_width_;
                    marker.color.r = 0.0;
                    marker.color.g = 0.0;
                    marker.color.b = 1.0;
                } else {
                    marker.scale.x = edge_width_ * 2.0;
                    marker.color.r = 1.0;
                    marker.color.g = 0.0;
                    marker.color.b = 0.0;
                }

                geometry_msgs::msg::Point p1, p2;
                p1.x = from_node.x; p1.y = from_node.y; p1.z = 0.0;
                p2.x = to_node.x;   p2.y = to_node.y;   p2.z = 0.0;
                marker.points.push_back(p1);
                marker.points.push_back(p2);
                marker_array.markers.push_back(marker);
            }

            publish_markers->publish(marker_array);
        }

        void publishRawPath() {
            nav_msgs::msg::Path path;
            path.header.frame_id = "map";
            path.header.stamp    = this->get_clock()->now();
            for (size_t i = 0; i < nodes_.size(); ++i) {
                geometry_msgs::msg::PoseStamped pose;
                pose.header            = path.header;
                pose.pose.position.x   = nodes_[i].x;
                pose.pose.position.y   = nodes_[i].y;
                pose.pose.position.z   = 0.0;
                pose.pose.orientation.z = sin(nodes_[i].theta / 2.0);
                pose.pose.orientation.w = cos(nodes_[i].theta / 2.0);
                path.poses.push_back(pose);
            }
            publish_raw_path->publish(path);
        }
    };
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto manager = std::make_shared<graph_slam::GraphManager>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(manager);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}