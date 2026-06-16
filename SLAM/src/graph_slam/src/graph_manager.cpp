#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/LaserScan.h>

#include <graph_slam/AddNode.h>
#include <graph_slam/AddEdge.h>
#include <graph_slam/GetGraph.h>
#include <graph_slam/UpdatePoses.h>

#include "types.hpp"

#include <mutex>
#include <vector>
using namespace std;

namespace graph_slam {

    class GraphManager {
    public:
        GraphManager(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
        : nh_(nh), private_nh_(private_nh)
        {
            add_node_srv_    = nh_.advertiseService("/graph_slam/add_node",    &GraphManager::addNodeCallback,    this);
            add_edge_srv_    = nh_.advertiseService("/graph_slam/add_edge",    &GraphManager::addEdgeCallback,    this);
            get_graph_srv_   = nh_.advertiseService("/graph_slam/get_graph",   &GraphManager::getGraphCallback,   this);
            update_poses_srv_= nh_.advertiseService("/graph_slam/update_poses",&GraphManager::updatePosesCallback,this);

            publish_markers        = nh_.advertise<visualization_msgs::MarkerArray>("graph_markers",    1, true);
            publish_raw_path       = nh_.advertise<nav_msgs::Path>("raw_path",       1, true);
            publish_optimized_path = nh_.advertise<nav_msgs::Path>("optimized_path", 1, true);

            private_nh_.param("node_scale", node_scale_, 0.05);
            private_nh_.param("edge_width", edge_width_, 0.02);

            ROS_INFO("GraphManager initialized with node_scale: %f, edge_width: %f", node_scale_, edge_width_);
        }

    private:

        vector<PoseNode> nodes_;
        vector<Edge>     edges_;

        ros::NodeHandle  nh_;
        ros::NodeHandle  private_nh_;
        ros::ServiceServer add_node_srv_;
        ros::ServiceServer add_edge_srv_;
        ros::ServiceServer get_graph_srv_;
        ros::ServiceServer update_poses_srv_;
        ros::Publisher publish_markers;
        ros::Publisher publish_raw_path;
        ros::Publisher publish_optimized_path;
        double node_scale_;
        double edge_width_;

        bool addNodeCallback(graph_slam::AddNode::Request& req, graph_slam::AddNode::Response& res)
        {
            PoseNode new_node;
            new_node.id        = nodes_.size();
            new_node.x         = req.x;
            new_node.y         = req.y;
            new_node.theta     = req.theta;
            new_node.scan      = req.scan;
            new_node.timestamp = req.timestamp;

            nodes_.push_back(new_node);
            res.id      = new_node.id;
            res.success = true;
            publishAll();
            return true;
        }

        bool addEdgeCallback(graph_slam::AddEdge::Request& req, graph_slam::AddEdge::Response& res)
        {
            if (req.from_id >= nodes_.size() || req.to_id >= nodes_.size()) {
                res.success = false;
                return true;
            }

            Edge new_edge;
            new_edge.from_id  = req.from_id;
            new_edge.to_id    = req.to_id;
            new_edge.type     = static_cast<Edge::Type>(req.type);
            new_edge.dx       = req.dx;
            new_edge.dy       = req.dy;
            new_edge.dtheta   = req.dtheta;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    new_edge.information(i, j) = req.information[i * 3 + j];

            edges_.push_back(new_edge);
            res.success = true;
            publishAll();
            return true;
        }

        bool getGraphCallback(graph_slam::GetGraph::Request& req, graph_slam::GetGraph::Response& res)
        {
            for (auto& node : nodes_) {
                res.ids.push_back(node.id);
                res.xs.push_back(node.x);
                res.ys.push_back(node.y);
                res.thetas.push_back(node.theta);
                res.timestamps.push_back(node.timestamp);
                res.scans.push_back(node.scan);
            }

            for (auto& edge : edges_) {
                res.edges_from_id.push_back(edge.from_id);
                res.edges_to_id.push_back(edge.to_id);
                res.edges_type.push_back(edge.type);
                res.edges_dx.push_back(edge.dx);
                res.edges_dy.push_back(edge.dy);
                res.edges_dtheta.push_back(edge.dtheta);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        res.edges_information.push_back(edge.information(i, j));
            }
            return true;
        }

        bool updatePosesCallback(graph_slam::UpdatePoses::Request& req, graph_slam::UpdatePoses::Response& res)
        {
            if (req.ids.size() != req.xs.size() || req.ids.size() != req.ys.size() || req.ids.size() != req.thetas.size()) {
                res.success = false;
                return true;
            }

            for (size_t i = 0; i < req.ids.size(); ++i) {
                int id = req.ids[i];
                if (id >= (int)nodes_.size()) {
                    res.success = false;
                    return true;
                }
                nodes_[id].x     = req.xs[i];
                nodes_[id].y     = req.ys[i];
                nodes_[id].theta = req.thetas[i];
            }

            res.success = true;
            publishAll();
            return true;
        }

        void publishAll() {
            publishMarkers();
            publishRawPath();
        }

        void publishMarkers() {
            visualization_msgs::MarkerArray marker_array;
            ros::Time now = ros::Time::now();

            visualization_msgs::Marker delete_marker;
            delete_marker.action = visualization_msgs::Marker::DELETEALL;
            marker_array.markers.push_back(delete_marker);

            for (auto& node : nodes_) {
                visualization_msgs::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp    = now;
                marker.ns              = "nodes";
                marker.id              = node.id;
                marker.type            = visualization_msgs::Marker::SPHERE;
                marker.action          = visualization_msgs::Marker::ADD;
                marker.pose.position.x = node.x;
                marker.pose.position.y = node.y;
                marker.pose.position.z = 0.0;
                marker.pose.orientation.w = 1.0;
                marker.scale.x = node_scale_;
                marker.scale.y = node_scale_;
                marker.scale.z = node_scale_;
                marker.color.a = 1.0;
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;

                visualization_msgs::Marker text_marker;
                text_marker.header.frame_id = "map";
                text_marker.header.stamp    = now;
                text_marker.ns              = "node_ids";
                text_marker.id              = node.id;
                text_marker.type            = visualization_msgs::Marker::TEXT_VIEW_FACING;
                text_marker.action          = visualization_msgs::Marker::ADD;
                text_marker.pose            = marker.pose;
                text_marker.pose.position.z += 0.1;
                text_marker.scale.z = 0.1;
                text_marker.color.a = 1.0;
                text_marker.color.r = 1.0;
                text_marker.color.g = 1.0;
                text_marker.color.b = 1.0;
                text_marker.text    = to_string(node.id);

                marker_array.markers.push_back(marker);
                marker_array.markers.push_back(text_marker);
            }

            int edge_id = 0;
            for (const auto& edge : edges_) {
                if (edge.from_id >= (int)nodes_.size() || edge.to_id >= (int)nodes_.size()) continue;

                auto& from_node = nodes_[edge.from_id];
                auto& to_node   = nodes_[edge.to_id];

                visualization_msgs::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp    = now;
                marker.ns    = (edge.type == Edge::ODOMETRY) ? "odometry_edges" : "loop_closure_edges";
                marker.id    = edge_id++;
                marker.type  = visualization_msgs::Marker::LINE_LIST;
                marker.action= visualization_msgs::Marker::ADD;
                marker.pose.orientation.w = 1.0;
                marker.color.a = 1.0;

                if (edge.type == Edge::ODOMETRY) {
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

                geometry_msgs::Point p1, p2;
                p1.x = from_node.x; p1.y = from_node.y; p1.z = 0.0;
                p2.x = to_node.x;   p2.y = to_node.y;   p2.z = 0.0;
                marker.points.push_back(p1);
                marker.points.push_back(p2);
                marker_array.markers.push_back(marker);
            }

            publish_markers.publish(marker_array);
        }

        void publishRawPath() {
            nav_msgs::Path path;
            path.header.frame_id = "map";
            path.header.stamp    = ros::Time::now();
            for (const auto& node : nodes_) {
                geometry_msgs::PoseStamped pose;
                pose.header            = path.header;
                pose.pose.position.x   = node.x;
                pose.pose.position.y   = node.y;
                pose.pose.position.z   = 0.0;
                pose.pose.orientation.z = sin(node.theta / 2.0);
                pose.pose.orientation.w = cos(node.theta / 2.0);
                path.poses.push_back(pose);
            }
            publish_raw_path.publish(path);
            publish_optimized_path.publish(path);
        }

    };
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "graph_manager");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    graph_slam::GraphManager manager(nh, private_nh);
    ros::spin();
    return 0;
}