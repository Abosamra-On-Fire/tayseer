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

#include "types.hpp"
#include <mutex>
#include <vector>
#include <string>

using namespace std;
namespace graph_slam {

 
class GraphManager : public rclcpp::Node           
{
public:
     
    explicit GraphManager()
    : Node("graph_manager")                        
    {
         
         
        node_scale_ = declare_parameter<double>("node_scale", 0.05);
        edge_width_ = declare_parameter<double>("edge_width", 0.02);

         
         
         
        add_node_srv_ = create_service<graph_slam::srv::AddNode>(
            "/graph_slam/add_node",
            [this](const graph_slam::srv::AddNode::Request::SharedPtr req,
                         graph_slam::srv::AddNode::Response::SharedPtr res)
            { addNodeCallback(req, res); });

        add_edge_srv_ = create_service<graph_slam::srv::AddEdge>(
            "/graph_slam/add_edge",
            [this](const graph_slam::srv::AddEdge::Request::SharedPtr req,
                         graph_slam::srv::AddEdge::Response::SharedPtr res)
            { addEdgeCallback(req, res); });

        get_graph_srv_ = create_service<graph_slam::srv::GetGraph>(
            "/graph_slam/get_graph",
            [this](const graph_slam::srv::GetGraph::Request::SharedPtr req,
                         graph_slam::srv::GetGraph::Response::SharedPtr res)
            { getGraphCallback(req, res); });

        update_poses_srv_ = create_service<graph_slam::srv::UpdatePoses>(
            "/graph_slam/update_poses",
            [this](const graph_slam::srv::UpdatePoses::Request::SharedPtr req,
                         graph_slam::srv::UpdatePoses::Response::SharedPtr res)
            { updatePosesCallback(req, res); });

         
         
         
         
        auto latched_qos = rclcpp::QoS(1).transient_local();

        pub_markers_        = create_publisher<visualization_msgs::msg::MarkerArray>(
                                  "graph_markers",    latched_qos);
        pub_raw_path_       = create_publisher<nav_msgs::msg::Path>(
                                  "raw_path",         latched_qos);
        pub_optimized_path_ = create_publisher<nav_msgs::msg::Path>(
                                  "optimized_path",   latched_qos);

        RCLCPP_INFO(get_logger(),                    
            "GraphManager initialized  node_scale=%.3f  edge_width=%.3f",
            node_scale_, edge_width_);
    }

private:
     
    vector<PoseNode> nodes_;
    vector<Edge>     edges_;

     
    rclcpp::Service<graph_slam::srv::AddNode>::SharedPtr    add_node_srv_;
    rclcpp::Service<graph_slam::srv::AddEdge>::SharedPtr    add_edge_srv_;
    rclcpp::Service<graph_slam::srv::GetGraph>::SharedPtr   get_graph_srv_;
    rclcpp::Service<graph_slam::srv::UpdatePoses>::SharedPtr update_poses_srv_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  pub_raw_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  pub_optimized_path_;

    double node_scale_;
    double edge_width_;

     
     
    void addNodeCallback(
        const graph_slam::srv::AddNode::Request::SharedPtr  req,
              graph_slam::srv::AddNode::Response::SharedPtr res)
    {
        PoseNode new_node;
        new_node.id        = static_cast<int>(nodes_.size());
        new_node.x         = req->x;
        new_node.y         = req->y;
        new_node.theta     = req->theta;
        new_node.scan      = req->scan;
        new_node.timestamp = req->timestamp;
        nodes_.push_back(new_node);

        res->id      = new_node.id;
        res->success = true;
        publishAll();
    }

    void addEdgeCallback(
        const graph_slam::srv::AddEdge::Request::SharedPtr  req,
              graph_slam::srv::AddEdge::Response::SharedPtr res)
    {
        if (req->from_id >= static_cast<int>(nodes_.size()) ||
            req->to_id   >= static_cast<int>(nodes_.size()))
        {
            res->success = false;
            return;
        }

        Edge new_edge;
        new_edge.from_id = req->from_id;
        new_edge.to_id   = req->to_id;
        new_edge.type    = static_cast<Edge::Type>(req->type);
        new_edge.dx      = req->dx;
        new_edge.dy      = req->dy;
        new_edge.dtheta  = req->dtheta;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                new_edge.information(i, j) = req->information[i * 3 + j];

        edges_.push_back(new_edge);
        res->success = true;
        publishAll();
    }

    void getGraphCallback(
        const graph_slam::srv::GetGraph::Request::SharedPtr  /*req*/,
              graph_slam::srv::GetGraph::Response::SharedPtr res)
    {
        for (auto& node : nodes_) {
            res->ids.push_back(node.id);
            res->xs.push_back(node.x);
            res->ys.push_back(node.y);
            res->thetas.push_back(node.theta);
            res->timestamps.push_back(node.timestamp);
            res->scans.push_back(node.scan);
        }
        for (auto& edge : edges_) {
            res->edges_from_id.push_back(edge.from_id);
            res->edges_to_id.push_back(edge.to_id);
            res->edges_type.push_back(static_cast<int>(edge.type));
            res->edges_dx.push_back(edge.dx);
            res->edges_dy.push_back(edge.dy);
            res->edges_dtheta.push_back(edge.dtheta);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    res->edges_information.push_back(edge.information(i, j));
        }
    }

    void updatePosesCallback(
        const graph_slam::srv::UpdatePoses::Request::SharedPtr  req,
              graph_slam::srv::UpdatePoses::Response::SharedPtr res)
    {
        if (req->ids.size() != req->xs.size() ||
            req->ids.size() != req->ys.size() ||
            req->ids.size() != req->thetas.size())
        {
            res->success = false;
            return;
        }
        for (size_t i = 0; i < req->ids.size(); ++i) {
            int id = req->ids[i];
            if (id >= static_cast<int>(nodes_.size())) {
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

     
    void publishAll()
    {
        publishMarkers();
        publishRawPath();
    }

    void publishMarkers()
    {
        visualization_msgs::msg::MarkerArray marker_array;
         
        auto now = this->get_clock()->now();           

        visualization_msgs::msg::Marker del;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(del);

        for (auto& node : nodes_) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";
            m.header.stamp    = now;
            m.ns              = "nodes";
            m.id              = node.id;
            m.type            = visualization_msgs::msg::Marker::SPHERE;
            m.action          = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = node.x;
            m.pose.position.y = node.y;
            m.pose.position.z = 0.0;
            m.pose.orientation.w = 1.0;
            m.scale.x = node_scale_;
            m.scale.y = node_scale_;
            m.scale.z = node_scale_;
            m.color.a = 1.0; m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0;

            visualization_msgs::msg::Marker tm;
            tm.header     = m.header;
            tm.ns         = "node_ids";
            tm.id         = node.id;
            tm.type       = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            tm.action     = visualization_msgs::msg::Marker::ADD;
            tm.pose       = m.pose;
            tm.pose.position.z += 0.1;
            tm.scale.z    = 0.1;
            tm.color.a = 1.0; tm.color.r = 1.0; tm.color.g = 1.0; tm.color.b = 1.0;
            tm.text       = to_string(node.id);

            marker_array.markers.push_back(m);
            marker_array.markers.push_back(tm);
        }

        int edge_id = 0;
        for (auto& edge : edges_) {
            if (edge.from_id >= static_cast<int>(nodes_.size()) ||
                edge.to_id   >= static_cast<int>(nodes_.size())) continue;

            auto& fn = nodes_[edge.from_id];
            auto& tn = nodes_[edge.to_id];

            visualization_msgs::msg::Marker em;
            em.header.frame_id = "map";
            em.header.stamp    = now;
            em.ns    = (edge.type == Edge::ODOMETRY) ? "odometry_edges" : "loop_closure_edges";
            em.id    = edge_id++;
            em.type  = visualization_msgs::msg::Marker::LINE_LIST;
            em.action= visualization_msgs::msg::Marker::ADD;
            em.pose.orientation.w = 1.0;
            em.color.a = 1.0;

            if (edge.type == Edge::ODOMETRY) {
                em.scale.x = edge_width_;
                em.color.r = 0.0; em.color.g = 0.0; em.color.b = 1.0;
            } else {
                em.scale.x = edge_width_ * 2.0;
                em.color.r = 1.0; em.color.g = 0.0; em.color.b = 0.0;
            }

            geometry_msgs::msg::Point p1, p2;
            p1.x = fn.x; p1.y = fn.y; p1.z = 0.0;
            p2.x = tn.x; p2.y = tn.y; p2.z = 0.0;
            em.points.push_back(p1);
            em.points.push_back(p2);
            marker_array.markers.push_back(em);
        }

        pub_markers_->publish(marker_array);           
    }

    void publishRawPath()
    {
        nav_msgs::msg::Path path;
        path.header.frame_id = "map";
        path.header.stamp    = this->get_clock()->now();

        for (auto& node : nodes_) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header              = path.header;
            pose.pose.position.x     = node.x;
            pose.pose.position.y     = node.y;
            pose.pose.position.z     = 0.0;
            pose.pose.orientation.z  = sin(node.theta / 2.0);
            pose.pose.orientation.w  = cos(node.theta / 2.0);
            path.poses.push_back(pose);
        }

        pub_raw_path_->publish(path);
        pub_optimized_path_->publish(path);
    }
};

}  

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<graph_slam::GraphManager>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}