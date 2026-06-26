#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

#include "graph_slam/srv/add_node.hpp"
#include "graph_slam/srv/add_edge.hpp"
#include "graph_slam/srv/get_graph.hpp"
#include "graph_slam/srv/update_poses.hpp"

#include "types.hpp"

#include <vector>
#include <algorithm>
using namespace std;

namespace graph_slam {

class GraphManager : public rclcpp::Node {
public:
    GraphManager()
    : Node("graph_manager")
    {


        srv_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        add_node_srv_ = this->create_service<graph_slam::srv::AddNode>("/graph_slam/add_node",
            bind(&GraphManager::onAddNode, this, placeholders::_1, placeholders::_2),
            rmw_qos_profile_services_default, srv_group_);

        add_edge_srv_ = this->create_service<graph_slam::srv::AddEdge>("/graph_slam/add_edge",
            bind(&GraphManager::onAddEdge, this, placeholders::_1, placeholders::_2),
            rmw_qos_profile_services_default, srv_group_);

        get_graph_srv_ = this->create_service<graph_slam::srv::GetGraph>("/graph_slam/get_graph",
            bind(&GraphManager::onGetGraph, this, placeholders::_1, placeholders::_2),
            rmw_qos_profile_services_default, srv_group_);

        update_poses_srv_ = this->create_service<graph_slam::srv::UpdatePoses>("/graph_slam/update_poses",
            bind(&GraphManager::onUpdatePoses, this, placeholders::_1, placeholders::_2),
            rmw_qos_profile_services_default, srv_group_);



        RCLCPP_INFO(this->get_logger(), "GraphManager initialized");
    }

private:
    vector<Edge> edges_;
    vector<PoseNode> nodes_;

    rclcpp::CallbackGroup::SharedPtr srv_group_;

    rclcpp::Service<graph_slam::srv::AddNode>::SharedPtr add_node_srv_;
    rclcpp::Service<graph_slam::srv::AddEdge>::SharedPtr add_edge_srv_;
    rclcpp::Service<graph_slam::srv::GetGraph>::SharedPtr get_graph_srv_;
    rclcpp::Service<graph_slam::srv::UpdatePoses>::SharedPtr update_poses_srv_;


    void onAddNode( shared_ptr<graph_slam::srv::AddNode::Request> req, shared_ptr<graph_slam::srv::AddNode::Response> res)
    {
        PoseNode n;
        n.id = nodes_.size();
        n.x = req->x;
        n.y = req->y;
        n.theta = req->theta;
        n.scan = req->scan;
        n.timestamp = req->timestamp;

        nodes_.push_back(n);
        res->id = n.id;
        res->success = true;
    }

    void onAddEdge( shared_ptr<graph_slam::srv::AddEdge::Request> req, shared_ptr<graph_slam::srv::AddEdge::Response> res)
    {
        if (req->from_id >= nodes_.size() || req->to_id >= nodes_.size()) {
            res->success = false;
            return;
        }

        for (int i = 0; i < edges_.size(); ++i) {
            if (edges_[i].from_id == req->from_id && edges_[i].to_id == req->to_id) {
                res->success = false;
                return;
            }
        }

        Edge e;
        e.from_id = req->from_id;
        e.to_id = req->to_id;
        e.type = static_cast<Edge::Type>(req->type);
        e.dx = req->dx;
        e.dy = req->dy;
        e.dtheta = req->dtheta;

        if (req->information.size() >= 9) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    e.information(r, c) = req->information[r * 3 + c];
        } else {
            e.information = Eigen::Matrix3d::Identity() * 100.0;
        }

        edges_.push_back(e);
        res->success = true;
    }


    void onGetGraph( shared_ptr<graph_slam::srv::GetGraph::Request> req, shared_ptr<graph_slam::srv::GetGraph::Response> res)
    {
        for (int i = 0; i < nodes_.size(); ++i) {
            res->ids.push_back(nodes_[i].id);
            res->xs.push_back(nodes_[i].x);
            res->ys.push_back(nodes_[i].y);
            res->thetas.push_back(nodes_[i].theta);
            res->timestamps.push_back(nodes_[i].timestamp);
            res->scans.push_back(nodes_[i].scan);
        }

        for (int i = 0; i < edges_.size(); ++i) {
            res->edges_from_id.push_back(edges_[i].from_id);
            res->edges_to_id.push_back(edges_[i].to_id);
            res->edges_type.push_back(edges_[i].type);
            res->edges_dx.push_back(edges_[i].dx);
            res->edges_dy.push_back(edges_[i].dy);
            res->edges_dtheta.push_back(edges_[i].dtheta);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    res->edges_information.push_back(edges_[i].information(r, c));
        }
    }

    void onUpdatePoses( shared_ptr<graph_slam::srv::UpdatePoses::Request> req, shared_ptr<graph_slam::srv::UpdatePoses::Response> res)
    {
        if (req->ids.size() != req->xs.size() || req->ids.size() != req->ys.size() || req->ids.size() != req->thetas.size()) {
            res->success = false;
            return;
        }

        for (int i = 0; i < req->ids.size(); ++i) {
            int id = req->ids[i];
            if (id >= nodes_.size()) {
                res->success = false;
                return;
            }
            nodes_[id].x = req->xs[i];
            nodes_[id].y = req->ys[i];
            nodes_[id].theta = req->thetas[i];
        }

        res->success = true;
    }
};
}

#ifndef GRAPH_SLAM_TESTING
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto manager = std::make_shared<graph_slam::GraphManager>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(manager);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
#endif