#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include "graph_slam/srv/get_graph.hpp"
#include "graph_slam/srv/update_poses.hpp"
#include "types.hpp"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam2d/types_slam2d.h>

#include <memory>

using namespace std;
namespace graph_slam {

using BlockSolver  = g2o::BlockSolver<g2o::BlockSolverTraits<3,3>>;
using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

static g2o::SparseOptimizer* makeOptimizer()
{
    auto lin = make_unique<LinearSolver>();
    lin->setBlockOrdering(false);
    auto blk = make_unique<BlockSolver>(move(lin));
    auto* alg = new g2o::OptimizationAlgorithmLevenberg(move(blk));
    auto* opt = new g2o::SparseOptimizer();
    opt->setAlgorithm(alg);
    opt->setVerbose(false);
    return opt;
}

class GraphOptimizer : public rclcpp::Node {
public:
    GraphOptimizer()
    : Node("graph_optimizer")
    {
        optimize_every_n_ = declare_parameter("optimize_every_n", 10);
        maximum_iterations_ = declare_parameter("maximum_iterations", 50);
        check_period_sec_  = declare_parameter("check_period_sec", 2.0);
        huber_delta_ = declare_parameter("huber_delta",0.3);

        cb_group_reentrant_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        cb_group_get_graph_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_update_poses_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        clt_get_graph_ = create_client<graph_slam::srv::GetGraph>   ("graph_slam/get_graph",rmw_qos_profile_services_default, cb_group_get_graph_);
        clt_update_poses_ = create_client<graph_slam::srv::UpdatePoses>("graph_slam/update_poses",rmw_qos_profile_services_default, cb_group_update_poses_);
        pub_trigger_remap_= create_publisher<std_msgs::msg::Bool>("graph_slam/trigger_remap", 1);
        timer_ = create_wall_timer(chrono::duration<double>(check_period_sec_),[this](){ onTimer(); },cb_group_reentrant_);

        last_node_count_         = 0;
        last_loop_closure_count_ = 0;

        RCLCPP_INFO(this->get_logger(),
            "[GraphOptimizer] every=%d iters=%d huber=%.2f",
            optimize_every_n_, maximum_iterations_,huber_delta_);
  
    }

private:
    rclcpp::Client<graph_slam::srv::GetGraph>::SharedPtr clt_get_graph_;
    rclcpp::Client<graph_slam::srv::UpdatePoses>::SharedPtr clt_update_poses_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_trigger_remap_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::CallbackGroup::SharedPtr cb_group_reentrant_;
    rclcpp::CallbackGroup::SharedPtr cb_group_get_graph_;
    rclcpp::CallbackGroup::SharedPtr cb_group_update_poses_;

    int optimize_every_n_, maximum_iterations_;
    int last_node_count_, last_loop_closure_count_;
    double check_period_sec_, huber_delta_;
    atomic<bool> optimization_in_flight_{false};

    void addVertices(g2o::SparseOptimizer* opt, graph_slam::srv::GetGraph::Response& resp)
    {
        for (int i = 0; i < resp.ids.size(); ++i) {
            g2o::VertexSE2* v = new g2o::VertexSE2();
            v->setId(resp.ids[i]);
            v->setEstimate(g2o::SE2(resp.xs[i], resp.ys[i], resp.thetas[i]));
            if (i == 0) 
            {
                v->setFixed(true);
            }
            opt->addVertex(v);
        }
    }

    Eigen::Matrix3d buildInfo(graph_slam::srv::GetGraph::Response& resp, int ei)
    {
        double MAX_XY = 400.0, MAX_YAW = 800.0;
        Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
        int base = ei * 9;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                info(r, c) = resp.edges_information[base + r*3 + c];
        info(0,0) = min(max(info(0,0), 0.0), MAX_XY);
        info(1,1) = min(max(info(1,1), 0.0), MAX_XY);
        info(2,2) = min(max(info(2,2), 0.0), MAX_YAW);
        return info;
    }

    pair<int,int> addEdges(g2o::SparseOptimizer* opt,graph_slam::srv::GetGraph::Response& resp)
    {
        int odometry_count = 0, loop_closure_count = 0;
        for (int i = 0; i < resp.edges_from_id.size(); ++i) {
            bool is_lc = (resp.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE));
           
            g2o::EdgeSE2* e = new g2o::EdgeSE2();
            e->vertices()[0] = opt->vertex(resp.edges_from_id[i]);
            e->vertices()[1] = opt->vertex(resp.edges_to_id[i]);
            if (!e->vertices()[0] || !e->vertices()[1]) {
                delete e;
                continue;
            }

            e->setMeasurement(g2o::SE2(resp.edges_dx[i], resp.edges_dy[i], resp.edges_dtheta[i]));
            e->setInformation(buildInfo(resp, i));

            if (is_lc) {
                auto* k = new g2o::RobustKernelHuber();
                k->setDelta(huber_delta_);
                e->setRobustKernel(k);
                ++loop_closure_count;
            } else {
                ++odometry_count;
            }
            opt->addEdge(e);
        }
        return {odometry_count, loop_closure_count};
    }

    void publishPoses(g2o::SparseOptimizer* opt,graph_slam::srv::GetGraph::Response& resp,int N_nodes)
    {
        auto update_req = std::make_shared<graph_slam::srv::UpdatePoses::Request>();
        for (int i = 0; i < N_nodes; ++i) {
            auto* v = dynamic_cast<g2o::VertexSE2*>(opt->vertex(resp.ids[i]));
            if (!v) 
            {
                continue;
            }
            g2o::SE2 est = v->estimate();
            update_req->ids.push_back(resp.ids[i]);
            update_req->xs.push_back(est.translation().x());
            update_req->ys.push_back(est.translation().y());
            update_req->thetas.push_back(est.rotation().angle());
        }

        clt_update_poses_->async_send_request(update_req,
            [this](rclcpp::Client<graph_slam::srv::UpdatePoses>::SharedFuture f) {
            auto result = f.get();
            if (result && result->success) {
                RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Poses updated, triggering remap");
                std_msgs::msg::Bool msg;
                msg.data = true;
                pub_trigger_remap_->publish(msg);
            } else {
                RCLCPP_ERROR(this->get_logger(), "[GraphOptimizer] Failed to update poses");
            }
        });
    }

    void runOptimization(graph_slam::srv::GetGraph::Response srv_response)
    {
        int N_nodes = srv_response.ids.size();
        int N_edges = srv_response.edges_from_id.size();

        int total_lc = 0;
        for (int i = 0; i < N_edges; ++i)
            if (srv_response.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE))
                ++total_lc;

        bool new_lc       = (total_lc > last_loop_closure_count_);
        bool enough_nodes = (N_nodes >= last_node_count_ + optimize_every_n_);

        if (!new_lc && !enough_nodes) 
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "[GraphOptimizer] %d nodes %d edges %d LC, optimizing...",
            N_nodes, N_edges, total_lc);

        g2o::SparseOptimizer* opt = makeOptimizer();
        addVertices(opt, srv_response);
        pair<int, int> counts = addEdges(opt, srv_response);
        opt->initializeOptimization();
        int iters = opt->optimize(maximum_iterations_);
        RCLCPP_INFO(this->get_logger(),
            "[GraphOptimizer] %d iters chi2=%.3f (%d odom %d LC)",
            iters, opt->chi2(), counts.first, counts.second);

        last_node_count_         = N_nodes;
        last_loop_closure_count_ = total_lc;
        publishPoses(opt, srv_response, N_nodes);
        delete opt;
    }

    void onTimer()
    {
        bool expected = false;
        if (!optimization_in_flight_.compare_exchange_strong(expected, true))
            {
                return;
            }
        auto req = std::make_shared<graph_slam::srv::GetGraph::Request>();
        clt_get_graph_->async_send_request(req,
            [this](rclcpp::Client<graph_slam::srv::GetGraph>::SharedFuture f) {
                optimization_in_flight_ = false;
            auto srv_response_ptr = f.get();
            if (!srv_response_ptr) 
            {
                return;
            }
            runOptimization(*srv_response_ptr);
        });
    }
};

}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<graph_slam::GraphOptimizer>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
