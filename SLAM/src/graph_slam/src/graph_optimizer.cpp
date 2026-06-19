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
#include <vector>
#include <set>
#include <tuple>

using namespace std;
namespace graph_slam {

using BlockSolver  = g2o::BlockSolver<g2o::BlockSolverTraits<3,3>>;
using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

static g2o::SparseOptimizer* makeOptimizer()
{
    auto lin =  make_unique<LinearSolver>();
    lin->setBlockOrdering(false);
    auto blk =  make_unique<BlockSolver>( move(lin));
    auto* alg = new g2o::OptimizationAlgorithmLevenberg( move(blk));
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
        optimize_every_n_     = this->declare_parameter("optimize_every_n",     10);
        maximum_iterations_   = this->declare_parameter("maximum_iterations",   50);
        check_period_sec_     = this->declare_parameter("check_period_sec",     2.0);
        huber_delta_          = this->declare_parameter("huber_delta",          0.3);
        phase1_iterations_    = this->declare_parameter("phase1_iterations",    30);
        lc_chi2_threshold_    = this->declare_parameter("lc_chi2_threshold",    7.81); // 95% confidence for 3 DOF
        do_outlier_rejection_ = this->declare_parameter("do_outlier_rejection", true);
        min_lc_to_optimize_   = this->declare_parameter("min_lc_to_optimize",   2);

        cb_group_reentrant_    = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        cb_group_get_graph_    = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_update_poses_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        clt_get_graph_    = this->create_client<graph_slam::srv::GetGraph>   ("graph_slam/get_graph",    rmw_qos_profile_services_default, cb_group_get_graph_);
        clt_update_poses_ = this->create_client<graph_slam::srv::UpdatePoses>("graph_slam/update_poses", rmw_qos_profile_services_default, cb_group_update_poses_);
        pub_trigger_remap_= this->create_publisher<std_msgs::msg::Bool>("graph_slam/trigger_remap", 1);
        clt_get_graph_->wait_for_service();
        clt_update_poses_->wait_for_service();
        last_node_count_ = 0;
        last_loop_closure_count_ = 0;

        RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] every=%d iters=%d+%d huber=%.2f chi2_thr=%.2f min_lc=%d",
                 optimize_every_n_, phase1_iterations_, maximum_iterations_,
                 huber_delta_, lc_chi2_threshold_, min_lc_to_optimize_);

        timer_ = this->create_wall_timer(
             chrono::duration<double>(check_period_sec_),
             bind(&GraphOptimizer::timerCallback, this),
            cb_group_reentrant_);
    }

private:
    rclcpp::Client<graph_slam::srv::GetGraph>::SharedPtr    clt_get_graph_;
    rclcpp::Client<graph_slam::srv::UpdatePoses>::SharedPtr clt_update_poses_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr        pub_trigger_remap_;
    rclcpp::TimerBase::SharedPtr                             timer_;

    rclcpp::CallbackGroup::SharedPtr cb_group_reentrant_;
    rclcpp::CallbackGroup::SharedPtr cb_group_get_graph_;
    rclcpp::CallbackGroup::SharedPtr cb_group_update_poses_;

    int    optimize_every_n_, maximum_iterations_, phase1_iterations_;
    int    last_node_count_, last_loop_closure_count_;
    int    min_lc_to_optimize_;
    double check_period_sec_, huber_delta_, lc_chi2_threshold_;
    bool   do_outlier_rejection_;

    void addVertices(g2o::SparseOptimizer* opt, graph_slam::srv::GetGraph::Response& resp)
    {
        
        for (int i = 0; i < resp.ids.size(); ++i) {
            g2o::VertexSE2* v = new g2o::VertexSE2();
            v->setId(resp.ids[i]);
            v->setEstimate(g2o::SE2(resp.xs[i], resp.ys[i], resp.thetas[i]));
            if (i == 0) v->setFixed(true);
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

    tuple<int,int,int> addEdges(g2o::SparseOptimizer* opt,graph_slam::srv::GetGraph::Response& resp,bool include_lc,vector<pair<g2o::EdgeSE2*, int>>* lc_edge_map = nullptr)
    {
        int odometry_count = 0, loop_closure_count = 0, skipped_count = 0;
        for (int i = 0; i < resp.edges_from_id.size(); ++i) {
            bool is_lc = (resp.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE));
            if (is_lc && !include_lc) 
                continue;

            g2o::EdgeSE2* e = new g2o::EdgeSE2();
            e->vertices()[0] = opt->vertex(resp.edges_from_id[i]);
            e->vertices()[1] = opt->vertex(resp.edges_to_id[i]);
            if (!e->vertices()[0] || !e->vertices()[1]) 
                { delete e; 
                  ++skipped_count; 
                  continue; 
                }

            e->setMeasurement(g2o::SE2(resp.edges_dx[i], resp.edges_dy[i],resp.edges_dtheta[i]));
            e->setInformation(buildInfo(resp, i));

            if (is_lc ) {
                auto* k = new g2o::RobustKernelHuber();
                k->setDelta(huber_delta_);
                e->setRobustKernel(k);
                ++loop_closure_count;
                if (lc_edge_map) 
                    lc_edge_map->push_back({e, i});
            } 
            else 
                ++odometry_count;
            opt->addEdge(e);
        }
        return {odometry_count, loop_closure_count, skipped_count};
    }

    void publishPoses(g2o::SparseOptimizer* opt,graph_slam::srv::GetGraph::Response& resp,int N_nodes, int N_nodes_val, int loop_closure_count)
    {
        auto update_req = make_shared<graph_slam::srv::UpdatePoses::Request>();
        for (int i = 0; i < N_nodes; ++i) {
            auto* v = dynamic_cast<g2o::VertexSE2*>(opt->vertex(resp.ids[i]));
            if (!v) 
                continue;
            g2o::SE2 est = v->estimate();

            update_req->ids.push_back(resp.ids[i]);
            update_req->xs.push_back(est.translation().x());
            update_req->ys.push_back(est.translation().y());
            update_req->thetas.push_back(est.rotation().angle());
        }

        auto future = clt_update_poses_->async_send_request(update_req);
        if (future.wait_for( chrono::seconds(5)) ==  future_status::ready) {
            auto result = future.get();
            if (result->success) {
                last_node_count_ = N_nodes_val;
                last_loop_closure_count_ = loop_closure_count;
                RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Poses updated, triggering remap");
                std_msgs::msg::Bool msg; 
                msg.data = true; 
                pub_trigger_remap_->publish(msg);
            } else {
                RCLCPP_ERROR(this->get_logger(), "[GraphOptimizer] Failed to update poses");
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "[GraphOptimizer] Failed to update poses");
        }
    }

    void timerCallback()
    {
        auto req =  make_shared<graph_slam::srv::GetGraph::Request>();
        auto future = clt_get_graph_->async_send_request(req);
        if (future.wait_for( chrono::seconds(5)) !=  future_status::ready) 
            return;

        graph_slam::srv::GetGraph::Response srv_response = *future.get();

        int N_nodes = srv_response.ids.size();
        int N_edges = srv_response.edges_from_id.size();
        if (N_nodes < 2) 
            return;

        int loop_closure_count = 0;
        for (int i = 0; i < N_edges; ++i)
            if (srv_response.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE))
                ++loop_closure_count;

        bool new_lc       = (loop_closure_count > last_loop_closure_count_);
        bool enough_nodes = (N_nodes >= last_node_count_ + optimize_every_n_);

        if (!new_lc && !enough_nodes) 
            return;

        if (new_lc && loop_closure_count < min_lc_to_optimize_) {
            RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Only %d LC, waiting for %d before optimizing with LC",loop_closure_count, min_lc_to_optimize_);
            if (!enough_nodes) 
                return;
        }

        bool use_lc = (loop_closure_count >= min_lc_to_optimize_);

        RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] %d nodes %d edges %d LC  use_lc=%d",
                 N_nodes, N_edges, loop_closure_count, use_lc);

        if (use_lc) {
             unique_ptr<g2o::SparseOptimizer> opt1(makeOptimizer());
            addVertices(opt1.get(), srv_response);
            auto [o1, l1, s1] = addEdges(opt1.get(), srv_response, false);
            if (o1 > 0) {
                opt1->initializeOptimization();
                opt1->optimize(phase1_iterations_);
                for (int i = 0; i < N_nodes; ++i) {
                    auto* v = dynamic_cast<g2o::VertexSE2*>(opt1->vertex(srv_response.ids[i]));
                    if (!v) 
                        continue;
                    srv_response.xs[i]     = v->estimate().translation().x();
                    srv_response.ys[i]     = v->estimate().translation().y();
                    srv_response.thetas[i] = v->estimate().rotation().angle();
                }
                RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Phase1: %d odom edges chi2=%.3f", o1, opt1->chi2());
            }
        }

        vector<pair<g2o::EdgeSE2*, int>> lc_edge_map;
         unique_ptr<g2o::SparseOptimizer> opt2(makeOptimizer());
        addVertices(opt2.get(), srv_response);
        auto [o2, l2, s2] = addEdges(opt2.get(), srv_response, use_lc, &lc_edge_map);

        RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Phase2: %d odom %d LC %d skipped", o2, l2, s2);
        opt2->initializeOptimization();
        int iters = opt2->optimize(maximum_iterations_);
        RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Phase2 done: %d iters chi2=%.3f", iters, opt2->chi2());

        if (do_outlier_rejection_ && l2 > 0) {
            vector<int> outlier_indices;
            for (auto& [e_ptr, orig_idx] : lc_edge_map) {
                e_ptr->setRobustKernel(nullptr);
                e_ptr->computeError();
                double chi2 = e_ptr->chi2();
                if (chi2 > lc_chi2_threshold_) {
                    outlier_indices.push_back(orig_idx);
                    RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] LC outlier edge %d<->%d chi2=%.2f > %.2f",srv_response.edges_from_id[orig_idx],srv_response.edges_to_id[orig_idx],chi2, lc_chi2_threshold_);
                }
            }

            if (!outlier_indices.empty()) {
                RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Phase3: removing %d outlier LC edges",(int)outlier_indices.size());

                set<int> outlier_set(outlier_indices.begin(), outlier_indices.end());

                for (int i = 0; i < N_nodes; ++i) {
                    auto* v = dynamic_cast<g2o::VertexSE2*>(opt2->vertex(srv_response.ids[i]));
                    if (!v) 
                        continue;
                    srv_response.xs[i]     = v->estimate().translation().x();
                    srv_response.ys[i]     = v->estimate().translation().y();
                    srv_response.thetas[i] = v->estimate().rotation().angle();
                }

                 unique_ptr<g2o::SparseOptimizer> opt3(makeOptimizer());
                addVertices(opt3.get(), srv_response);

                int N_e = srv_response.edges_from_id.size();
                int o3 = 0, l3 = 0;
                for (int i = 0; i < N_e; ++i) {
                    bool is_lc = (srv_response.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE));
                    if (is_lc && outlier_set.count(i)) 
                        continue;

                    auto* e = new g2o::EdgeSE2();
                    e->vertices()[0] = opt3->vertex(srv_response.edges_from_id[i]);
                    e->vertices()[1] = opt3->vertex(srv_response.edges_to_id[i]);
                    if (!e->vertices()[0] || !e->vertices()[1]) { delete e; continue; }
                    e->setMeasurement(g2o::SE2(srv_response.edges_dx[i],
                                               srv_response.edges_dy[i],
                                               srv_response.edges_dtheta[i]));
                    e->setInformation(buildInfo(srv_response, i));
                    if (is_lc) {
                        auto* k = new g2o::RobustKernelHuber();
                        k->setDelta(huber_delta_);
                        e->setRobustKernel(k);
                        ++l3;
                    } else ++o3;
                    opt3->addEdge(e);
                }

                opt3->initializeOptimization();
                int iters3 = opt3->optimize(maximum_iterations_);
                RCLCPP_INFO(this->get_logger(), "[GraphOptimizer] Phase3 done: %d iters chi2=%.3f (%d odom %d LC kept)",
                         iters3, opt3->chi2(), o3, l3);

                publishPoses(opt3.get(), srv_response, N_nodes, N_nodes, loop_closure_count);
                return;
            }
        }

        publishPoses(opt2.get(), srv_response, N_nodes, N_nodes, loop_closure_count);
    }
};

}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node =  make_shared<graph_slam::GraphOptimizer>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}