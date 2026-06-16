#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <graph_slam/GetGraph.h>
#include <graph_slam/UpdatePoses.h>
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

using namespace std;
namespace graph_slam {

using BlockSolver  = g2o::BlockSolver<g2o::BlockSolverTraits<3,3>>;
using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

static g2o::SparseOptimizer* makeOptimizer()
{
    auto lin = std::make_unique<LinearSolver>();
    lin->setBlockOrdering(false);
    auto blk = std::make_unique<BlockSolver>(std::move(lin));
    auto* alg = new g2o::OptimizationAlgorithmLevenberg(std::move(blk));
    auto* opt = new g2o::SparseOptimizer();
    opt->setAlgorithm(alg);
    opt->setVerbose(false);
    return opt;
}

class GraphOptimizer {
public:
    GraphOptimizer(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh)
    {
        private_nh_.param("optimize_every_n",     optimize_every_n_,     10);
        private_nh_.param("maximum_iterations",   maximum_iterations_,   50);
        private_nh_.param("check_period_sec",     check_period_sec_,     2.0);
        private_nh_.param("huber_delta",          huber_delta_,          0.3);
        private_nh_.param("phase1_iterations",    phase1_iterations_,    30);
        private_nh_.param("lc_chi2_threshold",    lc_chi2_threshold_,    15.0);
        private_nh_.param("do_outlier_rejection", do_outlier_rejection_, true);
        private_nh_.param("min_lc_to_optimize",   min_lc_to_optimize_,   2);

        clt_get_graph_    = nh_.serviceClient<graph_slam::GetGraph>   ("graph_slam/get_graph");
        clt_update_poses_ = nh_.serviceClient<graph_slam::UpdatePoses>("graph_slam/update_poses");
        pub_trigger_remap_= nh_.advertise<std_msgs::Bool>("graph_slam/trigger_remap", 1);
        clt_get_graph_.waitForExistence();
        clt_update_poses_.waitForExistence();
        last_node_count_ = 0;
        last_lc_count_   = 0;

        ROS_INFO("[GraphOptimizer] every=%d iters=%d+%d huber=%.2f chi2_thr=%.2f min_lc=%d",
                 optimize_every_n_, phase1_iterations_, maximum_iterations_,
                 huber_delta_, lc_chi2_threshold_, min_lc_to_optimize_);

        timer_ = nh_.createTimer(ros::Duration(check_period_sec_),
                                 &GraphOptimizer::timerCallback, this);
    }

private:
    ros::NodeHandle    nh_, private_nh_;
    ros::ServiceClient clt_get_graph_, clt_update_poses_;
    ros::Publisher     pub_trigger_remap_;
    ros::Timer         timer_;

    int    optimize_every_n_, maximum_iterations_, phase1_iterations_;
    int    last_node_count_, last_lc_count_;
    int    min_lc_to_optimize_;
    double check_period_sec_, huber_delta_, lc_chi2_threshold_;
    bool   do_outlier_rejection_;

    void addVertices(g2o::SparseOptimizer* opt,
                     const graph_slam::GetGraph::Response& resp)
    {
        int N = resp.ids.size();
        for (int i = 0; i < N; ++i) {
            auto* v = new g2o::VertexSE2();
            v->setId(resp.ids[i]);
            v->setEstimate(g2o::SE2(resp.xs[i], resp.ys[i], resp.thetas[i]));
            if (i == 0) v->setFixed(true);
            opt->addVertex(v);
        }
    }

    Eigen::Matrix3d buildInfo(const graph_slam::GetGraph::Response& resp, int ei)
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

    tuple<int,int,int> addEdges(
        g2o::SparseOptimizer* opt,
        const graph_slam::GetGraph::Response& resp,
        bool include_lc,
        bool use_robust,
        vector<pair<g2o::EdgeSE2*, int>>* lc_edge_map = nullptr)
    {
        int odom_ok = 0, lc_ok = 0, skipped = 0;
        int N_edges = resp.edges_from_id.size();
        for (int i = 0; i < N_edges; ++i) {
            bool is_lc = (resp.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE));
            if (is_lc && !include_lc) continue;

            auto* e = new g2o::EdgeSE2();
            e->vertices()[0] = opt->vertex(resp.edges_from_id[i]);
            e->vertices()[1] = opt->vertex(resp.edges_to_id[i]);
            if (!e->vertices()[0] || !e->vertices()[1]) { delete e; ++skipped; continue; }

            e->setMeasurement(g2o::SE2(resp.edges_dx[i], resp.edges_dy[i],
                                       resp.edges_dtheta[i]));
            e->setInformation(buildInfo(resp, i));

            if (is_lc && use_robust) {
                auto* k = new g2o::RobustKernelHuber();
                k->setDelta(huber_delta_);
                e->setRobustKernel(k);
            }
            opt->addEdge(e);

            if (is_lc) {
                ++lc_ok;
                if (lc_edge_map) lc_edge_map->push_back({e, i});
            } else ++odom_ok;
        }
        return {odom_ok, lc_ok, skipped};
    }

    void publishPoses(g2o::SparseOptimizer* opt,
                      const graph_slam::GetGraph::Response& resp,
                      int N_nodes, int N_nodes_val, int lc_count)
    {
        graph_slam::UpdatePoses update_srv;
        for (int i = 0; i < N_nodes; ++i) {
            auto* v = dynamic_cast<g2o::VertexSE2*>(opt->vertex(resp.ids[i]));
            if (!v) continue;
            g2o::SE2 est = v->estimate();
            update_srv.request.ids.push_back(resp.ids[i]);
            update_srv.request.xs.push_back(est.translation().x());
            update_srv.request.ys.push_back(est.translation().y());
            update_srv.request.thetas.push_back(est.rotation().angle());
        }
        if (clt_update_poses_.call(update_srv) && update_srv.response.success) {
            last_node_count_ = N_nodes_val;
            last_lc_count_   = lc_count;
            ROS_INFO("[GraphOptimizer] Poses updated, triggering remap");
            std_msgs::Bool msg; msg.data = true; pub_trigger_remap_.publish(msg);
        } else {
            ROS_ERROR("[GraphOptimizer] Failed to update poses");
        }
    }

    void timerCallback(const ros::TimerEvent&)
    {
        graph_slam::GetGraph srv;
        if (!clt_get_graph_.call(srv)) return;
        int N_nodes = srv.response.ids.size();
        int N_edges = srv.response.edges_from_id.size();
        if (N_nodes < 2) return;

        int lc_count = 0;
        for (int i = 0; i < N_edges; ++i)
            if (srv.response.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE))
                ++lc_count;

        bool new_lc       = (lc_count > last_lc_count_);
        bool enough_nodes = (N_nodes >= last_node_count_ + optimize_every_n_);

        if (!new_lc && !enough_nodes) return;

        if (new_lc && lc_count < min_lc_to_optimize_) {
            ROS_INFO("[GraphOptimizer] Only %d LC, waiting for %d before optimizing with LC",
                     lc_count, min_lc_to_optimize_);
            if (!enough_nodes) return;
        }

        bool use_lc = (lc_count >= min_lc_to_optimize_);

        ROS_INFO("[GraphOptimizer] %d nodes %d edges %d LC  use_lc=%d",
                 N_nodes, N_edges, lc_count, use_lc);

        if (use_lc) {
            std::unique_ptr<g2o::SparseOptimizer> opt1(makeOptimizer());
            addVertices(opt1.get(), srv.response);
            auto [o1, l1, s1] = addEdges(opt1.get(), srv.response, false, false);
            if (o1 > 0) {
                opt1->initializeOptimization();
                opt1->optimize(phase1_iterations_);
                for (int i = 0; i < N_nodes; ++i) {
                    auto* v = dynamic_cast<g2o::VertexSE2*>(opt1->vertex(srv.response.ids[i]));
                    if (!v) continue;
                    srv.response.xs[i]     = v->estimate().translation().x();
                    srv.response.ys[i]     = v->estimate().translation().y();
                    srv.response.thetas[i] = v->estimate().rotation().angle();
                }
                ROS_INFO("[GraphOptimizer] Phase1: %d odom edges chi2=%.3f", o1, opt1->chi2());
            }
        }

        vector<pair<g2o::EdgeSE2*, int>> lc_edge_map;
        std::unique_ptr<g2o::SparseOptimizer> opt2(makeOptimizer());
        addVertices(opt2.get(), srv.response);
        auto [o2, l2, s2] = addEdges(opt2.get(), srv.response, use_lc, true, &lc_edge_map);

        ROS_INFO("[GraphOptimizer] Phase2: %d odom %d LC %d skipped", o2, l2, s2);
        opt2->initializeOptimization();
        int iters = opt2->optimize(maximum_iterations_);
        ROS_INFO("[GraphOptimizer] Phase2 done: %d iters chi2=%.3f", iters, opt2->chi2());

        if (do_outlier_rejection_ && l2 > 0) {
            vector<int> outlier_indices;
            for (auto& [e_ptr, orig_idx] : lc_edge_map) {
                e_ptr->setRobustKernel(nullptr);
                e_ptr->computeError();
                double chi2 = e_ptr->chi2();
                if (chi2 > lc_chi2_threshold_) {
                    outlier_indices.push_back(orig_idx);
                    ROS_INFO("[GraphOptimizer] LC outlier edge %d<->%d chi2=%.2f > %.2f",
                             srv.response.edges_from_id[orig_idx],
                             srv.response.edges_to_id[orig_idx],
                             chi2, lc_chi2_threshold_);
                }
            }

            if (!outlier_indices.empty()) {
                ROS_INFO("[GraphOptimizer] Phase3: removing %d outlier LC edges",
                         (int)outlier_indices.size());

                set<int> outlier_set(outlier_indices.begin(), outlier_indices.end());

                for (int i = 0; i < N_nodes; ++i) {
                    auto* v = dynamic_cast<g2o::VertexSE2*>(opt2->vertex(srv.response.ids[i]));
                    if (!v) continue;
                    srv.response.xs[i]     = v->estimate().translation().x();
                    srv.response.ys[i]     = v->estimate().translation().y();
                    srv.response.thetas[i] = v->estimate().rotation().angle();
                }

                std::unique_ptr<g2o::SparseOptimizer> opt3(makeOptimizer());
                addVertices(opt3.get(), srv.response);

                int N_e = srv.response.edges_from_id.size();
                int o3 = 0, l3 = 0;
                for (int i = 0; i < N_e; ++i) {
                    bool is_lc = (srv.response.edges_type[i] == static_cast<int>(Edge::LOOP_CLOSURE));
                    if (is_lc && outlier_set.count(i)) continue;

                    auto* e = new g2o::EdgeSE2();
                    e->vertices()[0] = opt3->vertex(srv.response.edges_from_id[i]);
                    e->vertices()[1] = opt3->vertex(srv.response.edges_to_id[i]);
                    if (!e->vertices()[0] || !e->vertices()[1]) { delete e; continue; }
                    e->setMeasurement(g2o::SE2(srv.response.edges_dx[i],
                                               srv.response.edges_dy[i],
                                               srv.response.edges_dtheta[i]));
                    e->setInformation(buildInfo(srv.response, i));
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
                ROS_INFO("[GraphOptimizer] Phase3 done: %d iters chi2=%.3f (%d odom %d LC kept)",
                         iters3, opt3->chi2(), o3, l3);

                publishPoses(opt3.get(), srv.response, N_nodes, N_nodes, lc_count);
                return;
            }
        }

        publishPoses(opt2.get(), srv.response, N_nodes, N_nodes, lc_count);
    }
};

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "graph_optimizer");
    ros::NodeHandle nh, pnh("~");
    graph_slam::GraphOptimizer opt(nh, pnh);
    ros::spin();
    return 0;
}