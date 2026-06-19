#include <rclcpp/rclcpp.hpp>
#include <graph_slam/srv/get_graph.hpp>
#include <graph_slam/srv/add_edge.hpp>
#include <graph_slam/srv/scan_match.hpp>
#include "types.hpp"
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
using namespace std;
namespace graph_slam {
static const int NR = 20;
static const int NS = 60;
using SCMatrix = Eigen::Matrix<double, NR, NS>;
SCMatrix buildScanContext(sensor_msgs::msg::LaserScan scan, double max_range)
{
    SCMatrix sc = SCMatrix::Zero();
    double ring_width   = max_range / NR;
    double sector_width = 2.0 * M_PI / NS;
    double angle = scan.angle_min;
    for (int i = 0; i < scan.ranges.size(); ++i )
    {
        double r = scan.ranges[i];
        if (!   isfinite(r) || r <= scan.range_min || r >= scan.range_max || r > max_range)
           {
             angle += scan.angle_increment;
             continue;
           }
        int ring =    min(static_cast<int>(r / ring_width), NR - 1);
        double a = wrapAngle(angle);
        int sec =    min(static_cast<int>(a / sector_width), NS - 1);
        if (r > sc(ring, sec)) 
        {
            sc(ring, sec) = r;
        }
        angle += scan.angle_increment;
    }
    return sc;
}

pair<double, int> scanContextDistance(SCMatrix A, SCMatrix B)
{
    SCMatrix A_hat = SCMatrix::Zero();
    SCMatrix B_hat = SCMatrix::Zero();
    for (int j = 0; j < NS; ++j)
    {
        double na = A.col(j).norm();
        double nb = B.col(j).norm();
        if (na > 1e-9) 
        {
            A_hat.col(j) = A.col(j) / na;
        }
        if (nb > 1e-9) 
        {
            B_hat.col(j) = B.col(j) / nb;
        }
    }
    double best_dist  = numeric_limits<double>::max();
    int    best_shift = 0;
    for (int phi = 0; phi < NS; ++phi)
    {
        double cosine_sum = 0.0;
        for (int j = 0; j < NS; ++j)
        {
            int j_shifted = (j + phi) % NS;
            cosine_sum += A_hat.col(j).dot(B_hat.col(j_shifted));
        }
        double dist = 1.0 - cosine_sum / static_cast<double>(NS);
        if (dist < best_dist) 
        {
            best_dist = dist;
            best_shift = phi;
        }
    }
    return { best_dist, best_shift };
}
class LoopClosure : public rclcpp::Node
{
public:
    LoopClosure() : Node("loop_closure")
    {
        max_range_           = declare_parameter("max_range",             20.0);
        min_node_separation_ = declare_parameter("min_node_separation",   5);
        num_candidates_      = declare_parameter("num_candidates",        10);
        sc_threshold_        = declare_parameter("sc_threshold",          0.35);
        icp_threshold_       = declare_parameter("icp_threshold",         0.05);
        max_lc_per_tick_     = declare_parameter("max_lc_per_tick",       3);
        period_sec_          = declare_parameter("period_sec",            5.0);
        max_icp_translation_ = declare_parameter("max_icp_translation",   2.0);
        query_window_        = declare_parameter("query_window",          10);

        cb_group_reentrant_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        cb_group_graph_     = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_edge_      = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_icp_       = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        clt_graph_ = create_client<graph_slam::srv::GetGraph> ("graph_slam/get_graph",  rmw_qos_profile_services_default, cb_group_graph_);
        clt_edge_  = create_client<graph_slam::srv::AddEdge>  ("graph_slam/add_edge",   rmw_qos_profile_services_default, cb_group_edge_);
        clt_icp_   = create_client<graph_slam::srv::ScanMatch>("graph_slam/scan_match", rmw_qos_profile_services_default, cb_group_icp_);
        clt_graph_->wait_for_service();
        clt_edge_->wait_for_service();
        clt_icp_->wait_for_service();

        timer_ = create_wall_timer(
               chrono::duration<double>(period_sec_),
               bind(&LoopClosure::tick, this),
            cb_group_reentrant_);

        RCLCPP_INFO(get_logger(),
            "[LoopClosure] max_range=%.1f  min_sep=%d  candidates=%d  "
            "sc_thr=%.2f  icp_thr=%.2f  max_lc=%d  query_window=%d  max_trans=%.1f",
            max_range_, min_node_separation_, num_candidates_,
            sc_threshold_, icp_threshold_, max_lc_per_tick_,
            query_window_, max_icp_translation_);
    }
private:
    rclcpp::Client<graph_slam::srv::GetGraph>::SharedPtr  clt_graph_;
    rclcpp::Client<graph_slam::srv::AddEdge>::SharedPtr   clt_edge_;
    rclcpp::Client<graph_slam::srv::ScanMatch>::SharedPtr clt_icp_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::CallbackGroup::SharedPtr cb_group_reentrant_;
    rclcpp::CallbackGroup::SharedPtr cb_group_graph_;
    rclcpp::CallbackGroup::SharedPtr cb_group_edge_;
    rclcpp::CallbackGroup::SharedPtr cb_group_icp_;

    double max_range_;
    int    min_node_separation_;
    int    num_candidates_;
    double sc_threshold_;
    double icp_threshold_;
    int    max_lc_per_tick_;
    double period_sec_;
    double max_icp_translation_;
    int    query_window_;

    vector<SCMatrix>                     sc_cache_;
    vector<Eigen::Matrix<double, NR, 1>> rk_cache_;
    map<pair<int,int>>     added_lc_pairs_;

    static Eigen::Matrix<double, NR, 1> ringKey(SCMatrix& sc)
    {
        Eigen::Matrix<double, NR, 1> k;
        for (int i = 0; i < NR; ++i)
        {
            int nonzero = 0;
            for (int j = 0; j < NS; ++j)
               {
                 if (sc(i, j) > 1e-9) 
                 {
                     ++nonzero;
                 }
               }
            k(i) = static_cast<double>(nonzero) / NS;
        }
        return k;
    }

    void updateCaches(vector<PoseNode>& nodes)
    {
        for (int i = sc_cache_.size();i < nodes.size(); ++i)
        {
            sc_cache_.push_back(buildScanContext(nodes[i].scan, max_range_));
            rk_cache_.push_back(ringKey(sc_cache_.back()));
        }
    }

    vector<int> ringKeyTopK(int q, int num_refs)
    {
        struct Hit { int ref; double dist; };
        vector<Hit> hits;
        hits.reserve(num_refs);
        for (int ref = 0; ref < num_refs; ++ref)
        {
            double d = (rk_cache_[q] - rk_cache_[ref]).norm();
            hits.push_back({ ref, d });
        }
        int K = min(num_candidates_, static_cast<int>(hits.size()));
        partial_sort(hits.begin(), hits.begin() + K, hits.end(),
                         [](Hit& a, Hit& b) { return a.dist < b.dist; });
        vector<int> topk;
        topk.reserve(K);
        for (int i = 0; i < K; ++i)
             topk.push_back(hits[i].ref);
        return topk;
    }

    bool tryLoopClosure(int q, int num_refs, vector<PoseNode>& nodes)
    {
        vector<int> candidates = ringKeyTopK(q, num_refs);
        double best_sc    = numeric_limits<double>::max();
        int    best_ref   = -1;
        int   best_shift = 0;
        for (int i=0; i < candidates.size(); ++i)
        {
            auto [sc_dist, shift] = scanContextDistance(sc_cache_[q], sc_cache_[candidates[i]]);
            if (sc_dist < best_sc) 
            { best_sc = sc_dist;
              best_ref = candidates[i]; 
              best_shift = shift;
            }
        }
        if (best_ref < 0 || best_sc >= sc_threshold_) 
        {
            return false;
        }

        pair<int, int> lc_pair(nodes[best_ref].id, nodes[q].id);
        auto it = added_lc_pairs_.find(lc_pair);
        if (it != added_lc_pairs_.end()) {
                return false; 
        }

        double wdx = nodes[q].x - nodes[best_ref].x;
        double wdy = nodes[q].y - nodes[best_ref].y;
        double cr  = cos(nodes[best_ref].theta);
        double sr  = sin(nodes[best_ref].theta);
        double sector_width   = 2.0 * M_PI / NS;
        double shift_angle    = best_shift * sector_width;
        double raw_dtheta     = wrapAngle(nodes[q].theta - nodes[best_ref].theta);
        double init_dtheta    = wrapAngle(raw_dtheta - shift_angle);

        auto sm_req =    make_shared<graph_slam::srv::ScanMatch::Request>();
        sm_req->reference_scan = nodes[best_ref].scan;
        sm_req->current_scan   = nodes[q].scan;
        sm_req->init_dx        =  cr * wdx + sr * wdy;
        sm_req->init_dy        = -sr * wdx + cr * wdy;
        sm_req->init_dtheta    = init_dtheta;

        auto sm_future = clt_icp_->async_send_request(sm_req);
        if (sm_future.wait_for(chrono::seconds(5)) !=    future_status::ready) 
       {
         return false;
       }
        auto sm_res = sm_future.get();
        if (!sm_res->success) 
        {
            return false;
        }
        if (sm_res->score > icp_threshold_) 
        {
            return false;
        }

        double icp_trans = sqrt(sm_res->dx * sm_res->dx + sm_res->dy * sm_res->dy);
        if (icp_trans > max_icp_translation_)
        {
            RCLCPP_DEBUG(get_logger(), "[LC] REJECTED implausible translation: %.2fm > %.2fm",
                          icp_trans, max_icp_translation_);
            return false;
        }

        auto ae_req =    make_shared<graph_slam::srv::AddEdge::Request>();
        ae_req->from_id = nodes[best_ref].id;
        ae_req->to_id   = nodes[q].id;
        ae_req->type    = Edge::LOOP_CLOSURE;
        ae_req->dx      = sm_res->dx;
        ae_req->dy      = sm_res->dy;
        ae_req->dtheta  = sm_res->dtheta;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                ae_req->information.push_back(sm_res->information[r * 3 + c]);

        auto ae_future = clt_edge_->async_send_request(ae_req);
        if ( ae_future.wait_for(   chrono::seconds(5)) !=    future_status::ready) return false;
        auto ae_res = ae_future.get();
        if (ae_res->success)
        {
            added_lc_pairs_[lc_pair] = this->get_clock()->now();
            RCLCPP_INFO(get_logger(), "[LC] ACCEPTED  %d <-> %d   sc=%.3f  icp=%.4f  trans=%.3fm",
                         nodes[best_ref].id, nodes[q].id,
                         best_sc, sm_res->score, icp_trans);
            return true;
        }
        return false;
    }

    void tick()
    {
        auto req =    make_shared<graph_slam::srv::GetGraph::Request>();
        auto future = clt_graph_->async_send_request(req);
        if (future.wait_for(   chrono::seconds(5)) !=    future_status::ready) 
        {
            return;
        }
        auto srv_response = future.get();

        int N = static_cast<int>(srv_response->ids.size());
        if (N < min_node_separation_ + 2)
        {
            return;
        }

        vector<PoseNode> nodes(N);
        for (int i = 0; i < N; ++i)
        {
            nodes[i].id    = srv_response->ids[i];
            nodes[i].x     = srv_response->xs[i];
            nodes[i].y     = srv_response->ys[i];
            nodes[i].theta = srv_response->thetas[i];
            nodes[i].scan  = srv_response->scans[i];
        }

        updateCaches(nodes);

        int lc_added = 0;
        int q_start = max(min_node_separation_ + 1, N - query_window_);

        for (int q = q_start; q < N && lc_added < max_lc_per_tick_; ++q)
        {
            int num_refs = q - min_node_separation_;
            if (num_refs <= 0) continue;
            if (tryLoopClosure(q, num_refs, nodes))
                ++lc_added;
        }

        if (lc_added > 0)
            RCLCPP_INFO(get_logger(), "[LC] Added %d loop closure(s) this tick.", lc_added);
    }
};
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node =    make_shared<graph_slam::LoopClosure>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}