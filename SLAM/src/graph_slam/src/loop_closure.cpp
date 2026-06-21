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
#include <mutex>
#include <atomic>
#include <Eigen/Dense>
using namespace std;
namespace graph_slam {
static const int NR = 20;
static const int NS = 60;
using SCMatrix = Eigen::Matrix<double, NR, NS>;

SCMatrix buildScanContext(sensor_msgs::msg::LaserScan scan, double max_range)
{
    SCMatrix sc = SCMatrix::Zero();
    if (scan.ranges.empty()) return sc;
    double ring_width   = max_range / NR;
    double sector_width = 2.0 * M_PI / NS;
    if (ring_width <= 0.0 || sector_width <= 0.0) return sc;
    double angle = scan.angle_min;
    for (size_t i = 0; i < scan.ranges.size(); ++i)
    {
        double r = scan.ranges[i];
        if (!isfinite(r) || r <= scan.range_min || r >= scan.range_max || r > max_range)
        {
            angle += scan.angle_increment;
            continue;
        }
        int ring = min(static_cast<int>(r / ring_width), NR - 1);
        double a = angle;
        while (a <  0.0)       a += 2.0 * M_PI;
        while (a >= 2.0*M_PI)  a -= 2.0 * M_PI;
        int sec = min(static_cast<int>(a / sector_width), NS - 1);
        sec = max(sec, 0);
        if (r > sc(ring, sec)) sc(ring, sec) = r;
        angle += scan.angle_increment;
    }
    return sc;
}

pair<double, int> scanContextDistance(const SCMatrix& A, const SCMatrix& B)
{
    SCMatrix A_hat = SCMatrix::Zero();
    SCMatrix B_hat = SCMatrix::Zero();
    vector<bool> a_valid(NS, false), b_valid(NS, false);
    for (int j = 0; j < NS; ++j)
    {
        double na = A.col(j).norm();
        double nb = B.col(j).norm();
        if (na > 1e-9) { A_hat.col(j) = (A.col(j) / na).eval(); a_valid[j] = true; }
        if (nb > 1e-9) { B_hat.col(j) = (B.col(j) / nb).eval(); b_valid[j] = true; }
    }
    double best_dist  = numeric_limits<double>::max();
    int    best_shift = 0;
    for (int phi = 0; phi < NS; ++phi)
    {
        double cosine_sum = 0.0;
        int valid_count = 0;
        for (int j = 0; j < NS; ++j)
        {
            int j_shifted = (j + phi) % NS;
            if (!a_valid[j] || !b_valid[j_shifted]) continue;
            cosine_sum += A_hat.col(j).dot(B_hat.col(j_shifted));
            ++valid_count;
        }
        if (valid_count < 5) continue;
        double dist = 1.0 - cosine_sum / static_cast<double>(valid_count);
        if (dist < best_dist) { best_dist = dist; best_shift = phi; }
    }
    if (best_dist == numeric_limits<double>::max()) return {1.0, 0};
    return { best_dist, best_shift };
}

class LoopClosure : public rclcpp::Node
{
public:
    LoopClosure() : Node("loop_closure"), lc_in_flight_(0)
    {
        max_range_           = declare_parameter("max_range",            20.0);
        min_node_separation_ = declare_parameter("min_node_separation",  10);
        num_candidates_      = declare_parameter("num_candidates",       30);
        sc_threshold_        = declare_parameter("sc_threshold",         0.15);
        icp_threshold_       = declare_parameter("icp_threshold",        0.15);
        max_lc_per_tick_     = declare_parameter("max_lc_per_tick",      5);
        period_sec_          = declare_parameter("period_sec",           2.0);
        max_icp_translation_ = declare_parameter("max_icp_translation",  2.0);
        query_window_        = declare_parameter("query_window",         500);
        max_world_dist_      = declare_parameter("max_world_dist",       5.0);

        cb_group_reentrant_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        cb_group_graph_     = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_edge_      = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_icp_       = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        clt_graph_ = create_client<graph_slam::srv::GetGraph> ("graph_slam/get_graph",  rmw_qos_profile_services_default, cb_group_graph_);
        clt_edge_  = create_client<graph_slam::srv::AddEdge>  ("graph_slam/add_edge",   rmw_qos_profile_services_default, cb_group_edge_);
        clt_icp_   = create_client<graph_slam::srv::ScanMatch>("graph_slam/scan_match", rmw_qos_profile_services_default, cb_group_icp_);

        timer_ = create_wall_timer(
            chrono::duration<double>(period_sec_),
            bind(&LoopClosure::tick, this),
            cb_group_reentrant_);

        RCLCPP_INFO(get_logger(),
            "[LoopClosure] max_range=%.1f  min_sep=%d  candidates=%d  "
            "sc_thr=%.2f  icp_thr=%.2f  max_lc=%d  query_window=%d  max_trans=%.1f  max_world=%.1f",
            max_range_, min_node_separation_, num_candidates_,
            sc_threshold_, icp_threshold_, max_lc_per_tick_,
            query_window_, max_icp_translation_, max_world_dist_);
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
    double max_world_dist_;

    vector<SCMatrix>                     sc_cache_;
    vector<Eigen::Matrix<double, NR, 1>> rk_cache_;
    set<pair<int,int>>                   added_lc_pairs_;
    set<pair<int,int>>                   rejected_lc_pairs_;
    mutex                                cache_mutex_;
    atomic<int>                          lc_in_flight_;
    atomic<bool>                         tick_in_flight_{false};
    static Eigen::Matrix<double, NR, 1> ringKey(const SCMatrix& sc)
    {
        Eigen::Matrix<double, NR, 1> k;
        for (int i = 0; i < NR; ++i)
        {
            int nonzero = 0;
            for (int j = 0; j < NS; ++j)
                if (sc(i, j) > 1e-9) ++nonzero;
            k(i) = static_cast<double>(nonzero) / NS;
        }
        return k;
    }

    void updateCaches(const vector<PoseNode>& nodes)
    {
        size_t old_size;
        {
            lock_guard<mutex> lock(cache_mutex_);
            old_size = sc_cache_.size();
        }
        vector<SCMatrix>                     new_sc;
        vector<Eigen::Matrix<double, NR, 1>> new_rk;
        for (size_t i = old_size; i < nodes.size(); ++i)
        {
            SCMatrix sc = buildScanContext(nodes[i].scan, max_range_);
            new_sc.push_back(sc);
            new_rk.push_back(ringKey(sc));
        }
        if (!new_sc.empty())
        {
            lock_guard<mutex> lock(cache_mutex_);
            for (size_t i = 0; i < new_sc.size(); ++i)
            {
                sc_cache_.push_back(new_sc[i]);
                rk_cache_.push_back(new_rk[i]);
            }
        }
    }

    vector<int> ringKeyTopK(int q, int num_refs)
    {
        vector<Eigen::Matrix<double, NR, 1>> rk_snap;
        {
            lock_guard<mutex> lock(cache_mutex_);
            if (q < 0 || q >= static_cast<int>(rk_cache_.size())) return {};
            rk_snap.assign(rk_cache_.begin(), rk_cache_.end());
        }
        struct Hit { int ref; double dist; };
        vector<Hit> hits;
        hits.reserve(num_refs);
        for (int ref = 0; ref < num_refs; ++ref)
        {
            if (ref >= static_cast<int>(rk_snap.size())) break;
            double d = (rk_snap[q] - rk_snap[ref]).norm();
            hits.push_back({ ref, d });
        }
        int K = min(num_candidates_, static_cast<int>(hits.size()));
        if (K <= 0) return {};
        partial_sort(hits.begin(), hits.begin() + K, hits.end(),
                     [](const Hit& a, const Hit& b){ return a.dist < b.dist; });
        vector<int> topk;
        topk.reserve(K);
        for (int i = 0; i < K; ++i) topk.push_back(hits[i].ref);
        return topk;
    }

    void tryLoopClosureAsync(int q, int num_refs, vector<PoseNode> nodes,
                             vector<pair<int, SCMatrix>> cand_sc_pairs, SCMatrix q_sc)
    {
        double best_sc    = numeric_limits<double>::max();
        int    best_ref   = -1;
        int    best_shift = 0;

        for (auto& [ref_idx, ref_sc] : cand_sc_pairs)
        {
            auto [sc_dist, shift] = scanContextDistance(q_sc, ref_sc);
            if (sc_dist < best_sc)
            {
                best_sc    = sc_dist;
                best_ref   = ref_idx;
                best_shift = shift;
            }
        }

        if (best_ref < 0 || best_sc >= sc_threshold_) return;
        if (best_ref >= static_cast<int>(nodes.size()) || q >= static_cast<int>(nodes.size())) return;

        pair<int,int> lc_pair(nodes[best_ref].id, nodes[q].id);
        {
            lock_guard<mutex> lock(cache_mutex_);
            if (added_lc_pairs_.count(lc_pair))    return;
            if (rejected_lc_pairs_.count(lc_pair)) return;
        }

        double world_dx   = nodes[q].x - nodes[best_ref].x;
        double world_dy   = nodes[q].y - nodes[best_ref].y;
        double world_dist = sqrt(world_dx * world_dx + world_dy * world_dy);
        if (world_dist > max_world_dist_)
        {
            RCLCPP_INFO(get_logger(), "[LC] world dist %.2fm > %.2fm reject pre-ICP", world_dist, max_world_dist_);
            lock_guard<mutex> lock(cache_mutex_);
            rejected_lc_pairs_.insert(lc_pair);
            return;
        }

        RCLCPP_INFO(get_logger(), "[LC] Candidate: node %d <-> %d  sc=%.3f  shift=%d",
                    nodes[best_ref].id, nodes[q].id, best_sc, best_shift);

        double cr           = cos(nodes[best_ref].theta);
        double sr           = sin(nodes[best_ref].theta);
        double sector_width = 2.0 * M_PI / NS;
        double shift_angle  = best_shift * sector_width;
        double init_dx      =  cr * world_dx + sr * world_dy;
        double init_dy      = -sr * world_dx + cr * world_dy;
        double init_dtheta  = wrapAngle(nodes[q].theta - nodes[best_ref].theta - shift_angle);

        auto sm_req            = std::make_shared<graph_slam::srv::ScanMatch::Request>();
        sm_req->reference_scan = nodes[best_ref].scan;
        sm_req->current_scan   = nodes[q].scan;
        sm_req->init_dx        = init_dx;
        sm_req->init_dy        = init_dy;
        sm_req->init_dtheta    = init_dtheta;

        int ref_id = nodes[best_ref].id;
        int q_id   = nodes[q].id;

        lc_in_flight_.fetch_add(1);

        clt_icp_->async_send_request(sm_req,
            [this, ref_id, q_id, best_sc, lc_pair]
            (rclcpp::Client<graph_slam::srv::ScanMatch>::SharedFuture f)
            {
                auto sm_res = f.get();
                if (!sm_res || !sm_res->success)
                {
                    {
                        lock_guard<mutex> lock(cache_mutex_);
                        rejected_lc_pairs_.insert(lc_pair);
                    }
                    RCLCPP_INFO(get_logger(), "[LC] ICP failed %d<->%d", ref_id, q_id);
                    lc_in_flight_.fetch_sub(1);
                    return;
                }
                if (sm_res->score > icp_threshold_)
                {
                    {
                        lock_guard<mutex> lock(cache_mutex_);
                        rejected_lc_pairs_.insert(lc_pair);
                    }
                    RCLCPP_INFO(get_logger(), "[LC] score %.4f > thr %.4f reject", sm_res->score, icp_threshold_);
                    lc_in_flight_.fetch_sub(1);
                    return;
                }
                double icp_trans = sqrt(sm_res->dx * sm_res->dx + sm_res->dy * sm_res->dy);
                if (icp_trans > max_icp_translation_)
                {
                    {
                        lock_guard<mutex> lock(cache_mutex_);
                        rejected_lc_pairs_.insert(lc_pair);
                    }
                    RCLCPP_INFO(get_logger(), "[LC] translation %.2fm > %.2fm reject", icp_trans, max_icp_translation_);
                    lc_in_flight_.fetch_sub(1);
                    return;
                }

                auto ae_req      = std::make_shared<graph_slam::srv::AddEdge::Request>();
                ae_req->from_id  = ref_id;
                ae_req->to_id    = q_id;
                ae_req->type     = Edge::LOOP_CLOSURE;
                ae_req->dx       = sm_res->dx;
                ae_req->dy       = sm_res->dy;
                ae_req->dtheta   = sm_res->dtheta;

                Eigen::Matrix3d default_info = Eigen::Matrix3d::Identity() * 100.0;
                if (sm_res->information.size() >= 9)
                {
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c)
                            ae_req->information.push_back(sm_res->information[r * 3 + c]);
                }
                else
                {
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c)
                            ae_req->information.push_back(default_info(r, c));
                }

                double captured_score = sm_res->score;
                double captured_trans = icp_trans;

                clt_edge_->async_send_request(ae_req,
                    [this, lc_pair, ref_id, q_id, best_sc, captured_score, captured_trans]
                    (rclcpp::Client<graph_slam::srv::AddEdge>::SharedFuture f2)
                    {
                        auto ae_res = f2.get();
                        if (ae_res && ae_res->success)
                        {
                            lock_guard<mutex> lock(cache_mutex_);
                            added_lc_pairs_.insert(lc_pair);
                            RCLCPP_INFO(get_logger(),
                                        "[LC] ACCEPTED  %d <-> %d   sc=%.3f  icp=%.4f  trans=%.3fm",
                                        ref_id, q_id, best_sc, captured_score, captured_trans);
                        }
                        else
                        {
                            lock_guard<mutex> lock(cache_mutex_);
                            rejected_lc_pairs_.insert(lc_pair);
                        }
                        lc_in_flight_.fetch_sub(1);
                    });
            });
    }

void tick()
{
    bool expected = false;
    if (!tick_in_flight_.compare_exchange_strong(expected, true)) return;
    if (lc_in_flight_.load() > 0) {
        tick_in_flight_ = false;
        return;
    }

    auto req = std::make_shared<graph_slam::srv::GetGraph::Request>();
    clt_graph_->async_send_request(req,
        [this](rclcpp::Client<graph_slam::srv::GetGraph>::SharedFuture f) {
        auto srv_response = f.get();
        if (!srv_response) {
            tick_in_flight_ = false;
            return;
        }

        int N = static_cast<int>(srv_response->ids.size());
        if (N < min_node_separation_ + 2) {
            tick_in_flight_ = false;
            return;
        }
        if (static_cast<int>(srv_response->xs.size())     < N ||
            static_cast<int>(srv_response->ys.size())     < N ||
            static_cast<int>(srv_response->thetas.size()) < N ||
            static_cast<int>(srv_response->scans.size())  < N) {
            tick_in_flight_ = false;
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

        int lc_dispatched = 0;
        int q_start = max(min_node_separation_ + 1, N - query_window_);
        for (int q = N - 1; q >= q_start && lc_dispatched < max_lc_per_tick_; --q)
        {
            int num_refs = q - min_node_separation_;
            if (num_refs <= 0) continue;

            vector<int> candidates = ringKeyTopK(q, num_refs);
            if (candidates.empty()) continue;

            vector<pair<int, SCMatrix>> cand_sc_pairs;
            SCMatrix                    q_sc;
            bool skip = false;
            {
                lock_guard<mutex> lock(cache_mutex_);
                if (q < 0 || q >= static_cast<int>(sc_cache_.size()))
                {
                    skip = true;
                }
                else
                {
                    q_sc = sc_cache_[q];
                    for (int c : candidates)
                    {
                        if (c < 0 || c >= static_cast<int>(sc_cache_.size())) continue;
                        if (c >= static_cast<int>(nodes.size())) continue;
                        pair<int,int> lc_pair(nodes[c].id, nodes[q].id);
                        if (added_lc_pairs_.count(lc_pair))    continue;
                        if (rejected_lc_pairs_.count(lc_pair)) continue;
                        cand_sc_pairs.push_back({ c, sc_cache_[c] });
                    }
                }
            }
            if (skip || cand_sc_pairs.empty()) continue;

            tryLoopClosureAsync(q, num_refs, nodes, cand_sc_pairs, q_sc);
            ++lc_dispatched;
        }
        tick_in_flight_ = false;
    });
}
};
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<graph_slam::LoopClosure>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}