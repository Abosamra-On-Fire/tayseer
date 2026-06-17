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

SCMatrix buildScanContext(const sensor_msgs::msg::LaserScan& scan, double max_range)
{
    SCMatrix sc = SCMatrix::Zero();
    double ring_width   = max_range / NR;
    double sector_width = 2.0 * M_PI / NS;
    double angle = scan.angle_min;
    for (size_t i = 0; i < scan.ranges.size(); ++i, angle += scan.angle_increment)
    {
        double r = scan.ranges[i];
        if (!isfinite(r) || r <= scan.range_min || r >= scan.range_max || r > max_range)
            continue;
        int ring = min(static_cast<int>(r / ring_width), NR - 1);
        double a = fmod(angle, 2.0 * M_PI);
        if (a < 0.0) a += 2.0 * M_PI;
        int sec = min(static_cast<int>(a / sector_width), NS - 1);
        if (r > sc(ring, sec)) sc(ring, sec) = r;
    }
    return sc;
}

pair<double, int> scanContextDistance(const SCMatrix& A, const SCMatrix& B)
{
    SCMatrix A_hat = SCMatrix::Zero();
    SCMatrix B_hat = SCMatrix::Zero();
    for (int j = 0; j < NS; ++j)
    {
        double na = A.col(j).norm();
        double nb = B.col(j).norm();
        if (na > 1e-9) A_hat.col(j) = A.col(j) / na;
        if (nb > 1e-9) B_hat.col(j) = B.col(j) / nb;
    }

    double best_dist  = 2.0;
    int    best_shift = 0;
    for (int phi = 0; phi < NS; ++phi)
    {
        double cosine_sum = 0.0;
        for (int j = 0; j < NS; ++j)
            cosine_sum += A_hat.col(j).dot(B_hat.col((j + phi) % NS));
        double dist = 1.0 - cosine_sum / static_cast<double>(NS);
        if (dist < best_dist) { best_dist = dist; best_shift = phi; }
    }
    return { best_dist, best_shift };
}



class LoopClosure : public rclcpp::Node
{
public:
    LoopClosure() : Node("loop_closure")
    {
        max_range_           = declare_parameter("max_range",           3.0);
        min_node_separation_ = declare_parameter("min_node_separation", 40);
        num_candidates_      = declare_parameter("num_candidates",      5);
        sc_threshold_        = declare_parameter("sc_threshold",        0.15);
        icp_threshold_       = declare_parameter("icp_threshold",       0.02);
        max_lc_per_tick_     = declare_parameter("max_lc_per_tick",     1);
        period_sec_          = declare_parameter("period_sec",          8.0);
        max_icp_translation_ = declare_parameter("max_icp_translation", 0.5);
        query_window_        = declare_parameter("query_window",        5);
        max_euclidean_dist_  = declare_parameter("max_euclidean_dist",  3.0);

        clt_graph_ = create_client<graph_slam::srv::GetGraph>("/graph_slam/get_graph");
        clt_edge_  = create_client<graph_slam::srv::AddEdge> ("/graph_slam/add_edge");
        clt_icp_   = create_client<graph_slam::srv::ScanMatch>("/graph_slam/scan_match");

        clt_graph_->wait_for_service();
        clt_edge_->wait_for_service();
        clt_icp_->wait_for_service();

        busy_ = false;

        timer_ = create_wall_timer(
            std::chrono::duration<double>(period_sec_),
            std::bind(&LoopClosure::tick, this));

        RCLCPP_INFO(get_logger(),
            "[LoopClosure] max_range=%.1f  min_sep=%d  candidates=%d  "
            "sc_thr=%.2f  icp_thr=%.2f  max_lc=%d  query_window=%d  max_trans=%.1f  max_euc=%.1f",
            max_range_, min_node_separation_, num_candidates_,
            sc_threshold_, icp_threshold_, max_lc_per_tick_,
            query_window_, max_icp_translation_, max_euclidean_dist_);
    }

private:
    rclcpp::Client<graph_slam::srv::GetGraph>::SharedPtr  clt_graph_;
    rclcpp::Client<graph_slam::srv::AddEdge>::SharedPtr   clt_edge_;
    rclcpp::Client<graph_slam::srv::ScanMatch>::SharedPtr clt_icp_;
    rclcpp::TimerBase::SharedPtr timer_;

    double max_range_;
    int    min_node_separation_;
    int    num_candidates_;
    double sc_threshold_;
    double icp_threshold_;
    int    max_lc_per_tick_;
    double period_sec_;
    double max_icp_translation_;
    int    query_window_;
    double max_euclidean_dist_;
    bool   busy_;

    vector<SCMatrix>                     sc_cache_;
    vector<Eigen::Matrix<double, NR, 1>> rk_cache_;
    set<pair<int,int>>                   added_lc_pairs_;

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
        for (int i = static_cast<int>(sc_cache_.size());
             i < static_cast<int>(nodes.size()); ++i)
        {
            sc_cache_.push_back(buildScanContext(nodes[i].scan, max_range_));
            rk_cache_.push_back(ringKey(sc_cache_.back()));
        }
    }

    vector<int> ringKeyTopK(int q, int ref_limit, const vector<PoseNode>& nodes)
    {
        struct Hit { int ref; double dist; };
        vector<Hit> hits;
        hits.reserve(ref_limit);

        for (int ref = 0; ref < ref_limit; ++ref)
        {
            double edx = nodes[q].x - nodes[ref].x;
            double edy = nodes[q].y - nodes[ref].y;
            double euc = sqrt(edx*edx + edy*edy);
            if (euc > max_euclidean_dist_) continue;

            double d = (rk_cache_[q] - rk_cache_[ref]).norm();
            hits.push_back({ ref, d });
        }

        int K = min(num_candidates_, static_cast<int>(hits.size()));
        if (K <= 0) return {};
        partial_sort(hits.begin(), hits.begin() + K, hits.end(),
                     [](const Hit& a, const Hit& b) { return a.dist < b.dist; });

        vector<int> topk;
        topk.reserve(K);
        for (int i = 0; i < K; ++i) topk.push_back(hits[i].ref);
        return topk;
    }

    struct CandidateHit {
        int    ref;
        double sc_dist;
        double init_dtheta_sc;
    };

    vector<CandidateHit> buildVerifiedCandidates(int q, int ref_limit,
                                                  const vector<PoseNode>& nodes)
    {
        vector<int> candidates = ringKeyTopK(q, ref_limit, nodes);
        vector<CandidateHit> verified;
        verified.reserve(candidates.size());
        for (int ref : candidates)
        {
            auto [sc_dist, shift] = scanContextDistance(sc_cache_[q], sc_cache_[ref]);
            if (sc_dist >= sc_threshold_) continue;
            double shift_rad = shift * (2.0 * M_PI / NS);
            verified.push_back({ ref, sc_dist, shift_rad });
        }
        sort(verified.begin(), verified.end(),
             [](const CandidateHit& a, const CandidateHit& b){ return a.sc_dist < b.sc_dist; });
        return verified;
    }

    void tryNextCandidate(int q,
                          vector<PoseNode> nodes,
                          vector<CandidateHit> candidates,
                          int candidate_idx,
                          int lc_added,
                          int lc_added_total)
    {
        if (candidate_idx >= static_cast<int>(candidates.size()) ||
            lc_added >= max_lc_per_tick_)
        {
            tryNextQuery(q + 1, nodes, lc_added_total);
            return;
        }

        auto& hit = candidates[candidate_idx];
        int ref = hit.ref;

        auto lc_pair = make_pair(min(nodes[ref].id, nodes[q].id),
                                 max(nodes[ref].id, nodes[q].id));
        if (added_lc_pairs_.count(lc_pair))
        {
            tryNextCandidate(q, nodes, candidates, candidate_idx + 1, lc_added, lc_added_total);
            return;
        }

        double wdx = nodes[q].x - nodes[ref].x;
        double wdy = nodes[q].y - nodes[ref].y;
        double cr  = cos(nodes[ref].theta);
        double sr  = sin(nodes[ref].theta);

        double odom_dtheta = wrapAngle(nodes[q].theta - nodes[ref].theta);
        double sc_dtheta   = wrapAngle(-hit.init_dtheta_sc);
        double init_dtheta = (fabs(wrapAngle(odom_dtheta - sc_dtheta)) < M_PI / 2.0)
                              ? sc_dtheta : odom_dtheta;

        auto sm_req = std::make_shared<graph_slam::srv::ScanMatch::Request>();
        sm_req->reference_scan = nodes[ref].scan;
        sm_req->current_scan   = nodes[q].scan;
        sm_req->init_dx        =  cr * wdx + sr * wdy;
        sm_req->init_dy        = -sr * wdx + cr * wdy;
        sm_req->init_dtheta    = init_dtheta;

        clt_icp_->async_send_request(sm_req,
            [this, q, nodes, candidates, candidate_idx, lc_added, lc_added_total,
             ref, hit, lc_pair](rclcpp::Client<graph_slam::srv::ScanMatch>::SharedFuture sm_future)
            {
                auto sm_res = sm_future.get();

                if (!sm_res->success || sm_res->score > icp_threshold_)
                {
                    tryNextCandidate(q, nodes, candidates, candidate_idx + 1, lc_added, lc_added_total);
                    return;
                }

                double icp_trans = sqrt(sm_res->dx * sm_res->dx + sm_res->dy * sm_res->dy);
                if (icp_trans > max_icp_translation_)
                {
                    RCLCPP_DEBUG(get_logger(), "[LC] REJECTED implausible translation %.2fm  %d<->%d",
                                 icp_trans, nodes[ref].id, nodes[q].id);
                    tryNextCandidate(q, nodes, candidates, candidate_idx + 1, lc_added, lc_added_total);
                    return;
                }

                double angle_change = fabs(wrapAngle(sm_res->dtheta));
                if (angle_change > M_PI / 3.0)
                {
                    RCLCPP_DEBUG(get_logger(), "[LC] REJECTED large rotation %.1fdeg  %d<->%d",
                                 angle_change * 180.0 / M_PI, nodes[ref].id, nodes[q].id);
                    tryNextCandidate(q, nodes, candidates, candidate_idx + 1, lc_added, lc_added_total);
                    return;
                }

                auto ae_req = std::make_shared<graph_slam::srv::AddEdge::Request>();
                ae_req->from_id = nodes[ref].id;
                ae_req->to_id   = nodes[q].id;
                ae_req->type    = Edge::LOOP_CLOSURE;
                ae_req->dx      = sm_res->dx;
                ae_req->dy      = sm_res->dy;
                ae_req->dtheta  = sm_res->dtheta;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        ae_req->information.push_back(sm_res->information[r * 3 + c]);

                clt_edge_->async_send_request(ae_req,
                    [this, q, nodes, candidates, candidate_idx, lc_added, lc_added_total,
                     ref, hit, lc_pair, icp_trans, angle_change, sm_res](
                        rclcpp::Client<graph_slam::srv::AddEdge>::SharedFuture ae_future)
                    {
                        auto ae_res = ae_future.get();
                        int new_lc_added = lc_added;
                        int new_total    = lc_added_total;

                        if (ae_res->success)
                        {
                            added_lc_pairs_.insert(lc_pair);
                            ++new_lc_added;
                            ++new_total;
                            RCLCPP_INFO(get_logger(),
                                "[LC] ACCEPTED  %d <-> %d   sc=%.3f  icp=%.4f  trans=%.3fm  rot=%.1fdeg",
                                nodes[ref].id, nodes[q].id,
                                hit.sc_dist, sm_res->score, icp_trans, angle_change * 180.0 / M_PI);
                        }

                        tryNextCandidate(q, nodes, candidates, candidate_idx + 1,
                                         new_lc_added, new_total);
                    });
            });
    }

    void tryNextQuery(int q, vector<PoseNode> nodes, int lc_added_total)
    {
        int N       = static_cast<int>(nodes.size());
        int q_end   = N;

        if (q >= q_end || lc_added_total >= max_lc_per_tick_)
        {
            if (lc_added_total > 0)
                RCLCPP_INFO(get_logger(), "[LC] Added %d loop closure(s) this tick.", lc_added_total);
            busy_ = false;
            return;
        }

        int ref_limit = q - min_node_separation_;
        if (ref_limit <= 0)
        {
            tryNextQuery(q + 1, nodes, lc_added_total);
            return;
        }

        auto candidates = buildVerifiedCandidates(q, ref_limit, nodes);
        if (candidates.empty())
        {
            tryNextQuery(q + 1, nodes, lc_added_total);
            return;
        }

        tryNextCandidate(q, nodes, candidates, 0, 0, lc_added_total);
    }

    void tick()
    {
        if (busy_) return;
        busy_ = true;

        auto req = std::make_shared<graph_slam::srv::GetGraph::Request>();
        clt_graph_->async_send_request(req,
            [this](rclcpp::Client<graph_slam::srv::GetGraph>::SharedFuture future)
            {
                auto srv = future.get();
                int N = static_cast<int>(srv->ids.size());
                if (N < min_node_separation_ + 2)
                {
                    busy_ = false;
                    return;
                }

                vector<PoseNode> nodes(N);
                for (int i = 0; i < N; ++i)
                {
                    nodes[i].id    = srv->ids[i];
                    nodes[i].x     = srv->xs[i];
                    nodes[i].y     = srv->ys[i];
                    nodes[i].theta = srv->thetas[i];
                    nodes[i].scan  = srv->scans[i];
                }

                updateCaches(nodes);

                int q_start = max(min_node_separation_ + 1, N - query_window_);
                tryNextQuery(q_start, nodes, 0);
            });
    }
};

}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<graph_slam::LoopClosure>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}