#include <ros/ros.h>
#include <graph_slam/GetGraph.h>
#include <graph_slam/AddEdge.h>
#include <graph_slam/ScanMatch.h>
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

SCMatrix buildScanContext(const sensor_msgs::LaserScan& scan, double max_range)
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

class LoopClosure
{
public:
    LoopClosure(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh)
    {
        private_nh_.param("max_range",           max_range_,           3.0);
        private_nh_.param("min_node_separation", min_node_separation_, 40);
        private_nh_.param("num_candidates",      num_candidates_,      5);
        private_nh_.param("sc_threshold",        sc_threshold_,        0.15);
        private_nh_.param("icp_threshold",       icp_threshold_,       0.02);
        private_nh_.param("max_lc_per_tick",     max_lc_per_tick_,     1);
        private_nh_.param("period_sec",          period_sec_,          8.0);
        private_nh_.param("max_icp_translation", max_icp_translation_, 0.5);
        private_nh_.param("query_window",        query_window_,        5);
        private_nh_.param("max_euclidean_dist",  max_euclidean_dist_,  3.0);

        clt_graph_ = nh_.serviceClient<graph_slam::GetGraph> ("/graph_slam/get_graph");
        clt_edge_  = nh_.serviceClient<graph_slam::AddEdge>  ("/graph_slam/add_edge");
        clt_icp_   = nh_.serviceClient<graph_slam::ScanMatch>("/graph_slam/scan_match");
        clt_graph_.waitForExistence();
        clt_edge_.waitForExistence();
        clt_icp_.waitForExistence();

        timer_ = nh_.createTimer(ros::Duration(period_sec_),
                                 &LoopClosure::tick, this);

        ROS_INFO("[LoopClosure] max_range=%.1f  min_sep=%d  candidates=%d  "
                 "sc_thr=%.2f  icp_thr=%.2f  max_lc=%d  query_window=%d  max_trans=%.1f  max_euc=%.1f",
                 max_range_, min_node_separation_, num_candidates_,
                 sc_threshold_, icp_threshold_, max_lc_per_tick_,
                 query_window_, max_icp_translation_, max_euclidean_dist_);
    }

private:
    ros::NodeHandle    nh_, private_nh_;
    ros::ServiceClient clt_graph_, clt_edge_, clt_icp_;
    ros::Timer         timer_;

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

    vector<SCMatrix>                     sc_cache_;
    vector<Eigen::Matrix<double, NR, 1>> rk_cache_;

    set<pair<int,int>> added_lc_pairs_;

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

    bool tryLoopClosure(int q, int ref_limit, vector<PoseNode>& nodes)
    {
        vector<int> candidates = ringKeyTopK(q, ref_limit, nodes);
        if (candidates.empty()) return false;

        struct SCHit { int ref; double sc_dist; double init_dtheta_sc; };
        vector<SCHit> verified;
        verified.reserve(candidates.size());

        for (int ref : candidates)
        {
            auto [sc_dist, shift] = scanContextDistance(sc_cache_[q], sc_cache_[ref]);
            if (sc_dist >= sc_threshold_) continue;
            double shift_rad = shift * (2.0 * M_PI / NS);
            verified.push_back({ ref, sc_dist, shift_rad });
        }

        if (verified.empty()) return false;

        sort(verified.begin(), verified.end(),
             [](const SCHit& a, const SCHit& b) { return a.sc_dist < b.sc_dist; });

        for (auto& hit : verified)
        {
            int ref = hit.ref;

            auto lc_pair = make_pair(min(nodes[ref].id, nodes[q].id),
                                     max(nodes[ref].id, nodes[q].id));
            if (added_lc_pairs_.count(lc_pair)) continue;

            double wdx = nodes[q].x - nodes[ref].x;
            double wdy = nodes[q].y - nodes[ref].y;
            double cr  = cos(nodes[ref].theta);
            double sr  = sin(nodes[ref].theta);

            double odom_dtheta = wrapAngle(nodes[q].theta - nodes[ref].theta);
            double sc_dtheta   = wrapAngle(-hit.init_dtheta_sc);

            double init_dtheta = (fabs(wrapAngle(odom_dtheta - sc_dtheta)) < M_PI / 2.0)
                                  ? sc_dtheta : odom_dtheta;

            graph_slam::ScanMatch sm;
            sm.request.reference_scan = nodes[ref].scan;
            sm.request.current_scan   = nodes[q].scan;
            sm.request.init_dx        =  cr * wdx + sr * wdy;
            sm.request.init_dy        = -sr * wdx + cr * wdy;
            sm.request.init_dtheta    = init_dtheta;

            if (!clt_icp_.call(sm) || !sm.response.success) continue;
            if (sm.response.score > icp_threshold_)          continue;

            double icp_trans = sqrt(sm.response.dx * sm.response.dx +
                                    sm.response.dy * sm.response.dy);
            if (icp_trans > max_icp_translation_)
            {
                ROS_DEBUG("[LC] REJECTED implausible translation %.2fm  %d<->%d",
                          icp_trans, nodes[ref].id, nodes[q].id);
                continue;
            }

            double angle_change = fabs(wrapAngle(sm.response.dtheta));
            if (angle_change > M_PI / 3.0)
            {
                ROS_DEBUG("[LC] REJECTED large rotation %.1fdeg  %d<->%d",
                          angle_change * 180.0 / M_PI, nodes[ref].id, nodes[q].id);
                continue;
            }

            graph_slam::AddEdge ae;
            ae.request.from_id = nodes[ref].id;
            ae.request.to_id   = nodes[q].id;
            ae.request.type    = Edge::LOOP_CLOSURE;
            ae.request.dx      = sm.response.dx;
            ae.request.dy      = sm.response.dy;
            ae.request.dtheta  = sm.response.dtheta;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    ae.request.information.push_back(sm.response.information[r * 3 + c]);

            if (clt_edge_.call(ae) && ae.response.success)
            {
                added_lc_pairs_.insert(lc_pair);
                ROS_INFO("[LC] ACCEPTED  %d <-> %d   sc=%.3f  icp=%.4f  trans=%.3fm  rot=%.1fdeg",
                         nodes[ref].id, nodes[q].id,
                         hit.sc_dist, sm.response.score, icp_trans, angle_change * 180.0 / M_PI);
                return true;
            }
        }
        return false;
    }

    void tick(const ros::TimerEvent&)
    {
        graph_slam::GetGraph srv;
        if (!clt_graph_.call(srv)) return;

        int N = static_cast<int>(srv.response.ids.size());
        if (N < min_node_separation_ + 2) return;

        vector<PoseNode> nodes(N);
        for (int i = 0; i < N; ++i)
        {
            nodes[i].id    = srv.response.ids[i];
            nodes[i].x     = srv.response.xs[i];
            nodes[i].y     = srv.response.ys[i];
            nodes[i].theta = srv.response.thetas[i];
            nodes[i].scan  = srv.response.scans[i];
        }

        updateCaches(nodes);

        int lc_added = 0;
        int q_start  = max(min_node_separation_ + 1, N - query_window_);

        for (int q = q_start; q < N && lc_added < max_lc_per_tick_; ++q)
        {
            int ref_limit = q - min_node_separation_;
            if (ref_limit <= 0) continue;

            if (tryLoopClosure(q, ref_limit, nodes))
                ++lc_added;
        }

        if (lc_added > 0)
            ROS_INFO("[LC] Added %d loop closure(s) this tick.", lc_added);
    }
};

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "loop_closure");
    ros::NodeHandle nh, private_nh("~");
    graph_slam::LoopClosure lc(nh, private_nh);
    ros::spin();
    return 0;
}