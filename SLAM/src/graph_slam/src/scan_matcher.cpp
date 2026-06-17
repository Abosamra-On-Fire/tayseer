#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include "graph_slam/srv/scan_match.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <vector>
#include <limits>

using namespace std;
namespace graph_slam {

vector<Point2D> scanToCloud(sensor_msgs::msg::LaserScan& scan, double bucket_deg)
{
    vector<Point2D> cloud;
    if (scan.ranges.empty()) return cloud;
    double bucket_rad = bucket_deg * M_PI / 180.0;
    int num_buckets = max(1, static_cast<int>(ceil((scan.angle_max - scan.angle_min) / bucket_rad)));

    vector<float> best_r(num_buckets, numeric_limits<float>::max());
    vector<float> best_a(num_buckets, 0.0f);

    float angle = scan.angle_min;
    for (size_t i = 0; i < scan.ranges.size(); ++i)
    {
        float r = scan.ranges[i];
        if (!isfinite(r) || r <= scan.range_min || r >= scan.range_max)
            { angle += scan.angle_increment; continue; }
        int bucket = static_cast<int>((angle - scan.angle_min) / bucket_rad);
        if (bucket < 0 || bucket >= num_buckets)
            { angle += scan.angle_increment; continue; }
        if (r < best_r[bucket]) { best_r[bucket] = r; best_a[bucket] = angle; }
        angle += scan.angle_increment;
    }
    cloud.reserve(num_buckets);
    for (int b = 0; b < num_buckets; ++b)
        if (best_r[b] < numeric_limits<float>::max())
            cloud.push_back({ best_r[b]*cos(best_a[b]), best_r[b]*sin(best_a[b]) });
    return cloud;
}

vector<Point2D> applyTransform(vector<Point2D> cloud, double dx, double dy, double dtheta)
{
    double c = cos(dtheta), s = sin(dtheta);
    vector<Point2D> out(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        Point2D p = cloud[i];
        out[i] = { c*p.x - s*p.y + dx, s*p.x + c*p.y + dy };
    }
    return out;
}

struct LineFeature {
    Eigen::Vector2d point;
    Eigen::Vector2d normal;
    bool valid = false;
};

vector<LineFeature> extractLines(Cloud cloud, int k_neighbours = 10)
{
    int N = static_cast<int>(cloud.size());
    vector<LineFeature> features(N);

    for (int i = 0; i < N; ++i)
    {
        vector<pair<double,int>> dists;
        dists.reserve(N);
        for (int j = 0; j < N; ++j) {
            double dx = cloud[i].x - cloud[j].x, dy = cloud[i].y - cloud[j].y;
            dists.push_back({dx*dx + dy*dy, j});
        }
        int actual_k = min(k_neighbours, N);
        partial_sort(dists.begin(), dists.begin()+actual_k, dists.end());

        double mx = 0, my = 0;
        for (int ki = 0; ki < actual_k; ++ki) {
            mx += cloud[dists[ki].second].x;
            my += cloud[dists[ki].second].y;
        }
        mx /= actual_k; my /= actual_k;

        Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
        for (int ki = 0; ki < actual_k; ++ki) {
            Eigen::Vector2d d(cloud[dists[ki].second].x - mx,
                              cloud[dists[ki].second].y - my);
            cov += d * d.transpose();
        }
        cov /= actual_k;

        Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector2d normal = svd.matrixU().col(1);

        double l0 = svd.singularValues()(0);
        double l1 = svd.singularValues()(1);
        if (l0 < 1e-9 || l1/l0 > 0.5) continue;

        features[i].point  = Eigen::Vector2d(mx, my);
        features[i].normal = normal;
        features[i].valid  = true;
    }
    return features;
}

struct Association {
    int    src_idx;
    int    tgt_idx;
    double perp_dist;
};

vector<Association> associatePointToLine(Cloud src, Cloud target,
                                          vector<LineFeature> features, double max_dist)
{
    vector<Association> assoc;
    assoc.reserve(src.size());

    for (size_t i = 0; i < src.size(); ++i)
    {
        double best_dist_sq = max_dist * max_dist;
        int    best_j = -1;

        for (size_t j = 0; j < target.size(); ++j)
        {
            if (!features[j].valid) continue;
            double dx = src[i].x - target[j].x, dy = src[i].y - target[j].y;
            double d2 = dx*dx + dy*dy;
            if (d2 < best_dist_sq) { best_dist_sq = d2; best_j = static_cast<int>(j); }
        }
        if (best_j < 0) continue;

        Eigen::Vector2d diff(src[i].x - features[best_j].point(0),
                             src[i].y - features[best_j].point(1));
        double pd = features[best_j].normal.dot(diff);
        assoc.push_back({static_cast<int>(i), best_j, pd});
    }
    return assoc;
}

bool computeTransformPointToLine(Cloud src, vector<Association> assoc,
                                  vector<LineFeature> features,
                                  double& dx, double& dy, double& dtheta)
{
    if (assoc.size() < 3) return false;

    Eigen::MatrixXd A(assoc.size(), 4);
    Eigen::VectorXd b(assoc.size());

    for (int k = 0; k < static_cast<int>(assoc.size()); ++k)
    {
        auto& a  = assoc[k];
        auto& lf = features[a.tgt_idx];
        double px = src[a.src_idx].x, py = src[a.src_idx].y;
        double nx = lf.normal(0),     ny = lf.normal(1);
        double qx = lf.point(0),      qy = lf.point(1);

        double np  =  nx*px + ny*py;
        double ncr = -nx*py + ny*px;

        A(k, 0) = np;
        A(k, 1) = ncr;
        A(k, 2) = nx;
        A(k, 3) = ny;

        b(k) = nx*qx + ny*qy - np;
    }

    Eigen::Matrix4d ATA = A.transpose() * A;
    Eigen::Vector4d ATb = A.transpose() * b;

    if (std::abs(ATA.determinant()) < 1e-12) return false;

    Eigen::Vector4d x = ATA.ldlt().solve(ATb);

    double c1 = x(0), s1 = x(1);
    dtheta = atan2(s1, c1 + 1.0);
    dx     = x(2);
    dy     = x(3);

    if (!isfinite(dx) || !isfinite(dy) || !isfinite(dtheta)) return false;

    const double MAX_STEP_T = 2.0, MAX_STEP_R = M_PI;
    if (fabs(dx) > MAX_STEP_T || fabs(dy) > MAX_STEP_T || fabs(dtheta) > MAX_STEP_R)
    {
        dx     = max(-MAX_STEP_T, min(MAX_STEP_T, dx));
        dy     = max(-MAX_STEP_T, min(MAX_STEP_T, dy));
        dtheta = max(-MAX_STEP_R, min(MAX_STEP_R, dtheta));
    }

    return true;
}

double fitnessScorePointToLine(vector<Association> assoc)
{
    if (assoc.empty()) return numeric_limits<double>::max();
    double total = 0;
    for (auto& a : assoc) total += a.perp_dist * a.perp_dist;
    return total / assoc.size();
}

void composeSE2(double ax, double ay, double ath,
                double bx, double by, double bth,
                double& ox, double& oy, double& oth)
{
    double ca = cos(ath), sa = sin(ath);
    ox  = ax + ca*bx - sa*by;
    oy  = ay + sa*bx + ca*by;
    oth = wrapAngle(ath + bth);
}

class ScanMatcher : public rclcpp::Node
{
public:
    explicit ScanMatcher()
    : Node("scan_matcher")
    {
        max_iterations_    = declare_parameter<int>   ("max_iterations",     50);
        max_correspondence_ = declare_parameter<double>("max_correspondence", 0.6);
        fitness_threshold_  = declare_parameter<double>("fitness_threshold",  0.05);
        transformation_eps_ = declare_parameter<double>("transformation_eps", 1e-6);
        downsample_deg_     = declare_parameter<double>("downsample_deg",     0.5);
        line_neighbours_    = declare_parameter<int>   ("line_neighbours",    10);

        svc_ = create_service<graph_slam::srv::ScanMatch>(
            "graph_slam/scan_match",
            [this](const graph_slam::srv::ScanMatch::Request::SharedPtr  req,
                         graph_slam::srv::ScanMatch::Response::SharedPtr res)
            { matchCallback(req, res); });

        RCLCPP_INFO(get_logger(),
            "[ScanMatcher] Point-to-line ICP ready. downsample=%.1fdeg max_corr=%.2fm line_k=%d",
            downsample_deg_, max_correspondence_, line_neighbours_);
    }

private:
    rclcpp::Service<graph_slam::srv::ScanMatch>::SharedPtr svc_;
    int    max_iterations_, line_neighbours_;
    double max_correspondence_, fitness_threshold_, transformation_eps_;
    double downsample_deg_;

    void matchCallback(const graph_slam::srv::ScanMatch::Request::SharedPtr  req,
                             graph_slam::srv::ScanMatch::Response::SharedPtr res)
    {
        Cloud ref  = scanToCloud(req->reference_scan, downsample_deg_);
        Cloud curr = scanToCloud(req->current_scan,   downsample_deg_);
        if (ref.empty() || curr.empty()) { res->success = false; return; }

        vector<LineFeature> ref_lines = extractLines(ref, line_neighbours_);

        double acc_dx  = req->init_dx;
        double acc_dy  = req->init_dy;
        double acc_dth = req->init_dtheta;

        Cloud transformed = applyTransform(curr, acc_dx, acc_dy, acc_dth);
        double prev_score = numeric_limits<double>::max();

        for (int iter = 0; iter < max_iterations_; ++iter)
        {
            auto assoc = associatePointToLine(transformed, ref, ref_lines, max_correspondence_);
            if (assoc.size() < 6) break;

            double sdx, sdy, sdth;
            if (!computeTransformPointToLine(transformed, assoc, ref_lines, sdx, sdy, sdth))
                break;

            double ndx, ndy, ndth;
            composeSE2(acc_dx, acc_dy, acc_dth, sdx, sdy, sdth, ndx, ndy, ndth);
            acc_dx = ndx; acc_dy = ndy; acc_dth = ndth;

            transformed = applyTransform(transformed, sdx, sdy, sdth);

            double score = fitnessScorePointToLine(assoc);

            double step_mag = sdx*sdx + sdy*sdy + sdth*sdth;
            if (step_mag < transformation_eps_) break;

            if (score > prev_score * 1.5 && iter > 5) break;
            prev_score = score;
        }

        auto final_assoc = associatePointToLine(transformed, ref, ref_lines, max_correspondence_);
        double score = fitnessScorePointToLine(final_assoc);

        if (score > fitness_threshold_ || final_assoc.size() < 6) {
            res->success = false;
            res->score   = score;
            return;
        }

        res->success = true;
        res->dx      = acc_dx;
        res->dy      = acc_dy;
        res->dtheta  = acc_dth;
        res->score   = score;

        double sigma2   = max(score, 1e-6);
        double info_xy  = min(1.0 / sigma2,   400.0);
        double info_yaw = min(info_xy * 0.25, 200.0);

        Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
        info(0,0) = info_xy;
        info(1,1) = info_xy;
        info(2,2) = info_yaw;

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                res->information.push_back(info(r,c));
    }
};

} // namespace graph_slam

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<graph_slam::ScanMatcher>());
    rclcpp::shutdown();
    return 0;
}