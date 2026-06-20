#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include "graph_slam/srv/scan_match.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

using namespace std;
namespace graph_slam {

vector<Point2D> scanToCloud(sensor_msgs::msg::LaserScan& scan)
{
    vector<Point2D> cloud;
    cloud.reserve(scan.ranges.size());
    double angle = scan.angle_min;
    for (size_t i = 0; i < scan.ranges.size(); ++i)
    {
        double r = scan.ranges[i];
        if (!isfinite(r) || r <= scan.range_min || r >= scan.range_max)
        {
            angle += scan.angle_increment;
            continue;
        }
        cloud.push_back({r*cos(angle), r*sin(angle)});
        angle += scan.angle_increment;
    }
    return cloud;
}

Eigen::Matrix2d R(double theta)
{
    double c = cos(theta), s = sin(theta);
    Eigen::Matrix2d M;
    M << c, -s,
         s,  c;
    return M;
}

vector<Point2D> applyTransform(vector<Point2D> cloud, double tx, double ty, double theta)
{
    Eigen::Matrix2d Rm = R(theta);
    vector<Point2D> out(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        Eigen::Vector2d p(cloud[i].x, cloud[i].y);
        Eigen::Vector2d pw = Rm*p + Eigen::Vector2d(tx, ty);
        out[i] = { pw(0), pw(1) };
    }
    return out;
}

struct Correspondence {
    int i;
    int j1;
    int j2;
    bool valid = false;
};

vector<Correspondence> findCorrespondences(Cloud transformed, Cloud ref)
{
    vector<Correspondence> C(transformed.size());
    int M =ref.size();
    for (int i = 0; i < transformed.size(); ++i)
    {
        double best1 = numeric_limits<double>::max();
        double best2 = numeric_limits<double>::max();
        int j1 = -1, j2 = -1;
        for (int j = 0; j < M; ++j)
        {
            double dx = transformed[i].x - ref[j].x;
            double dy = transformed[i].y - ref[j].y;
            double d2 = dx*dx + dy*dy;
            if (d2 < best1)
            {
                best2 = best1; 
                j2 = j1;
                best1 = d2; 
                j1 = j;
            }
            else if (d2 < best2)
            {
                best2 = d2; j2 = j;
            }
        }
        if (j1 < 0 || j2 < 0)
        {
            C[i].valid = false;
            continue;
        }
        C[i].i = i;
        C[i].j1 = j1;
        C[i].j2 = j2;
        C[i].valid = true;
    }
    return C;
}

vector<Correspondence> trimCorrespondences(Cloud transformed, Cloud ref,vector<Correspondence> C, double trim_ratio)
{
    vector<pair<double,int>> errors;
    errors.reserve(C.size());
    for (int k = 0; k < C.size(); ++k)
    {
        if (!C[k].valid) 
        {
            continue;
        }
        Eigen::Vector2d pj1(ref[C[k].j1].x, ref[C[k].j1].y);
        Eigen::Vector2d pj2(ref[C[k].j2].x, ref[C[k].j2].y);
        Eigen::Vector2d d = pj2 - pj1;
        double len = d.norm();
        if (len < 1e-12) 
        {
            continue;
        }
        Eigen::Vector2d n(-d(1)/len, d(0)/len);
        Eigen::Vector2d pw(transformed[C[k].i].x, transformed[C[k].i].y);
        double e = n.dot(pw - pj1);
        errors.push_back({e*e, k});
    }
    sort(errors.begin(), errors.end());
    int keep = trim_ratio * errors.size();
    vector<Correspondence> Ctrim;
    Ctrim.reserve(keep);
    for (int k = 0; k < keep && k < errors.size(); ++k)
    {
        Ctrim.push_back(C[errors[k].second]);
    }
    return Ctrim;
}

double polishedCost(Cloud src, vector<Correspondence> C, Cloud ref, Eigen::Vector4d x)
{
    double total = 0;
    for (int k = 0; k < C.size(); ++k)
    {
        Eigen::Vector2d pj1(ref[C[k].j1].x, ref[C[k].j1].y);
        Eigen::Vector2d pj2(ref[C[k].j2].x, ref[C[k].j2].y);
        Eigen::Vector2d d = pj2 - pj1;
        double len = d.norm();
        if (len < 1e-12) 
        {
            continue;
        }
        Eigen::Vector2d n(-d(1)/len, d(0)/len);
        Eigen::Matrix2d Ci = n*n.transpose();
        double p0 = src[C[k].i].x, p1 = src[C[k].i].y;
        Eigen::Matrix<double,2,4> Mi;
        Mi << 1, 0, p0, -p1,
              0, 1, p1,  p0;
        Eigen::Vector2d e = Mi*x - pj1;
        total += e.transpose()*Ci*e;
    }
    return total;
}

vector<double> solveQuartic(double a4, double a3, double a2, double a1, double a0)
{
    vector<double> roots;
    if (fabs(a4) < 1e-15) 
    {
        return roots;
    }
    Eigen::Matrix4d comp = Eigen::Matrix4d::Zero();
    comp(0,1) = 1; 
    comp(1,2) = 1; 
    comp(2,3) = 1;
    comp(3,0) = -a0/a4;
    comp(3,1) = -a1/a4;
    comp(3,2) = -a2/a4;
    comp(3,3) = -a3/a4;
    Eigen::EigenSolver<Eigen::Matrix4d> es(comp);
    for (int k = 0; k < 4; ++k)
    {
        if (fabs(es.eigenvalues()(k).imag()) < 1e-6)
            roots.push_back(es.eigenvalues()(k).real());
    }
    return roots;
}

bool computeTransformPLICP(Cloud src, vector<Correspondence> C, Cloud ref,
                            double& dx, double& dy, double& dtheta)
{
    if (C.size() < 3) return false;

    Eigen::Matrix4d Macc = Eigen::Matrix4d::Zero();
    Eigen::Vector4d g = Eigen::Vector4d::Zero();

    for (auto& c : C)
    {
        Eigen::Vector2d pj1(ref[c.j1].x, ref[c.j1].y);
        Eigen::Vector2d pj2(ref[c.j2].x, ref[c.j2].y);
        Eigen::Vector2d d = pj2 - pj1;
        double len = d.norm();
        if (len < 1e-12) continue;
        Eigen::Vector2d n(-d(1)/len, d(0)/len);
        Eigen::Matrix2d Ci = n*n.transpose();

        double p0 = src[c.i].x, p1 = src[c.i].y;
        Eigen::Matrix<double,2,4> Mi;
        Mi << 1, 0, p0, -p1,
              0, 1, p1,  p0;

        Macc += Mi.transpose()*Ci*Mi;
        g    += -2.0 * Mi.transpose()*Ci*pj1;
    }

    Eigen::Matrix4d Wm = Eigen::Matrix4d::Zero();
    Wm(2,2) = 1; Wm(3,3) = 1;

    Eigen::Matrix2d A2 = 2.0*Macc.block<2,2>(0,0);
    Eigen::Matrix2d B2 = 2.0*Macc.block<2,2>(0,2);
    Eigen::Matrix2d D2 = 2.0*Macc.block<2,2>(2,2);

    if (fabs(A2.determinant()) < 1e-12) return false;
    Eigen::Matrix2d Ainv = A2.inverse();
    Eigen::Matrix2d P = Ainv*B2;
    Eigen::Matrix2d S = D2 - B2.transpose()*P;

    double Tt = S.trace();
    double Dt = S.determinant();
    Eigen::Matrix2d Sadj;
    Sadj << S(1,1), -S(0,1),
           -S(1,0),  S(0,0);

    Eigen::Vector2d g1 = g.segment<2>(0);
    Eigen::Vector2d g2 = g.segment<2>(2);
    Eigen::Vector2d v = P.transpose()*g1;
    Eigen::Vector2d w = v - g2;

    double c2 = 4.0*w.dot(w);
    double c1 = 4.0*w.dot(Sadj*w);
    Eigen::Vector2d Sw = Sadj*w;
    double c0 = Sw.dot(Sw);

    double a4 = 16.0;
    double a3 = 16.0*Tt;
    double a2 = 4.0*Tt*Tt + 8.0*Dt - c2;
    double a1 = 4.0*Tt*Dt - c1;
    double a0 = Dt*Dt - c0;

    vector<double> roots = solveQuartic(a4, a3, a2, a1, a0);
    if (roots.empty()) return false;

    bool found = false;
    double best_cost = numeric_limits<double>::max();
    Eigen::Vector4d best_x;

    for (double lambda : roots)
    {
        Eigen::Matrix4d Z = 2.0*Macc + 2.0*lambda*Wm;
        if (fabs(Z.determinant()) < 1e-12) continue;
        Eigen::Vector4d x = -Z.inverse()*g;
        double norm2 = x(2)*x(2) + x(3)*x(3);
        if (!isfinite(norm2) || norm2 < 1e-9) continue;
        double scale = 1.0/sqrt(norm2);
        x(2) *= scale; x(3) *= scale;

        double cost = polishedCost(src, C, ref, x);
        if (cost < best_cost)
        {
            best_cost = cost;
            best_x = x;
            found = true;
        }
    }
    if (!found) return false;

    dx = best_x(0);
    dy = best_x(1);
    dtheta = atan2(best_x(3), best_x(2));
    return isfinite(dx) && isfinite(dy) && isfinite(dtheta);
}

double fitnessScorePLICP(Cloud transformed, vector<Correspondence> C, Cloud ref)
{
    if (C.empty()) return numeric_limits<double>::max();
    double total = 0;
    for (auto& c : C)
    {
        Eigen::Vector2d pj1(ref[c.j1].x, ref[c.j1].y);
        Eigen::Vector2d pj2(ref[c.j2].x, ref[c.j2].y);
        Eigen::Vector2d d = pj2 - pj1;
        double len = d.norm();
        if (len < 1e-12) continue;
        Eigen::Vector2d n(-d(1)/len, d(0)/len);
        Eigen::Vector2d pw(transformed[c.i].x, transformed[c.i].y);
        double e = n.dot(pw - pj1);
        total += e*e;
    }
    return total / C.size();
}

void composeSE2(double ax, double ay, double ath, double bx, double by, double bth,
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
    ScanMatcher() : Node("scan_matcher")
    {
        max_iterations_     = this->declare_parameter("max_iterations",     50);
        max_correspondence_ = this->declare_parameter("max_correspondence", 0.6);
        fitness_threshold_  = this->declare_parameter("fitness_threshold",  0.05);
        trim_ratio_         = this->declare_parameter("trim_ratio",         0.9);

        svc_ = this->create_service<graph_slam::srv::ScanMatch>(
            "graph_slam/scan_match",
            std::bind(&ScanMatcher::matchCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "[ScanMatcher] PL-ICP ready.");
    }

private:
    rclcpp::Service<graph_slam::srv::ScanMatch>::SharedPtr svc_;
    int    max_iterations_;
    double max_correspondence_, fitness_threshold_, trim_ratio_;

    void matchCallback(const std::shared_ptr<graph_slam::srv::ScanMatch::Request> req,
                       std::shared_ptr<graph_slam::srv::ScanMatch::Response> res)
    {
        Cloud ref  = scanToCloud(req->reference_scan);
        Cloud curr = scanToCloud(req->current_scan);
        if (ref.empty() || curr.empty())
        {
            res->success = false;
            return;
        }

        double acc_dx  = req->init_dx;
        double acc_dy  = req->init_dy;
        double acc_dth = req->init_dtheta;

        Cloud transformed = applyTransform(curr, acc_dx, acc_dy, acc_dth);

        vector<Correspondence> last_assoc;

        for (int iter = 0; iter < max_iterations_; ++iter)
        {
            auto C0 = findCorrespondences(transformed, ref);
            auto C1 = trimCorrespondences(transformed, ref, C0, trim_ratio_);
            if (C1.size() < 3) { last_assoc = C1; break; }

            double sdx, sdy, sdth;
            if (!computeTransformPLICP(transformed, C1, ref, sdx, sdy, sdth))
            {
                last_assoc = C1;
                break;
            }

            double ndx, ndy, ndth;
            composeSE2(sdx, sdy, sdth, acc_dx, acc_dy, acc_dth, ndx, ndy, ndth);
            acc_dx = ndx; acc_dy = ndy; acc_dth = ndth;

            transformed = applyTransform(transformed, sdx, sdy, sdth);
            last_assoc = C1;

            if (fabs(sdx) < 1e-9 && fabs(sdy) < 1e-9 && fabs(sdth) < 1e-9) break;
        }

        auto final_C0 = findCorrespondences(transformed, ref);
        auto final_C1 = trimCorrespondences(transformed, ref, final_C0, trim_ratio_);
        double score = fitnessScorePLICP(transformed, final_C1, ref);

        if (score > fitness_threshold_ || final_C1.size() < 3)
        {
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
        double info_xy  = min(1.0 / sigma2,        400.0);
        double info_yaw = min(info_xy * 0.25,      200.0);

        Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
        info(0,0) = info_xy;
        info(1,1) = info_xy;
        info(2,2) = info_yaw;

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                res->information.push_back(info(r,c));
    }
};

}
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<graph_slam::ScanMatcher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}