/**
 * Evaluation Node
 *
 * Central evaluation and dropout control node. Runs at 10 Hz and:
 *   1. Collects poses from all three filters and ground truth (/odom)
 *   2. Computes RMSE, position error, covariance, and landmark distance
 *   3. Simulates sensor dropout by publishing /measurement_dropout (Bool)
 *   4. Writes all metrics to CSV files
 *   5. Triggers plot_results.py automatically when trajectory is done
 *
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>

struct Pose2D { double x=0, y=0, theta=0; };

// A time interval [start, end) during which dropout is active
struct DropoutInterval { double start, end; };

class EvaluationNode : public rclcpp::Node
{
public:
  EvaluationNode() : Node("evaluation_node"), trajectory_started_(false)
  {
    declare_parameter("output_dir",  "/home/david/ros2_ws/src/prob_lab/results");
    declare_parameter("plot_script", "/home/david/ros2_ws/src/prob_lab/plot_results.py");
    declare_parameter("landmark_x",  0.6);
    declare_parameter("landmark_y",  0.6);

    output_dir_  = get_parameter("output_dir").as_string();
    plot_script_ = get_parameter("plot_script").as_string();
    landmark_x_  = get_parameter("landmark_x").as_double();
    landmark_y_  = get_parameter("landmark_y").as_double();

    // Dropout intervals in seconds after trajectory start
    dropouts_ = {
      {70.0, 71.0},
      {74.0, 75.0},
      {78.0, 79.0},
      {79.0, 80.0},
    };

    system(("mkdir -p " + output_dir_).c_str());
    openFiles();

    // Subscribe to ground truth and all filter outputs
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&EvaluationNode::odomCb, this, std::placeholders::_1));
    kf_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/kf_pose", 10, std::bind(&EvaluationNode::kfCb, this, std::placeholders::_1));
    ekf_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/ekf_pose", 10, std::bind(&EvaluationNode::ekfCb, this, std::placeholders::_1));
    pf_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/pf_pose", 10, std::bind(&EvaluationNode::pfCb, this, std::placeholders::_1));

    // /trajectory_done triggers CSV flush and plot generation
    done_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/trajectory_done", 10, std::bind(&EvaluationNode::doneCb, this, std::placeholders::_1));

    // Publishes dropout signal to all filter nodes
    dropout_pub_ = create_publisher<std_msgs::msg::Bool>("/measurement_dropout", 10);

    // Main evaluation loop at 10 Hz
    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&EvaluationNode::timerCb, this));

    start_time_ = this->now();
    RCLCPP_INFO(get_logger(), "Evaluation Node started. Output: %s", output_dir_.c_str());
    RCLCPP_INFO(get_logger(), "Dropout intervals: %zu configured", dropouts_.size());
  }

private:
  // Open all CSV output files with headers
  void openFiles()
  {
    poses_file_.open(output_dir_ + "/poses.csv");
    poses_file_ << "time,gt_x,gt_y,gt_theta,kf_x,kf_y,kf_theta,ekf_x,ekf_y,ekf_theta,pf_x,pf_y,pf_theta\n";
    rmse_file_.open(output_dir_ + "/rmse.csv");
    rmse_file_ << "time,rmse_kf,rmse_ekf,rmse_pf,error_x_kf,error_y_kf,error_x_ekf,error_y_ekf,error_x_pf,error_y_pf\n";
    cov_file_.open(output_dir_ + "/covariance.csv");
    cov_file_ << "time,kf_cov_x,kf_cov_y,kf_cov_theta,ekf_cov_x,ekf_cov_y,ekf_cov_theta,pf_cov_x,pf_cov_y,pf_cov_theta\n";
    landmark_file_.open(output_dir_ + "/landmark.csv");
    landmark_file_ << "time,true_dist,kf_dist,ekf_dist,pf_dist,kf_err,ekf_err,pf_err\n";
    dropout_file_.open(output_dir_ + "/interruption.csv");
    dropout_file_ << "time,in_dropout,gt_x,gt_y,kf_x,kf_y,ekf_x,ekf_y,pf_x,pf_y,kf_err,ekf_err,pf_err\n";
  }

  // Format double to fixed 4 decimal places for CSV output
  std::string fp(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4) << v;
    return ss.str();
  }

  // Check if trajectory time t falls within any dropout interval
  bool isDropout(double t) {
    for (const auto & d : dropouts_) {
      if (t >= d.start && t < d.end) return true;
    }
    return false;
  }

  // Ground truth from /odom — also detects when robot starts moving
  void odomCb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    gt_.x = msg->pose.pose.position.x;
    gt_.y = msg->pose.pose.position.y;
    gt_received_ = true;
    // Dropout timer starts relative to first robot movement, not node start
    if (!trajectory_started_) {
      double v = msg->twist.twist.linear.x;
      double w = msg->twist.twist.angular.z;
      if (std::abs(v) > 0.01 || std::abs(w) > 0.01) {
        trajectory_started_ = true;
        traj_start_time_ = this->now();
        RCLCPP_INFO(get_logger(), "Trajectory movement detected — dropout timer started");
      }
    }
  }

  // Store latest filter estimates and their covariances
  void kfCb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  { kf_.x=msg->pose.pose.position.x; kf_.y=msg->pose.pose.position.y;
    kf_cov_x_=msg->pose.covariance[0]; kf_cov_y_=msg->pose.covariance[7];
    kf_cov_t_=msg->pose.covariance[35]; kf_received_=true; }

  void ekfCb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  { ekf_.x=msg->pose.pose.position.x; ekf_.y=msg->pose.pose.position.y;
    ekf_cov_x_=msg->pose.covariance[0]; ekf_cov_y_=msg->pose.covariance[7];
    ekf_cov_t_=msg->pose.covariance[35]; ekf_received_=true; }

  void pfCb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  { pf_.x=msg->pose.pose.position.x; pf_.y=msg->pose.pose.position.y;
    pf_cov_x_=msg->pose.covariance[0]; pf_cov_y_=msg->pose.covariance[7];
    pf_cov_t_=msg->pose.covariance[35]; pf_received_=true; }

  // Called when trajectory_node signals completion
  // Flushes all CSV files and launches plot_results.py
  void doneCb(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg->data || done_) return;
    done_ = true;
    RCLCPP_INFO(get_logger(), "Trajectory done! Saving results...");
    if (n_>0) RCLCPP_INFO(get_logger(),
      "Final RMSE — KF: %.4fm  EKF: %.4fm  PF: %.4fm",
      std::sqrt(sq_kf_/n_), std::sqrt(sq_ekf_/n_), std::sqrt(sq_pf_/n_));
    poses_file_.flush();    poses_file_.close();
    rmse_file_.flush();     rmse_file_.close();
    cov_file_.flush();      cov_file_.close();
    landmark_file_.flush(); landmark_file_.close();
    dropout_file_.flush();  dropout_file_.close();
    system(("python3 " + plot_script_ + " &").c_str());
    RCLCPP_INFO(get_logger(), "Results saved to: %s", output_dir_.c_str());
  }

  // Main loop at 10 Hz: compute metrics, publish dropout, write CSV
  void timerCb()
  {
    if (!gt_received_ || !kf_received_ || !ekf_received_ || done_) return;

    double t = (this->now() - start_time_).seconds();

    // Dropout is relative to trajectory start time
    double traj_t = trajectory_started_ ?
      (this->now() - traj_start_time_).seconds() : -1.0;
    bool dropout = trajectory_started_ && isDropout(traj_t);

    // Publish dropout signal — all filter nodes subscribe and skip UPDATE if true
    std_msgs::msg::Bool dmsg; dmsg.data = dropout;
    dropout_pub_->publish(dmsg);

    if (dropout && !was_dropout_) {
      RCLCPP_WARN(get_logger(), "=== DROPOUT START traj_t=%.1fs ===", traj_t);
      was_dropout_ = true;
    }
    if (!dropout && was_dropout_) {
      RCLCPP_WARN(get_logger(), "=== DROPOUT END traj_t=%.1fs ===", traj_t);
      was_dropout_ = false;
    }

    // Euclidean position error vs ground truth
    double err_kf  = dist(gt_,kf_);
    double err_ekf = dist(gt_,ekf_);
    double err_pf  = pf_received_ ? dist(gt_,pf_) : 0.0;

    // Cumulative RMSE: running sum of squared errors divided by sample count
    sq_kf_+=err_kf*err_kf; sq_ekf_+=err_ekf*err_ekf; sq_pf_+=err_pf*err_pf; n_++;
    double rmse_kf=std::sqrt(sq_kf_/n_), rmse_ekf=std::sqrt(sq_ekf_/n_), rmse_pf=std::sqrt(sq_pf_/n_);

    // Landmark: distance from each filter estimate to fixed point (landmark_x, landmark_y)
    auto ldist=[&](Pose2D&p){return std::sqrt(std::pow(p.x-landmark_x_,2)+std::pow(p.y-landmark_y_,2));};
    double gt_ld=ldist(gt_),kf_ld=ldist(kf_),ekf_ld=ldist(ekf_),pf_ld=pf_received_?ldist(pf_):0.0;

    // Write all metrics to CSV files
    poses_file_<<fp(t)<<","<<fp(gt_.x)<<","<<fp(gt_.y)<<","<<fp(gt_.theta)<<","
      <<fp(kf_.x)<<","<<fp(kf_.y)<<","<<fp(kf_.theta)<<","
      <<fp(ekf_.x)<<","<<fp(ekf_.y)<<","<<fp(ekf_.theta)<<","
      <<fp(pf_.x)<<","<<fp(pf_.y)<<","<<fp(pf_.theta)<<"\n";

    rmse_file_<<fp(t)<<","<<fp(rmse_kf)<<","<<fp(rmse_ekf)<<","<<fp(rmse_pf)<<","
      <<fp(kf_.x-gt_.x)<<","<<fp(kf_.y-gt_.y)<<","
      <<fp(ekf_.x-gt_.x)<<","<<fp(ekf_.y-gt_.y)<<","
      <<fp(pf_.x-gt_.x)<<","<<fp(pf_.y-gt_.y)<<"\n";

    cov_file_<<fp(t)<<","<<fp(kf_cov_x_)<<","<<fp(kf_cov_y_)<<","<<fp(kf_cov_t_)<<","
      <<fp(ekf_cov_x_)<<","<<fp(ekf_cov_y_)<<","<<fp(ekf_cov_t_)<<","
      <<fp(pf_cov_x_)<<","<<fp(pf_cov_y_)<<","<<fp(pf_cov_t_)<<"\n";

    landmark_file_<<fp(t)<<","<<fp(gt_ld)<<","<<fp(kf_ld)<<","<<fp(ekf_ld)<<","<<fp(pf_ld)<<","
      <<fp(std::abs(kf_ld-gt_ld))<<","<<fp(std::abs(ekf_ld-gt_ld))<<","<<fp(std::abs(pf_ld-gt_ld))<<"\n";

    dropout_file_<<fp(t)<<","<<(dropout?1:0)<<","
      <<fp(gt_.x)<<","<<fp(gt_.y)<<","<<fp(kf_.x)<<","<<fp(kf_.y)<<","
      <<fp(ekf_.x)<<","<<fp(ekf_.y)<<","<<fp(pf_.x)<<","<<fp(pf_.y)<<","
      <<fp(err_kf)<<","<<fp(err_ekf)<<","<<fp(err_pf)<<"\n";

    if (n_%50==0) RCLCPP_INFO(get_logger(),
      "t=%.1fs traj=%.1fs | KF=%.3fm EKF=%.3fm PF=%.3fm | %s",
      t, traj_t, rmse_kf, rmse_ekf, rmse_pf, dropout?"DROPOUT":"ok");
  }

  // Euclidean distance between two 2D poses
  double dist(const Pose2D&a,const Pose2D&b)
  { return std::sqrt(std::pow(a.x-b.x,2)+std::pow(a.y-b.y,2)); }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr kf_sub_,ekf_sub_,pf_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr done_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr dropout_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  Pose2D gt_,kf_,ekf_,pf_;
  bool gt_received_=false,kf_received_=false,ekf_received_=false,pf_received_=false;
  double kf_cov_x_=0,kf_cov_y_=0,kf_cov_t_=0;
  double ekf_cov_x_=0,ekf_cov_y_=0,ekf_cov_t_=0;
  double pf_cov_x_=0,pf_cov_y_=0,pf_cov_t_=0;
  double sq_kf_=0,sq_ekf_=0,sq_pf_=0;
  int n_=0;
  double landmark_x_,landmark_y_;
  bool was_dropout_=false,done_=false;
  bool trajectory_started_=false;
  rclcpp::Time start_time_,traj_start_time_;
  std::string output_dir_,plot_script_;
  std::vector<DropoutInterval> dropouts_;
  std::ofstream poses_file_,rmse_file_,cov_file_,landmark_file_,dropout_file_;
};

int main(int argc,char**argv)
{ rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<EvaluationNode>()); rclcpp::shutdown(); }
