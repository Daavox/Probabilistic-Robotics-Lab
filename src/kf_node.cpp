#include <array>
#include <cmath>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <Eigen/Dense>

// ── Helper: extract yaw angle from quaternion (odom orientation) ──────────
static double yaw_from_odom(const nav_msgs::msg::Odometry::SharedPtr & msg)
{
  tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  double r, p, y; tf2::Matrix3x3(q).getRPY(r, p, y); return y;
}

// ── Helper: keep angle in [-π, π] to avoid wrap-around errors ────────────
static double normalize_angle(double a)
{
  while (a >  M_PI) a -= 2*M_PI;
  while (a < -M_PI) a += 2*M_PI;
  return a;
}

class KalmanFilterNode : public rclcpp::Node
{
public:
  KalmanFilterNode() : Node("kf_node"), last_time_(-1.0),
                       last_v_(0.0), last_w_(0.0),
                       have_cmd_(false), in_dropout_(false)
  {
    // ── ROS2 parameters (tunable from launch file) ───────────────────────
    // Q = process noise  → how much we trust the motion model
    // R = measurement noise → how much we trust the sensor (/odom)
    declare_parameter("process_noise_q",     0.01);
    declare_parameter("measurement_noise_r", 0.05);
    declare_parameter("output_topic", std::string("/kf_pose"));
    declare_parameter("frame_id",     std::string("odom"));

    q_        = get_parameter("process_noise_q").as_double();
    r_        = get_parameter("measurement_noise_r").as_double();
    auto out  = get_parameter("output_topic").as_string();
    frame_id_ = get_parameter("frame_id").as_string();

    // ── ROBOT STATE initialization ───────────────────────────────────────
    // μ  = mean state vector [x, y, θ] — the best estimate of robot pose
    // Σ  = covariance matrix — uncertainty around the state estimate
    mu_    = Eigen::Vector3d::Zero();      // start at origin, unknown
    Sigma_ = Eigen::Matrix3d::Identity();  // high initial uncertainty

    // A = state transition matrix (Identity: state does not change on its own)
    A_ = Eigen::Matrix3d::Identity();

    // C = measurement matrix (Identity: we directly measure [x, y, θ])
    C_ = Eigen::Matrix3d::Identity();

    // R_ = process noise covariance matrix Q (model uncertainty)
    // → small Q: trust the motion model strongly
    // → large Q: allow larger deviation from predicted state
    R_ = Eigen::Matrix3d::Zero();
    R_(0,0) = q_; R_(1,1) = q_; R_(2,2) = q_;

    // Q_ = measurement noise covariance matrix R (sensor uncertainty)
    // → small R: trust the sensor strongly, correction pulls estimate to z
    // → large R: distrust sensor, correction has little effect
    Q_ = Eigen::Matrix3d::Zero();
    Q_(0,0) = r_; Q_(1,1) = r_; Q_(2,2) = r_;

    // ── Subscriptions ────────────────────────────────────────────────────
    // /cmd_vel → control input u = [v, ω] for the motion model (PREDICT step)
    sub_cmd_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "/cmd_vel", 10,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr m){
        last_v_ = m->twist.linear.x;   // linear velocity v
        last_w_ = m->twist.angular.z;  // angular velocity ω
        have_cmd_ = true;
      });

    // /odom → measurement z = [x, y, θ] for the UPDATE step
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&KalmanFilterNode::odom_cb, this, std::placeholders::_1));

    // /measurement_dropout → if true, skip the UPDATE step (sensor lost)
    sub_dropout_ = create_subscription<std_msgs::msg::Bool>(
      "/measurement_dropout", 10,
      [this](const std_msgs::msg::Bool::SharedPtr m){ in_dropout_ = m->data; });

    pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(out, 10);
    RCLCPP_INFO(get_logger(), "KF Node started — Q=%.4f R=%.4f", q_, r_);
  }

private:
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double now = get_clock()->now().nanoseconds() * 1e-9;

    // ── First message: initialize robot state from odometry ──────────────
    if (last_time_ < 0.0) {
      mu_(0) = msg->pose.pose.position.x;  // initial x
      mu_(1) = msg->pose.pose.position.y;  // initial y
      mu_(2) = yaw_from_odom(msg);         // initial θ
      last_time_ = now;
      RCLCPP_INFO(get_logger(), "KF initialized at x=%.2f y=%.2f theta=%.2f",
        mu_(0), mu_(1), mu_(2));
      return;
    }

    // Time step dt between two odom messages
    double dt = now - last_time_;
    last_time_ = now;
    if (dt <= 0.0 || dt > 1.0) return;  // skip if dt invalid

    // Control input: prefer /cmd_vel, fall back to odom twist if not received
    double v = have_cmd_ ? last_v_ : msg->twist.twist.linear.x;
    double w = have_cmd_ ? last_w_ : msg->twist.twist.angular.z;

    // ════════════════════════════════════════════════════════════════════
    // MOTION MODEL (PREDICT step) — always runs, even during dropout
    // Uses control input u = [v, ω] to propagate the robot state forward
    // ════════════════════════════════════════════════════════════════════

    // B matrix: maps control input [v, ω] to state change [Δx, Δy, Δθ]
    // Linearized at current heading θ — this is the KEY difference to EKF:
    // KF fixes B once per step, EKF recomputes full Jacobian G each time
    Eigen::Matrix<double,3,2> B;
    B(0,0) = std::cos(mu_(2)) * dt;  // Δx = v · cos(θ) · dt
    B(0,1) = 0.0;
    B(1,0) = std::sin(mu_(2)) * dt;  // Δy = v · sin(θ) · dt
    B(1,1) = 0.0;
    B(2,0) = 0.0;
    B(2,1) = dt;                      // Δθ = ω · dt

    Eigen::Vector2d u(v, w);

    // Predicted state mean: μ̄ = A·μ + B·u
    Eigen::Vector3d mu_bar = A_ * mu_ + B * u;
    mu_bar(2) = normalize_angle(mu_bar(2));

    // Predicted state covariance: Σ̄ = A·Σ·Aᵀ + R  (uncertainty grows)
    Eigen::Matrix3d Sigma_bar = A_ * Sigma_ * A_.transpose() + R_;

    // ════════════════════════════════════════════════════════════════════
    // SENSOR MODEL (UPDATE step) — skipped during sensor dropout
    // Uses measurement z = [x, y, θ] from /odom to correct the prediction
    // ════════════════════════════════════════════════════════════════════
    if (!in_dropout_) {

      // Measurement vector z from odometry sensor
      Eigen::Vector3d z;
      z(0) = msg->pose.pose.position.x;  // measured x
      z(1) = msg->pose.pose.position.y;  // measured y
      z(2) = yaw_from_odom(msg);         // measured θ

      // Innovation covariance: S = C·Σ̄·Cᵀ + Q
      Eigen::Matrix3d S = C_ * Sigma_bar * C_.transpose() + Q_;

      // Kalman gain: K = Σ̄·Cᵀ·S⁻¹
      // → K balances trust between prediction and measurement
      // → high R (Q_): K small → correction weak → trust prediction more
      // → low R (Q_): K large → correction strong → trust sensor more
      Eigen::Matrix3d K = Sigma_bar * C_.transpose() * S.inverse();

      // Innovation: difference between measurement and predicted measurement
      Eigen::Vector3d innov = z - C_ * mu_bar;
      innov(2) = normalize_angle(innov(2));  // wrap angle difference

      // Corrected state mean: μ = μ̄ + K·(z − C·μ̄)
      mu_    = mu_bar + K * innov;
      mu_(2) = normalize_angle(mu_(2));

      // Corrected covariance: Σ = (I − K·C)·Σ̄  (uncertainty shrinks)
      Sigma_ = (Eigen::Matrix3d::Identity() - K * C_) * Sigma_bar;

    } else {
      // During dropout: only prediction, no correction
      // Covariance keeps growing → uncertainty increases until sensor returns
      mu_    = mu_bar;
      Sigma_ = Sigma_bar;
    }

    // ── Publish estimated robot pose with covariance ─────────────────────
    geometry_msgs::msg::PoseWithCovarianceStamped out_msg;
    out_msg.header.stamp    = get_clock()->now();
    out_msg.header.frame_id = frame_id_;
    out_msg.pose.pose.position.x = mu_(0);  // estimated x
    out_msg.pose.pose.position.y = mu_(1);  // estimated y
    tf2::Quaternion q; q.setRPY(0, 0, mu_(2));
    out_msg.pose.pose.orientation = tf2::toMsg(q);
    // Diagonal elements of Σ represent uncertainty in x, y, θ
    out_msg.pose.covariance[0]  = Sigma_(0,0);  // variance x
    out_msg.pose.covariance[7]  = Sigma_(1,1);  // variance y
    out_msg.pose.covariance[35] = Sigma_(2,2);  // variance θ
    pub_->publish(out_msg);
  }

  // ── ROBOT STATE variables ────────────────────────────────────────────
  Eigen::Vector3d mu_;     // state mean [x, y, θ] — best estimate of pose
  Eigen::Matrix3d Sigma_;  // state covariance — uncertainty of estimate
  Eigen::Matrix3d A_;      // state transition matrix (Identity)
  Eigen::Matrix3d C_;      // measurement matrix (Identity)
  Eigen::Matrix3d R_;      // process noise covariance Q (motion model noise)
  Eigen::Matrix3d Q_;      // measurement noise covariance R (sensor noise)

  double q_, r_;           // scalar noise values from parameters
  double last_time_;       // timestamp of last odom message
  double last_v_, last_w_; // last received control inputs
  bool have_cmd_;          // true once /cmd_vel has been received
  bool in_dropout_;        // true when sensor dropout is active
  std::string frame_id_;

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_cmd_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           sub_odom_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr               sub_dropout_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KalmanFilterNode>());
  rclcpp::shutdown();
}
