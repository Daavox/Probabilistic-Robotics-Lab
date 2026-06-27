#include <cmath>
#include <string>
#include <array>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// ── 3x3 matrix and 3-vector types (Eigen-free implementation) ─────────────
using Mat3 = std::array<std::array<double,3>,3>;
using Vec3 = std::array<double,3>;

// ── Matrix helper functions ───────────────────────────────────────────────
static Mat3 eye3()   // 3x3 identity matrix
{ Mat3 m{}; m[0][0]=m[1][1]=m[2][2]=1.0; return m; }

static Mat3 mul33(const Mat3&A,const Mat3&B)  // matrix multiplication A·B
{ Mat3 C{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
    C[i][j]+=A[i][k]*B[k][j]; return C; }

static Mat3 tr3(const Mat3&A)  // matrix transpose Aᵀ
{ Mat3 T{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) T[i][j]=A[j][i]; return T; }

static Vec3 mv3(const Mat3&A,const Vec3&v)  // matrix-vector product A·v
{ Vec3 r{}; for(int i=0;i<3;++i) for(int k=0;k<3;++k) r[i]+=A[i][k]*v[k]; return r; }

static Mat3 add33(const Mat3&A,const Mat3&B)  // matrix addition A + B
{ Mat3 R{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) R[i][j]=A[i][j]+B[i][j]; return R; }

static Mat3 sub33(const Mat3&A,const Mat3&B)  // matrix subtraction A - B
{ Mat3 R{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) R[i][j]=A[i][j]-B[i][j]; return R; }

static Mat3 inv3(const Mat3&M)  // 3x3 matrix inverse via cofactor expansion
{
  double d=M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
           -M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
           +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
  Mat3 inv{};
  inv[0][0]=(M[1][1]*M[2][2]-M[1][2]*M[2][1])/d;
  inv[0][1]=-(M[0][1]*M[2][2]-M[0][2]*M[2][1])/d;
  inv[0][2]=(M[0][1]*M[1][2]-M[0][2]*M[1][1])/d;
  inv[1][0]=-(M[1][0]*M[2][2]-M[1][2]*M[2][0])/d;
  inv[1][1]=(M[0][0]*M[2][2]-M[0][2]*M[2][0])/d;
  inv[1][2]=-(M[0][0]*M[1][2]-M[0][2]*M[1][0])/d;
  inv[2][0]=(M[1][0]*M[2][1]-M[1][1]*M[2][0])/d;
  inv[2][1]=-(M[0][0]*M[2][1]-M[0][1]*M[2][0])/d;
  inv[2][2]=(M[0][0]*M[1][1]-M[0][1]*M[1][0])/d;
  return inv;
}

// ── Helper: signed angle difference, result in [-π, π] ───────────────────
static double angle_diff(double a,double b)
{ return std::atan2(std::sin(a-b),std::cos(a-b)); }

// ── Helper: extract yaw angle from quaternion (odom orientation) ──────────
static double yaw_from_odom(const nav_msgs::msg::Odometry::SharedPtr&msg)
{
  tf2::Quaternion q(msg->pose.pose.orientation.x,msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z,msg->pose.pose.orientation.w);
  double r,p,y; tf2::Matrix3x3(q).getRPY(r,p,y); return y;
}

class EKFNode : public rclcpp::Node
{
public:
  EKFNode() : Node("ekf_node"), last_time_(-1.0), last_v_(0.0), last_w_(0.0),
              have_cmd_(false), in_dropout_(false)
  {
    // ── ROS2 parameters (tunable from launch file) ───────────────────────
    // Q = process noise  → how much we trust the motion model
    // R = measurement noise → how much we trust the sensor (/odom)
    declare_parameter("process_noise_q",     0.01);
    declare_parameter("measurement_noise_r", 0.05);
    declare_parameter("output_topic", std::string("/ekf_pose"));
    declare_parameter("frame_id",     std::string("odom"));

    q_        = get_parameter("process_noise_q").as_double();
    r_        = get_parameter("measurement_noise_r").as_double();
    auto out  = get_parameter("output_topic").as_string();
    frame_id_ = get_parameter("frame_id").as_string();

    // ── ROBOT STATE initialization ───────────────────────────────────────
    // x_ = state vector [x, y, θ] — best estimate of robot pose
    // P_ = covariance matrix — uncertainty around the state estimate
    x_ = {0,0,0};
    P_ = eye3();

    // Q_ = process noise covariance (motion model uncertainty)
    // → small Q: trust the motion model strongly, covariance grows slowly
    // → large Q: distrust motion model, covariance grows faster
    for(auto&row:Q_) row.fill(0.0);
    Q_[0][0]=q_; Q_[1][1]=q_; Q_[2][2]=q_;

    // R_ = measurement noise covariance (sensor uncertainty)
    // → small R: trust sensor strongly, correction pulls estimate to z
    // → large R: distrust sensor, correction has little effect
    for(auto&row:R_) row.fill(0.0);
    R_[0][0]=r_; R_[1][1]=r_; R_[2][2]=r_;

    // ── Subscriptions ────────────────────────────────────────────────────
    // /cmd_vel → control input u = [v, ω] used in MOTION MODEL (PREDICT)
    sub_cmd_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "/cmd_vel", 10,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr m){
        last_v_=m->twist.linear.x;   // linear velocity v
        last_w_=m->twist.angular.z;  // angular velocity ω
        have_cmd_=true;});

    // /odom → measurement z = [x, y, θ] used in SENSOR MODEL (UPDATE)
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&EKFNode::odom_cb, this, std::placeholders::_1));

    // /measurement_dropout → if true, skip the UPDATE step (sensor lost)
    sub_dropout_ = create_subscription<std_msgs::msg::Bool>(
      "/measurement_dropout", 10,
      [this](const std_msgs::msg::Bool::SharedPtr m){ in_dropout_ = m->data; });

    pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(out, 10);
    RCLCPP_INFO(get_logger(), "EKF Node started — Q=%.4f R=%.4f", q_, r_);
  }

private:
  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double now = get_clock()->now().nanoseconds() * 1e-9;

    // ── First message: initialize robot state from odometry ──────────────
    if (last_time_ < 0.0) {
      x_[0]=msg->pose.pose.position.x;  // initial x
      x_[1]=msg->pose.pose.position.y;  // initial y
      x_[2]=yaw_from_odom(msg);         // initial θ
      last_time_=now; return;
    }

    // Time step dt between two odom messages
    double dt = now - last_time_;
    last_time_ = now;
    if (dt<=0.0||dt>1.0) return;  // skip if dt invalid

    // Control input: prefer /cmd_vel, fall back to odom twist if not received
    double v = have_cmd_ ? last_v_ : msg->twist.twist.linear.x;
    double w = have_cmd_ ? last_w_ : msg->twist.twist.angular.z;
    double theta = x_[2];  // current heading — needed for nonlinear model

    // ════════════════════════════════════════════════════════════════════
    // MOTION MODEL (PREDICT step) — always runs, even during dropout
    //
    // EKF uses the full nonlinear motion model g(u, x):
    //   x += v · cos(θ) · dt     (nonlinear: depends on current θ)
    //   y += v · sin(θ) · dt     (nonlinear: depends on current θ)
    //   θ += ω · dt
    //
    // This is MORE ACCURATE than KF which uses a fixed B matrix,
    // because the trigonometric relationship is applied exactly.
    // ════════════════════════════════════════════════════════════════════

    // Apply nonlinear motion model directly to state vector
    x_[0] += v*dt*std::cos(theta);  // Δx = v · cos(θ) · dt
    x_[1] += v*dt*std::sin(theta);  // Δy = v · sin(θ) · dt
    x_[2] += w*dt;                  // Δθ = ω · dt
    x_[2]  = std::atan2(std::sin(x_[2]),std::cos(x_[2]));  // normalize θ

    // Jacobian F = ∂g/∂x — linearization of g around current state
    // This is the KEY difference to KF:
    // KF fixes B once, EKF recomputes F fresh at every timestep
    // F captures how a small change in θ affects the predicted x and y
    Mat3 F = eye3();
    F[0][2] = -v*dt*std::sin(theta);  // ∂(Δx)/∂θ = -v · sin(θ) · dt
    F[1][2] =  v*dt*std::cos(theta);  // ∂(Δy)/∂θ =  v · cos(θ) · dt

    // Predicted covariance: P̄ = F·P·Fᵀ + Q  (uncertainty grows with motion)
    P_ = add33(mul33(mul33(F,P_),tr3(F)), Q_);

    // ════════════════════════════════════════════════════════════════════
    // SENSOR MODEL (UPDATE step) — skipped during sensor dropout
    //
    // Measurement model is linear: z = H·x with H = Identity
    // (we directly measure [x, y, θ] from odometry)
    // ════════════════════════════════════════════════════════════════════
    if (!in_dropout_) {

      // Measurement vector z from odometry sensor
      Vec3 z = {msg->pose.pose.position.x,  // measured x
                msg->pose.pose.position.y,  // measured y
                yaw_from_odom(msg)};        // measured θ

      // Innovation: difference between measurement and predicted state
      // angle_diff used for θ to handle wrap-around correctly
      Vec3 innov = {z[0]-x_[0], z[1]-x_[1], angle_diff(z[2],x_[2])};

      // Innovation covariance: S = P̄ + R  (H = Identity so H·P̄·Hᵀ = P̄)
      Mat3 S = add33(P_, R_);

      // Kalman gain: K = P̄·S⁻¹
      // → K balances trust between prediction and measurement
      // → small R: K large → sensor dominates correction
      // → large R: K small → prediction dominates, sensor barely used
      Mat3 K = mul33(P_, inv3(S));

      // Corrected state: x = x̄ + K·innovation
      Vec3 Kinn = mv3(K, innov);
      for(int i=0;i<3;++i) x_[i]+=Kinn[i];
      x_[2] = std::atan2(std::sin(x_[2]),std::cos(x_[2]));  // normalize θ

      // Corrected covariance: P = (I − K)·P̄  (uncertainty shrinks)
      P_ = mul33(sub33(eye3(),K), P_);

    }
    // During dropout: x_ and P_ keep the predicted values from MOTION MODEL
    // P_ keeps growing each step → uncertainty increases until sensor returns

    // ── Publish estimated robot pose with covariance ─────────────────────
    geometry_msgs::msg::PoseWithCovarianceStamped out_msg;
    out_msg.header.stamp    = get_clock()->now();
    out_msg.header.frame_id = frame_id_;
    out_msg.pose.pose.position.x = x_[0];  // estimated x
    out_msg.pose.pose.position.y = x_[1];  // estimated y
    tf2::Quaternion q; q.setRPY(0,0,x_[2]);
    out_msg.pose.pose.orientation = tf2::toMsg(q);
    // Diagonal elements of P represent uncertainty in x, y, θ
    out_msg.pose.covariance[0]  = P_[0][0];  // variance x
    out_msg.pose.covariance[7]  = P_[1][1];  // variance y
    out_msg.pose.covariance[35] = P_[2][2];  // variance θ
    pub_->publish(out_msg);
  }

  // ── ROBOT STATE variables ────────────────────────────────────────────
  Vec3 x_;   // state vector [x, y, θ] — best estimate of robot pose
  Mat3 P_;   // covariance matrix — uncertainty of the state estimate
  Mat3 Q_;   // process noise covariance (motion model noise)
  Mat3 R_;   // measurement noise covariance (sensor noise)

  double q_, r_;           // scalar noise values from parameters
  double last_time_;       // timestamp of last odom message
  double last_v_, last_w_; // last received control inputs [v, ω]
  bool have_cmd_;          // true once /cmd_vel has been received
  bool in_dropout_;        // true when sensor dropout is active
  std::string frame_id_;

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_cmd_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_dropout_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;
};

int main(int argc,char**argv)
{ rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<EKFNode>()); rclcpp::shutdown(); }
