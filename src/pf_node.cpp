#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <numeric>
#include <limits>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// ── Helper: extract yaw from quaternion ──────────────────────────────────
static double yaw_from_quat(const geometry_msgs::msg::Quaternion&q){
  tf2::Quaternion tq(q.x,q.y,q.z,q.w);double r,p,y;tf2::Matrix3x3(tq).getRPY(r,p,y);return y;}

// ── Helper: convert yaw angle to quaternion for publishing ────────────────
static geometry_msgs::msg::Quaternion quat_from_yaw(double yaw){
  tf2::Quaternion q;q.setRPY(0,0,yaw);return tf2::toMsg(q);}

class ParticleFilterNode : public rclcpp::Node
{
public:
  ParticleFilterNode()
  : Node("pf_node"),last_time_(-1.0),last_v_(0.0),last_w_(0.0),
    have_cmd_(false),in_dropout_(false)
  {
    // ── ROS2 parameters (tunable from launch file) ───────────────────────
    // num_particles: more particles = more accurate but slower
    // init_spread:   how widely particles are scattered at initialization
    // process_noise_v/w: noise added to motion model (equivalent to Q)
    //   → small: particles move tightly together (trust model)
    //   → large: particles spread out (distrust model, more robust)
    // measurement_noise_r/yaw: uncertainty of /odom measurement (equivalent to R)
    //   → small: weights pulled strongly toward measurement
    //   → large: measurement has little effect on weights
    declare_parameter("num_particles",        500);
    declare_parameter("init_spread_x",        0.5);
    declare_parameter("init_spread_y",        0.5);
    declare_parameter("init_spread_yaw",      0.3);
    declare_parameter("process_noise_v",      0.3);
    declare_parameter("process_noise_w",      0.2);
    declare_parameter("measurement_noise_r",  0.3);
    declare_parameter("measurement_noise_yaw",0.3);
    declare_parameter("output_topic",    std::string("/pf_pose"));
    declare_parameter("particles_topic", std::string("/pf_particles"));
    declare_parameter("frame_id",        std::string("odom"));

    N_         = get_parameter("num_particles").as_int();
    spread_x_  = get_parameter("init_spread_x").as_double();
    spread_y_  = get_parameter("init_spread_y").as_double();
    spread_yaw_= get_parameter("init_spread_yaw").as_double();
    sigma_v_   = get_parameter("process_noise_v").as_double();
    sigma_w_   = get_parameter("process_noise_w").as_double();
    sigma_z_   = get_parameter("measurement_noise_r").as_double();
    sigma_yaw_ = get_parameter("measurement_noise_yaw").as_double();
    frame_id_  = get_parameter("frame_id").as_string();
    auto out   = get_parameter("output_topic").as_string();
    auto parts = get_parameter("particles_topic").as_string();

    rng_.seed(42);  // fixed seed for reproducibility

    // ── ROBOT STATE initialization ───────────────────────────────────────
    // particles_ = N hypotheses of robot pose, each [x, y, θ]
    // weights_   = probability of each hypothesis (uniform at start)
    // These two together ARE the robot state in the particle filter
    particles_.resize(N_,{0,0,0});
    weights_.assign(N_,1.0/N_);  // uniform weights: all hypotheses equally likely

    // ── Subscriptions ────────────────────────────────────────────────────
    // /cmd_vel → control input u = [v, ω] for MOTION MODEL (PREDICT step)
    sub_cmd_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "/cmd_vel",10,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr m){
        last_v_=m->twist.linear.x;   // linear velocity v
        last_w_=m->twist.angular.z;  // angular velocity ω
        have_cmd_=true;});

    // /odom → measurement z = [x, y, θ] for SENSOR MODEL (UPDATE step)
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom",10,std::bind(&ParticleFilterNode::odom_cb,this,std::placeholders::_1));

    // /measurement_dropout → if true, skip UPDATE and RESAMPLE
    sub_dropout_ = create_subscription<std_msgs::msg::Bool>(
      "/measurement_dropout",10,
      [this](const std_msgs::msg::Bool::SharedPtr m){ in_dropout_=m->data; });

    pub_pose_  = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(out,10);
    pub_parts_ = create_publisher<geometry_msgs::msg::PoseArray>(parts,10);  // for RViz arrows

    RCLCPP_INFO(get_logger(),"PF Node started — N=%d sigma_v=%.2f sigma_w=%.2f",
      N_,sigma_v_,sigma_w_);
  }

private:
  // A single particle: [x, y, θ] — one hypothesis of robot pose
  using Particle=std::array<double,3>;

  void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    double now=get_clock()->now().nanoseconds()*1e-9;

    // ── First message: initialize ROBOT STATE (particle cloud) ───────────
    // Scatter N particles around the initial odom position with Gaussian spread
    // This represents high initial uncertainty — we don't know exactly where we are
    if (last_time_<0.0) {
      double ix=msg->pose.pose.position.x;
      double iy=msg->pose.pose.position.y;
      double iyaw=yaw_from_quat(msg->pose.pose.orientation);
      // Each particle drawn from Gaussian centered on initial odom reading
      std::normal_distribution<double> dx(ix,spread_x_),dy(iy,spread_y_),dw(iyaw,spread_yaw_);
      for(auto&p:particles_){p[0]=dx(rng_);p[1]=dy(rng_);p[2]=dw(rng_);}
      last_time_=now; return;
    }

    // Time step dt between two odom messages
    double dt=now-last_time_; last_time_=now;
    if (dt<=0.0||dt>1.0) return;

    // Control input: prefer /cmd_vel, fall back to odom twist if not received
    double v=have_cmd_?last_v_:msg->twist.twist.linear.x;
    double w=have_cmd_?last_w_:msg->twist.twist.angular.z;

    // Noise distributions for the motion model
    std::normal_distribution<double> nv(0.0,sigma_v_),nw(0.0,sigma_w_);

    // ════════════════════════════════════════════════════════════════════
    // MOTION MODEL (PREDICT step) — always runs, even during dropout
    //
    // Each particle is propagated independently using the same nonlinear
    // motion model as EKF, but with added random noise (sigma_v, sigma_w).
    // This noise makes the particle cloud spread out over time, reflecting
    // growing uncertainty about the robot's true position.
    //
    // No linearization needed — the full nonlinear model is applied directly
    // to each particle. This is a key advantage over KF/EKF.
    // ════════════════════════════════════════════════════════════════════
    for(auto&p:particles_){
      double vn=v+nv(rng_);  // noisy linear velocity for this particle
      double wn=w+nw(rng_);  // noisy angular velocity for this particle
      // Apply nonlinear differential drive model to move this particle
      p[0]+=vn*dt*std::cos(p[2]);  // Δx = v_noisy · cos(θ) · dt
      p[1]+=vn*dt*std::sin(p[2]);  // Δy = v_noisy · sin(θ) · dt
      p[2]+=wn*dt;                  // Δθ = ω_noisy · dt
      p[2]=std::atan2(std::sin(p[2]),std::cos(p[2]));  // normalize θ
    }

    // ════════════════════════════════════════════════════════════════════
    // SENSOR MODEL (UPDATE step) — skipped during sensor dropout
    //
    // Each particle is assigned a weight based on how well its predicted
    // position matches the actual sensor measurement from /odom.
    // Particles close to the measurement get high weights,
    // particles far away get low weights.
    //
    // Log-weights are used instead of direct weights to avoid numerical
    // underflow when probabilities become very small.
    // ════════════════════════════════════════════════════════════════════
    if (!in_dropout_) {
      // Measurement z from odometry sensor
      double z_x=msg->pose.pose.position.x;    // measured x
      double z_y=msg->pose.pose.position.y;    // measured y
      double z_yaw=yaw_from_quat(msg->pose.pose.orientation);  // measured θ

      // Precompute inverse variances for Gaussian likelihood
      double inv2z2=1.0/(2.0*sigma_z_*sigma_z_);    // position likelihood
      double inv2y2=1.0/(2.0*sigma_yaw_*sigma_yaw_); // yaw likelihood

      // Compute log-likelihood for each particle
      // log p(z | particle_i) = -((dx² + dy²)/2σ_z² + dθ²/2σ_yaw²)
      // Higher = particle is closer to measurement = more likely to be correct
      std::vector<double> log_w(N_);
      double log_max=-std::numeric_limits<double>::infinity();
      for(int i=0;i<N_;++i){
        double dx=particles_[i][0]-z_x;  // position error x
        double dy=particles_[i][1]-z_y;  // position error y
        // Angle difference wrapped to [-π, π]
        double dyaw=std::atan2(std::sin(particles_[i][2]-z_yaw),
                               std::cos(particles_[i][2]-z_yaw));
        // Log-likelihood: Gaussian sensor model in log space
        log_w[i]=-(dx*dx+dy*dy)*inv2z2-dyaw*dyaw*inv2y2;
        log_max=std::max(log_max,log_w[i]);  // track maximum for numerical stability
      }

      // Convert log-weights to weights and multiply with prior weights
      // Subtract log_max before exp() to prevent numerical underflow
      double sum=0.0;
      for(int i=0;i<N_;++i){
        weights_[i]=std::exp(log_w[i]-log_max)*weights_[i];
        sum+=weights_[i];
      }
      // Normalize weights so they sum to 1
      if(sum>1e-12){for(auto&ww:weights_)ww/=sum;}
      else{weights_.assign(N_,1.0/N_);}  // reset if all weights collapsed

      // ── RESAMPLE when ESS < N/2 ─────────────────────────────────────
      // ESS (Effective Sample Size) measures how many particles contribute
      // meaningfully. ESS = 1/Σ(w²). Low ESS = most weight on few particles.
      // When ESS drops below N/2, resample to avoid particle degeneracy.
      double ess=0.0;
      for(auto ww:weights_)ess+=ww*ww; ess=1.0/ess;
      if(ess<N_/2.0){
        // Systematic resampling: draw new particle set proportional to weights
        // Particles with high weight are copied multiple times,
        // particles with low weight are discarded
        particles_=systematic_resample(particles_,weights_);
        weights_.assign(N_,1.0/N_);  // reset to uniform after resampling
      }
    }
    // During dropout: weights unchanged, particles only spread from PREDICT
    // The particle cloud grows but no hypotheses are eliminated
    // → PF is naturally robust to sensor loss (no single estimate to drift)

    // ════════════════════════════════════════════════════════════════════
    // ESTIMATE — compute final robot state from weighted particle set
    //
    // The estimated pose is the weighted mean over all particles.
    // The covariance is the weighted variance — directly reflects
    // how spread out the particles are (honest uncertainty estimate).
    // ════════════════════════════════════════════════════════════════════

    // Weighted mean position
    double mx=0,my=0,ss=0,cs=0;
    for(int i=0;i<N_;++i){
      mx+=weights_[i]*particles_[i][0];           // weighted mean x
      my+=weights_[i]*particles_[i][1];           // weighted mean y
      ss+=weights_[i]*std::sin(particles_[i][2]); // for circular mean θ
      cs+=weights_[i]*std::cos(particles_[i][2]); // for circular mean θ
    }
    double myaw=std::atan2(ss,cs);  // circular mean of θ (correct for angles)

    // Weighted variance — represents uncertainty of the estimate
    double vx=0,vy=0,vyaw=0;
    for(int i=0;i<N_;++i){
      double ex=particles_[i][0]-mx;   // deviation from mean x
      double ey=particles_[i][1]-my;   // deviation from mean y
      double et=particles_[i][2]-myaw; // deviation from mean θ
      vx+=weights_[i]*ex*ex;     // weighted variance x
      vy+=weights_[i]*ey*ey;     // weighted variance y
      vyaw+=weights_[i]*et*et;   // weighted variance θ
    }

    // ── Publish estimated robot pose with covariance ─────────────────────
    geometry_msgs::msg::PoseWithCovarianceStamped pm;
    pm.header.stamp=get_clock()->now(); pm.header.frame_id=frame_id_;
    pm.pose.pose.position.x=mx;   // estimated x (weighted mean)
    pm.pose.pose.position.y=my;   // estimated y (weighted mean)
    pm.pose.pose.orientation=quat_from_yaw(myaw);  // estimated θ
    pm.pose.covariance[0]=vx;     // variance x (spread of particles)
    pm.pose.covariance[7]=vy;     // variance y (spread of particles)
    pm.pose.covariance[35]=vyaw;  // variance θ (spread of particles)
    pub_pose_->publish(pm);

    // ── Publish all particles as PoseArray for RViz visualization ────────
    // This shows the particle cloud as arrows in RViz
    geometry_msgs::msg::PoseArray pa; pa.header=pm.header; pa.poses.reserve(N_);
    for(const auto&p:particles_){
      geometry_msgs::msg::Pose pose;
      pose.position.x=p[0]; pose.position.y=p[1];
      pose.orientation=quat_from_yaw(p[2]); pa.poses.push_back(pose);
    }
    pub_parts_->publish(pa);
  }

  // ── Systematic Resampling ────────────────────────────────────────────
  // Draws N new particles from the current set proportional to their weights.
  // Uses a single random offset r and equally spaced thresholds to ensure
  // even coverage — more efficient and less noisy than multinomial resampling.
  std::vector<Particle> systematic_resample(
    const std::vector<Particle>&parts,const std::vector<double>&w)
  {
    std::uniform_real_distribution<double> uni(0.0,1.0);
    double r=uni(rng_)/N_;  // single random starting offset
    std::vector<Particle> out; out.reserve(N_);
    double cum=w[0]; int i=0;
    for(int j=0;j<N_;++j){
      double thr=r+(double)j/N_;  // equally spaced thresholds
      // Advance i until cumulative weight exceeds threshold
      while(thr>cum&&i<N_-1){++i;cum+=w[i];}
      out.push_back(parts[i]);  // copy particle i into new set
    }
    return out;
  }

  // ── ROBOT STATE variables ────────────────────────────────────────────
  // The robot state IS the particle set — not a single vector like KF/EKF
  std::vector<Particle> particles_;  // N hypotheses: each [x, y, θ]
  std::vector<double> weights_;      // probability of each hypothesis

  int N_;                              // number of particles
  double spread_x_,spread_y_,spread_yaw_;  // initialization spread
  double sigma_v_,sigma_w_;           // process noise (equivalent to Q)
  double sigma_z_,sigma_yaw_;         // measurement noise (equivalent to R)
  double last_time_,last_v_,last_w_;  // timing and last control input
  bool have_cmd_,in_dropout_;
  std::string frame_id_;
  std::mt19937 rng_;                  // random number generator

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_cmd_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_dropout_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_parts_;
};

int main(int argc,char**argv)
{ rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<ParticleFilterNode>()); rclcpp::shutdown(); }
