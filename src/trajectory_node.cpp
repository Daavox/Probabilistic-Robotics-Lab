/**
 * Trajectory Node
 *
 * Controls the TurtleBot3 to drive a U-shaped trajectory using a
 * simple state machine. Reads current pose from /odom and publishes
 * velocity commands to /cmd_vel.
 *
 *
 * When done, publishes true on /trajectory_done which triggers the
 * evaluation node to flush CSVs and generate all plots.
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

using namespace std::chrono_literals;

// State machine states for the U-curve trajectory
enum class State { STRAIGHT_1, TURN_RIGHT_1, STRAIGHT_2, TURN_RIGHT_2, STRAIGHT_3, DONE };

class TrajectoryNode : public rclcpp::Node
{
public:
  TrajectoryNode() : Node("trajectory_node"), state_(State::STRAIGHT_1)
  {
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);
    done_pub_    = this->create_publisher<std_msgs::msg::Bool>("/trajectory_done", 10);

    // /odom provides current robot pose for distance and angle calculations
    odom_sub_    = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&TrajectoryNode::odomCallback, this, std::placeholders::_1));

    // Control loop runs at 10 Hz, advancing the state machine each tick
    timer_ = this->create_wall_timer(100ms, std::bind(&TrajectoryNode::controlLoop, this));
    RCLCPP_INFO(this->get_logger(), "Trajectory Node started — U-curve");
  }

private:
  // Update current robot pose from odometry; record start pose on first message
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    x_ = msg->pose.pose.position.x;
    y_ = msg->pose.pose.position.y;
    tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    double r, p; tf2::Matrix3x3(q).getRPY(r, p, theta_);
    if (!initialized_) {
      start_x_ = x_; start_y_ = y_; start_theta_ = theta_;
      initialized_ = true;
      RCLCPP_INFO(this->get_logger(), "Start: x=%.2f y=%.2f theta=%.2f", x_, y_, theta_);
    }
  }

  void publishVel(double v, double w)
  {
    geometry_msgs::msg::TwistStamped msg;
    msg.header.stamp = this->now();
    msg.twist.linear.x  = v;
    msg.twist.angular.z = w;
    cmd_vel_pub_->publish(msg);
  }

  void stop() { publishVel(0, 0); }

  double normalizeAngle(double a)
  {
    while (a >  M_PI) a -= 2*M_PI;
    while (a < -M_PI) a += 2*M_PI;
    return a;
  }

  // State machine: each state checks a condition and transitions when met
  void controlLoop()
  {
    if (!initialized_) return;
    const double LIN = 0.15;   // linear speed [m/s]
    const double ANG = 0.4;    // angular speed [rad/s]
    const double ATOL = 0.03;  // angle tolerance [rad]

    switch (state_) {
      case State::STRAIGHT_1: {
        double d = std::hypot(x_-start_x_, y_-start_y_);
        if (d < 2.5) publishVel(LIN, 0);
        else {
          stop(); target_theta_ = normalizeAngle(start_theta_ - M_PI/2);
          RCLCPP_INFO(this->get_logger(), "STRAIGHT_1 done");
          state_ = State::TURN_RIGHT_1;
        }
        break;
      }
      case State::TURN_RIGHT_1: {
        double err = normalizeAngle(target_theta_ - theta_);
        if (std::abs(err) > ATOL) publishVel(0, err>0?ANG:-ANG);
        else { stop(); wp_x_=x_; wp_y_=y_;
          RCLCPP_INFO(this->get_logger(), "TURN_RIGHT_1 done");
          state_ = State::STRAIGHT_2; }
        break;
      }
      case State::STRAIGHT_2: {
        double d = std::hypot(x_-wp_x_, y_-wp_y_);
        if (d < 1.0) publishVel(LIN, 0);
        else {
          stop(); target_theta_ = normalizeAngle(start_theta_ - M_PI);
          RCLCPP_INFO(this->get_logger(), "STRAIGHT_2 done");
          state_ = State::TURN_RIGHT_2;
        }
        break;
      }
      case State::TURN_RIGHT_2: {
        double err = normalizeAngle(target_theta_ - theta_);
        if (std::abs(err) > ATOL) publishVel(0, err>0?ANG:-ANG);
        else { stop(); wp_x_=x_; wp_y_=y_;
          RCLCPP_INFO(this->get_logger(), "TURN_RIGHT_2 done");
          state_ = State::STRAIGHT_3; }
        break;
      }
      case State::STRAIGHT_3: {
        double d = std::hypot(x_-wp_x_, y_-wp_y_);
        if (d < 1.5) publishVel(LIN, 0);
        else {
          stop();
          RCLCPP_INFO(this->get_logger(), "U-curve DONE!");
          state_ = State::DONE;
          // Signal evaluation node to save results and generate plots
          std_msgs::msg::Bool done_msg;
          done_msg.data = true;
          done_pub_->publish(done_msg);
        }
        break;
      }
      case State::DONE: stop(); break;
    }
  }

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr               done_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  State state_;
  bool initialized_ = false;
  double x_=0, y_=0, theta_=0;          // current robot pose from /odom
  double start_x_=0, start_y_=0, start_theta_=0;  // pose at trajectory start
  double target_theta_=0, wp_x_=0, wp_y_=0;       // waypoint for current state
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryNode>());
  rclcpp::shutdown();
  return 0;
}
