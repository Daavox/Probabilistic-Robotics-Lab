/**
 * Path Publisher Node
 *
 * Collects poses from all filters and ground truth, and publishes them
 * as nav_msgs/Path topics for visualization in RViz.
 *
 * Recording only starts when the robot actually moves (/cmd_vel > threshold).
 * This prevents pre-movement jitter from appearing in the trajectory plots.
 *
 * Published paths grow over time (poses are appended, never cleared):
 *   /ground_truth_path  — odometry ground truth
 *   /kf_path            — Kalman Filter estimated trajectory
 *   /ekf_path           — Extended Kalman Filter estimated trajectory
 *   /pf_path            — Particle Filter estimated trajectory
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

class PathPublisherNode : public rclcpp::Node
{
public:
  PathPublisherNode() : Node("path_publisher_node"), recording_(false)
  {
    // Publishers — one Path topic per filter + ground truth
    gt_pub_  = this->create_publisher<nav_msgs::msg::Path>("/ground_truth_path", 10);
    kf_pub_  = this->create_publisher<nav_msgs::msg::Path>("/kf_path", 10);
    ekf_pub_ = this->create_publisher<nav_msgs::msg::Path>("/ekf_path", 10);
    pf_pub_  = this->create_publisher<nav_msgs::msg::Path>("/pf_path", 10);

    // Subscribers — ground truth, all filter poses, and cmd_vel for start detection
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&PathPublisherNode::odomCb, this, std::placeholders::_1));
    kf_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/kf_pose", 10, std::bind(&PathPublisherNode::kfCb, this, std::placeholders::_1));
    ekf_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/ekf_pose", 10, std::bind(&PathPublisherNode::ekfCb, this, std::placeholders::_1));
    pf_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/pf_pose", 10, std::bind(&PathPublisherNode::pfCb, this, std::placeholders::_1));
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
      "/cmd_vel", 10, std::bind(&PathPublisherNode::cmdVelCb, this, std::placeholders::_1));

    // All paths published in odom frame except PF which publishes in map frame
    gt_path_.header.frame_id  = "odom";
    kf_path_.header.frame_id  = "odom";
    ekf_path_.header.frame_id = "odom";
    pf_path_.header.frame_id  = "map";

    RCLCPP_INFO(this->get_logger(), "Path Publisher started — waiting for movement");
  }

private:
  // Start recording only when robot actually moves — avoids pre-trajectory noise
  void cmdVelCb(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
  {
    if (!recording_ && (std::abs(msg->twist.linear.x) > 0.01 ||
                        std::abs(msg->twist.angular.z) > 0.01)) {
      recording_ = true;
      RCLCPP_INFO(this->get_logger(), "Recording started!");
    }
  }

  // Each callback appends the latest pose to its path and republishes the full path
  void odomCb(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!recording_) return;
    geometry_msgs::msg::PoseStamped ps;
    ps.header = msg->header;
    ps.pose   = msg->pose.pose;
    gt_path_.header.stamp = msg->header.stamp;
    gt_path_.poses.push_back(ps);
    gt_pub_->publish(gt_path_);
  }

  void kfCb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    if (!recording_) return;
    geometry_msgs::msg::PoseStamped ps;
    ps.header = msg->header;
    ps.pose   = msg->pose.pose;
    kf_path_.header.stamp = msg->header.stamp;
    kf_path_.poses.push_back(ps);
    kf_pub_->publish(kf_path_);
  }

  void ekfCb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    if (!recording_) return;
    geometry_msgs::msg::PoseStamped ps;
    ps.header = msg->header;
    ps.pose   = msg->pose.pose;
    ekf_path_.header.stamp = msg->header.stamp;
    ekf_path_.poses.push_back(ps);
    ekf_pub_->publish(ekf_path_);
  }

  void pfCb(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    if (!recording_) return;
    geometry_msgs::msg::PoseStamped ps;
    ps.header = msg->header;
    ps.pose   = msg->pose.pose;
    pf_path_.header.stamp = msg->header.stamp;
    pf_path_.poses.push_back(ps);
    pf_pub_->publish(pf_path_);
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gt_pub_, kf_pub_, ekf_pub_, pf_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    kf_sub_, ekf_sub_, pf_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;

  // Accumulated paths — each grows by one pose per incoming message
  nav_msgs::msg::Path gt_path_, kf_path_, ekf_path_, pf_path_;

  bool recording_;  // false until first cmd_vel movement detected
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPublisherNode>());
  rclcpp::shutdown();
  return 0;
}
