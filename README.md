# Probabilistic Robotics Lab — State Estimation with ROS2 & TurtleBot3

**Course:** Probabilistic Robotics Lab (FH Technikum Wien)  
**Environment:** ROS2 Jazzy · Ubuntu Noble · WSL2 · TurtleBot3 Burger · Gazebo Harmonic

---

## Overview

This project implements and evaluates three fundamental probabilistic state estimation methods for mobile robotics:

| Filter  | Method                             | Control Input | Measurement |
|---------|------------------------------------|---------------|-------------|
| **KF**  | Kalman Filter (linear, 4D state)   | `/cmd_vel`    | `/odom`     |
| **EKF** | Extended Kalman Filter (nonlinear) | `/cmd_vel`    | `/odom`     |
| **PF**  | Particle Filter (Monte Carlo)      | `/cmd_vel`    | `/odom`     |

All filters subscribe to `/measurement_dropout` to simulate sensor interruptions: during dropout only the **prediction step** runs, the **correction step** is skipped.

---

## Prerequisites

### System Requirements
- ROS2 Jazzy
- Ubuntu 24.04 (Noble) — native or WSL2
- Python 3.12 with `matplotlib`, `pandas`, `numpy`

### Required ROS2 Packages
```bash
sudo apt install ros-jazzy-turtlebot3-gazebo
sudo apt install ros-jazzy-nav2-bringup
sudo apt install ros-jazzy-nav2-amcl
sudo apt install ros-jazzy-nav2-map-server
sudo apt install ros-jazzy-nav2-lifecycle-manager
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
pip3 install matplotlib pandas numpy --break-system-packages
```

---

## Repository Structure

```
prob_lab/
├── src/
│   ├── kf_node.cpp           # Kalman Filter (4D state: x, y, vx, vy)
│   ├── ekf_node.cpp          # Extended Kalman Filter (3D state: x, y, θ)
│   ├── pf_node.cpp           # Particle Filter (Monte Carlo Localization)
│   ├── trajectory_node.cpp   # U-curve trajectory controller
│   ├── path_publisher_node.cpp # Publishes paths for RViz visualization
│   └── evaluation_node.cpp   # RMSE, covariance, landmark & dropout evaluation
├── maps/
│   ├── turtlebot3_world.pgm  # Map image
│   └── turtlebot3_world.yaml # Map metadata
├── launch/
│   └── prob_lab.launch.py    # Main launch file (starts everything)
├── rviz/
│   └── prob_lab.rviz         # RViz configuration
├── plot_results.py           # Generates all evaluation plots from CSV
└── results/                  # Auto-generated after experiment
    ├── poses.csv
    ├── rmse.csv
    ├── covariance.csv
    ├── landmark.csv
    ├── interruption.csv
    └── plots/
        ├── 01_trajectories.png
        ├── 02_rmse.png
        ├── 03_position_error.png
        ├── 04_covariance.png
        ├── 05_landmark.png
        ├── 06_dropout.png
        └── 07_rmse_bar.png
```

---

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select prob_lab
source install/setup.bash
```

---

## Run — One Command

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch prob_lab prob_lab.launch.py
```

### What happens automatically:
| Time | Event |
|------|-------|
| 0 s | Gazebo + all filter nodes + Nav2 localization start |
| 5 s | RViz opens |
| 8 s | Initial pose sent to AMCL (Nav2) |
| 15 s | TurtleBot3 starts U-curve trajectory |
| ~110 s | Trajectory finishes → CSVs saved → plots generated |

---

## RViz Setup

After launch, set **Fixed Frame** to `odom` in RViz. The following topics are pre-configured:

| Topic | Color | Description |
|-------|-------|-------------|
| `/ground_truth_path` | Green | Ground truth (odometry) |
| `/kf_path` | Red | Kalman Filter path |
| `/ekf_path` | Blue | Extended Kalman Filter path |
| `/pf_path` | Orange | Particle Filter path |
| `/pf_particles` | Green arrows | Particle cloud (add manually as PoseArray) |

To add the particle cloud in RViz: **Add → By Topic → `/pf_particles` → PoseArray**

---

## Experiments

### Q/R Variation — Process & Measurement Noise

Run with different noise parameters to compare filter behavior:

```bash
# High process noise (distrust model)
ros2 run prob_lab kf_node --ros-args -p process_noise_q:=0.5

# High measurement noise (distrust sensor)
ros2 run prob_lab ekf_node --ros-args -p measurement_noise_r:=1.0

# Default values
ros2 run prob_lab pf_node --ros-args -p process_noise_v:=0.3 -p process_noise_w:=0.2
```

### Sensor Dropout (Measurement Interruption)

Dropout is configured in `evaluation_node.cpp` and runs automatically. The dropout schedule relative to trajectory start:

| Interval | Duration |
|----------|----------|
| t = 70–71 s | 1 s dropout |
| t = 74–75 s | 1 s dropout |
| t = 78–79 s | 1 s dropout |
| t = 79–80 s | 1 s dropout (consecutive with above) |

During dropout: only **prediction** runs, **correction** is skipped. The filter drifts and then recovers when measurements resume.

### Landmark Detection

A fixed landmark at position `(0.6, 0.6)` in the odom frame is used. The `landmark.csv` tracks each filter's estimated distance to the landmark vs. the true distance.

---

## Results

After the trajectory completes, all results are saved automatically to `results/plots/`. You can also re-run the plot script manually:

```bash
python3 ~/ros2_ws/src/prob_lab/plot_results.py
```

### Generated Plots

| File | Content |
|------|---------|
| `01_trajectories.png` | GT vs KF vs EKF vs PF paths |
| `02_rmse.png` | Cumulative RMSE over time |
| `03_position_error.png` | X and Y error separately |
| `04_covariance.png` | Uncertainty (covariance) over time |
| `05_landmark.png` | Distance to landmark |
| `06_dropout.png` | Filter behavior during sensor dropout |
| `07_rmse_bar.png` | Final RMSE comparison bar chart |

---

## Filter Design Summary

### Kalman Filter (KF)
- **State:** `[px, py, vx, vy]` — constant velocity model
- **Prediction:** `x̄ = F·x + noise`, covariance grows by Q
- **Update:** Kalman gain weights prediction vs measurement
- **Key difference from EKF:** Linear motion model (A matrix fixed)

### Extended Kalman Filter (EKF)
- **State:** `[px, py, θ]`
- **Prediction:** Nonlinear `g(u, μ)` with Jacobian G at each step
- **Update:** Same structure as KF but linearized around current state
- **Key difference from KF:** Jacobian G replaces fixed A → more accurate in curves

### Particle Filter (PF)
- **Particles:** 500 × `[px, py, θ]`, initialized with Gaussian spread around start
- **Prediction:** Each particle propagated with noisy motion model
- **Update:** Log-likelihood weighting against odometry measurement
- **Resampling:** Systematic resampling when ESS < N/2
- **Key difference:** Non-parametric, handles multimodal distributions

---

## Troubleshooting

**Gazebo/RViz not starting:**
```bash
pkill -f gazebo && pkill -f gz && pkill -f ros2
sleep 3
source ~/ros2_ws/install/setup.bash
ros2 launch prob_lab prob_lab.launch.py
```

**Build errors:**
```bash
rm -rf ~/ros2_ws/build/prob_lab
cd ~/ros2_ws && colcon build --packages-select prob_lab
```

**CycloneDDS missing:**
```bash
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

---

## Tunable Parameters

| Parameter               | Default | Node    | Description                            |
|-------------------------|---------|---------|----------------------------------------|
| `process_noise_q`       | 0.01    | KF, EKF | Process noise (model uncertainty)      |
| `measurement_noise_r`   | 0.05    | KF, EKF | Measurement noise (sensor uncertainty) |
| `process_noise_v`       | 0.3     | PF      | Linear velocity noise                  |
| `process_noise_w`       | 0.2     | PF      | Angular velocity noise                 |
| `measurement_noise_r`   | 0.3     | PF      | Position measurement noise             |
| `measurement_noise_yaw` | 0.3     | PF      | Yaw measurement noise                  |
| `num_particles`         | 500     | PF      | Number of particles                    |
