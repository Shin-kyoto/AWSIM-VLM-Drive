# VLM Trajectory Selector and Planner

This package contains two main components for VLM-based autonomous driving:

1. **VLM Trajectory Selector**: Selects from candidate trajectories using VLM
2. **VLM Planner**: Generates complete trajectories using VLM

## Setup

### Environment Setup

```sh
sudo apt-get update && sudo apt-get install -y wmctrl x11-utils gnome-screenshot
```

```sh
uv venv -p python3.10
```

```sh
source .venv/bin/activate
```

```sh
uv pip install .
```

### Set Gemini API Key

```sh
export GEMINI_API_KEY="YOUR_API_KEY"
```

## Components

### 1. VLM Trajectory Selector

Selects the best trajectory from a set of candidate trajectories based on camera images and VLM inference.

**Files:**
- `trajectory_selector.py` - Main ROS 2 node
- `vlm_selector.py` - VLM inference logic

**Run:**
```sh
# Run the trajectory selector node with custom topics
python -m trajectory_selector --ros-args -p input_topic:="/vad/output/trajectories_base" -p output_topic:="/planning/vad/auto/trajectory"
```

**Topics:**
- Subscribe: 
  - `/input/trajectory` (CandidateTrajectories)
  - `/sensing/camera/image_raw` (Image)
- Publish: 
  - `/output/trajectory` (Trajectory)

### 2. VLM Planner

Generates complete trajectories from scratch using VLM inference based on camera images and vehicle state.

**Files:**
- `vlm_planner_node.py` - Main ROS 2 node
- `vlm_planner.py` - VLM trajectory generation logic

**Run:**
```sh
# Run the VLM planner node with custom output topic
python vlm_planner_node.py --ros-args -p output_topic:="/planning/vad/auto/trajectory"
```

**Topics:**
- Subscribe:
  - `/sensing/camera/image_raw` (Image)
  - `/localization/kinematic_state` (Odometry)
  - `/localization/acceleration` (AccelWithCovarianceStamped)
- Publish:
  - `/output/trajectory` (Trajectory)

**Parameters:**
- `output_topic`: Output trajectory topic (default: `/output/trajectory`)
- Inference interval: 2.0 seconds (configurable in code)

## Features

### VLM Trajectory Selector
- Real-time trajectory selection from candidates
- Image-based decision making using Gemini AI
- Track sector awareness for racing scenarios
- Rate-limited inference to balance performance and accuracy

### VLM Planner
- Complete trajectory generation from camera input
- Vehicle state awareness (velocity, acceleration)
- 5-second lookahead trajectory planning
- Smooth trajectory point generation with proper timing
- Race track sector detection and navigation

## Track Knowledge

Both components include knowledge of a 13-sector race track:

1. Starting Straight
2. R-Hairpin (with white sign landmark)
3. Short Straight
4. L-Hairpin
5. Short Straight
6. U-shaped Right
7. 90-degree Left
8. R-Hairpin
9. L-Hairpin
10. S-Curve (L->R)
11. 90-degree Right
12. S-Curve (L->R)
13. Sweeping Right Corner

## Architecture

```
Camera Image → VLM (Gemini) → Trajectory Decision/Generation → ROS 2 Message → Autoware
```

The VLM components use Google's Gemini 2.5 Flash Lite model for fast inference while maintaining accuracy for autonomous driving decisions.

## Development

### VLM Selector Architecture
- `VLMSelector` class handles Gemini API calls and image preprocessing
- `VlmTrajectorySelectorNode` handles ROS 2 communication and message conversion

### VLM Planner Architecture  
- `VLMPlanner` class handles trajectory generation using Gemini
- `VlmPlannerNode` handles ROS 2 communication and vehicle state management

Both components are designed to be modular and can be used independently depending on your autonomous driving pipeline requirements.


## Acknowledgment

The implementation of this code was greatly inspired by the following repository. Many thanks for their excellent work:
https://github.com/soyaoki/AWSIM-VLM-Drive
