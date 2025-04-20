# Tennis Detection and Analysis System

<div align="center">
  <img src="frame_images/tennis_analysis_quarter_frame53.png" width="800" alt="Tennis Analysis System">
  <p><em>Computer Vision-based Tennis Match Analysis</em></p>
</div>

## Overview

This project implements a comprehensive computer vision system for tennis match analysis. It detects players and the ball, tracks their movements, analyzes shots, and provides real-time statistics. The system uses state-of-the-art computer vision techniques to extract valuable insights from tennis match videos.

## Quickstart

```bash
# Clone the repository
git clone https://github.com/HarshTomar1234/Tennis-Vision.git
cd Tennis-Vision

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py

# View output video in output_videos/output_video.avi
```

## Features

- **Player Detection and Tracking**: Accurately identifies and tracks players throughout the match
- **Ball Detection and Trajectory Analysis**: Follows the ball's path and identifies moments when shots are made
- **Court Line Detection**: Identifies the tennis court lines for spatial reference
- **Shot Classification**: Categorizes shots as serve, forehand, backhand, volley, or smash
- **Mini Court Visualization**: Provides a bird's-eye view of player and ball positions
- **Statistical Analysis**: Real-time statistics on player movement and shot speed
- **Enhanced Visual Interface**: Clearly displays all analysis with intuitive visual elements

## Directory Structure

```
Tennis-Vision/
├── analysis/               # Analysis utilities and algorithms
├── constants/              # Project constants and configuration
├── court_line_detector/    # Court line detection module
├── frame_images/           # Extracted video frames for analysis
├── input_videos/           # Input tennis match videos
├── mini_visual_court/      # Mini court visualization module
├── models/                 # Trained ML models
│   ├── keypoints_model.pth # Court keypoint detection model
│   └── last.pt             # Ball detection model
├── output_videos/          # Processed videos with analysis 
├── runs/                   # Training runs and logs
├── trackers/               # Object tracking modules
│   ├── ball_tracker.py     # Ball tracking implementation
│   └── player_tracker.py   # Player tracking implementation
├── tracker_stubs/          # Serialized tracking data for development
├── training/               # Training scripts and utilities
├── utils/                  # Utility functions
│   ├── bbox_utils.py       # Bounding box utilities
│   ├── conversions.py      # Unit conversion utilities
│   ├── drawing_utils.py    # Visualization utilities
│   ├── player_stats_drawer_utils.py # Player statistics visualization
│   ├── shot_classifier.py  # Shot classification implementation
│   └── video_utils.py      # Video handling utilities
├── main.py                 # Main application entry point
├── requirements.txt        # Project dependencies
└── yolov8x.pt              # YOLOv8 model for player detection
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/HarshTomar1234/Tennis-Vision.git
   cd Tennis-Vision
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the required models:
   - YOLOv8x model for player detection
   - Court keypoint detection model
   - Ball detection model

## Usage

### Basic Usage

Run the main script with a tennis video:

```
python main.py
```

By default, the script will:
- Process the video at `input_videos/input_video.mp4`
- Generate an output video with analysis at `output_videos/output_video.avi`

### Customization

Edit the `main.py` file to customize:
- Input video path
- Detection thresholds
- Visual styling
- Analysis parameters

## Example Input/Output

### Input Video

The demo uses tennis match footage from professional tournaments. The project includes a sample input video:
- Location: `input_videos/input_video.mp4` (12MB)
- Content: Professional tennis match with clear court visibility and player movements
- Duration: ~9 seconds at 24 FPS (214 frames)

### Output Visualization

The system produces a video with comprehensive visual analysis:

#### Output Video
- Location: `output_videos/output_video.avi` (6.5MB)
- Alternative format: `output_videos/output_video.mp4` (6.4MB)
- Resolution: Matches input video
- Content: Enhanced visualization with player tracking, shot classification, and statistics

#### Visual Breakdown

![Beginning of Match](frame_images/tennis_analysis_beginning_frame0.png)
*Initial state of the analysis at the start of the match*

![Mid-Match Analysis](frame_images/tennis_analysis_middle_frame107.png)
*Analysis during an active rally showing player positions, ball trajectory, and shot classification*

![Shot Analysis](frame_images/tennis_analysis_sixty_percent_frame128.png)
*Detailed shot analysis with player statistics and shot classification*

#### Key Visual Elements

1. **Player Stats Board**: Located at the center bottom, displays player speeds and shot information
2. **Shot Analysis Panel**: Located on the left side, shows recent shots with color-coded indicators
3. **Shot Type Legend**: Located at the bottom right, explains the shot type abbreviations and colors
4. **Player Tracking**: Bounding boxes track players with real-time position data
5. **Ball Tracking**: Highlights the ball position and trajectory
6. **Mini Court View**: Top-right corner visualization showing bird's-eye view of the match

## Technical Details

### Player Detection

The system uses YOLOv8, a state-of-the-art object detection model, to identify and track players on the court. The player tracking pipeline includes:

1. Initial detection using YOLOv8
2. Player identification based on court position
3. Frame-to-frame tracking with position prediction

### Ball Detection and Tracking

Ball detection utilizes a specialized model trained on tennis footage. The tracking algorithm:

1. Applies the detection model to identify ball candidates
2. Filters detections based on size, shape, and motion
3. Interpolates positions to handle occlusions and missed detections
4. Identifies shot moments based on trajectory changes

### Court Line Detection

The court detection module identifies the tennis court structure using:

1. A keypoint-based approach to identify court corners and lines
2. Perspective transformation to map court coordinates
3. Robust line fitting to handle partial occlusions

### Shot Classification

The shot classifier analyzes:

1. Player position relative to the court
2. Ball trajectory before and after contact
3. Temporal context of the rally
4. Player orientation and movement

Based on these factors, shots are classified as:

- **Serve**: First shot of a rally
- **Forehand**: Standard shot with racket on dominant side
- **Backhand**: Shot with racket across body
- **Volley**: Shot near the net without bounce
- **Smash**: Overhead shot with downward trajectory

## Future Enhancements

The system can be extended with:

- **Player Pose Estimation**: Analyze player technique and form
- **Tactical Pattern Recognition**: Identify recurring strategies and patterns
- **Match Statistics Aggregation**: Compile comprehensive match statistics
- **Multi-Camera Support**: Synchronize and analyze footage from multiple cameras
- **Real-Time Processing**: Optimize for live analysis during matches
- **Player Identification**: Automatically identify specific players

## Requirements

The project requires the following dependencies, listed in `requirements.txt`:

- OpenCV for image processing
- PyTorch for neural network models
- NumPy for numerical operations
- Pandas for data analysis
- YOLOv8 for object detection

## Credits

This project builds upon research and implementations in computer vision and sports analysis domains:

- YOLOv8 for object detection
- OpenCV for image processing
- PyTorch for deep learning components

## License

This project is licensed under the MIT License - see the !LICENSE[LICENSE] file for details.
