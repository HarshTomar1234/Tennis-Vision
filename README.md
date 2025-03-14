# Tennis Detection and Visualization System

## Overview

This project implements a comprehensive tennis match analysis system using computer vision techniques. It detects players and the ball in tennis match videos, tracks their movements, and visualizes them on both the actual court and a simplified mini-court representation for easier analysis.

![Tennis Detection System](output_videos/output_preview.jpg)

## Key Features

### Player Detection and Tracking
- Accurate detection of players using YOLOv8x
- Robust tracking with position validation and smoothing
- Fallback mechanisms for handling missing or invalid detections
- Clear visual representation with consistent green circles on the mini-court

### Ball Detection and Tracking
- Precise ball detection using a custom-trained YOLO model
- Advanced interpolation for smooth ball trajectory visualization
- Shot detection for identifying key moments in the match
- Distinct purple visualization on the mini-court for clear differentiation from players

### Court Line Detection
- Automatic detection of court lines and keypoints
- Accurate mapping between the actual court and mini-court coordinates
- Professional styling with clean lines and high-contrast visuals

### Mini-Court Visualization
- Real-time synchronization between actual court and mini-court
- Clear visual distinction between players (green) and ball (purple)
- Improved coordinate transformation for accurate positioning
- Enhanced error handling for robust visualization

## Project Structure

```
Tennis Detection CV project/
├── court_line_detector/       # Court line detection module
├── mini_visual_court/         # Mini-court visualization module
├── trackers/                  # Player and ball tracking modules
├── models/                    # Pre-trained models
├── input_videos/              # Input tennis match videos
├── output_videos/             # Processed output videos
├── utils/                     # Utility functions
├── main.py                    # Main application entry point
└── README.md                  # Project documentation
```

## Technical Implementation

### Player Tracking
- Uses YOLOv8x for robust player detection
- Implements history-based smoothing for consistent tracking
- Validates player positions against court boundaries
- Creates reasonable defaults for player positions when detections fail

### Ball Tracking
- Custom-trained YOLO model optimized for tennis ball detection
- Advanced interpolation techniques for handling occlusions and fast movements
- Proximity-based possession determination (150-pixel threshold)
- Micro-positioning with small random offsets for realistic visualization

### Coordinate Transformation
- Normalized relative positioning system for accurate court-to-mini-court mapping
- Court boundary constraints to ensure positions stay within valid areas
- Distance-based smoothing for natural movement visualization

### Visualization Enhancements
- Consistent player representation with identical green circles and black outlines
- Distinct ball visualization with purple color and black outline
- Clean mini-court design without distracting elements
- Frame information overlay with shot detection indicators

## Usage

1. Place your tennis match video in the `input_videos/` directory
2. Run the main application:

```bash
python main.py
```

3. The processed video will be saved to the `output_videos/` directory

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- Pandas
- PyTorch
- Ultralytics YOLO

## Future Improvements

- Add player identification and statistics tracking
- Implement shot classification (forehand, backhand, serve, etc.)
- Create heatmaps for player movement patterns
- Add rally counting and point scoring detection
- Develop a user interface for interactive analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
