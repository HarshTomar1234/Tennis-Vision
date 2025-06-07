# Cricket Vision Analysis System

<div align="center">
  <img src="https://images.pexels.com/photos/1661950/pexels-photo-1661950.jpeg?auto=compress&cs=tinysrgb&w=800" width="800" alt="Cricket Analysis System">
  <p><em>AI-Powered Cricket Match Analysis with Computer Vision</em></p>
</div>

## 🏏 Overview

This project implements a comprehensive computer vision system for cricket match analysis, adapted from the Tennis Vision project. It detects players, tracks the ball, analyzes shots, and provides real-time cricket statistics with a beautiful mini pitch visualization.

## ✨ Features

### 🎯 Core Detection & Tracking
- **Player Detection & Role Identification**: Automatically identifies batsman, bowler, wicket-keeper, and fielders
- **Cricket Ball Tracking**: Advanced ball detection optimized for cricket's unique ball characteristics
- **Pitch Boundary Detection**: Identifies cricket pitch lines, creases, and boundaries
- **Real-time Event Detection**: Detects deliveries, shots, boundaries, and other cricket events

### 📊 Advanced Analytics
- **Cricket Shot Classification**: Categorizes shots as:
  - Defensive shots
  - Drives (straight, cover, on-drive)
  - Cuts and pulls
  - Sweeps and hooks
  - Flicks and lofts
  - Boundaries (fours and sixes)
- **Ball Speed Analysis**: Calculates delivery speed and shot power
- **Strike Rate Calculation**: Real-time batting statistics
- **Shot Direction Analysis**: Off-side, leg-side, straight, behind wicket

### 🎨 Visualization
- **Mini Cricket Pitch**: Bird's-eye view with:
  - Realistic cricket field colors (green field, brown pitch)
  - Boundary circle visualization
  - Player position tracking with role-specific colors
  - Ball trajectory visualization
- **Shot Analysis Board**: Real-time shot statistics and recent shot history
- **Cricket Statistics Panel**: Live match statistics including runs, balls faced, strike rate
- **Shot Type Legend**: Color-coded shot classification guide

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd cricket-vision

# Install dependencies
pip install -r requirements.txt

# Add your cricket video to input_videos/
# Download sample cricket videos from the sources below

# Run the cricket analysis
python cricket_main.py

# View output in output_videos/cricket_analysis.avi
```

## 📁 Project Structure

```
Cricket-Vision/
├── cricket_main.py                    # Main cricket analysis script
├── cricket_constants/                 # Cricket-specific constants
│   └── __init__.py                   # Field dimensions, shot types, etc.
├── cricket_trackers/                  # Cricket player and ball tracking
│   ├── cricket_player_tracker.py     # Player detection with role identification
│   └── cricket_ball_tracker.py       # Cricket ball tracking and event detection
├── cricket_pitch_detector/            # Cricket pitch detection
│   └── cricket_pitch_detector.py     # Pitch keypoint detection
├── mini_cricket_pitch/                # Mini pitch visualization
│   └── mini_cricket_pitch.py         # Cricket field visualization
├── utils/                            # Utility functions
│   └── cricket_shot_classifier.py    # Cricket shot classification
├── models/                           # AI models (you need to train/obtain these)
│   ├── cricket_ball_model.pt         # Cricket ball detection model
│   └── cricket_pitch_keypoints.pth   # Pitch detection model
├── input_videos/                     # Input cricket videos
├── output_videos/                    # Processed analysis videos
└── tracker_stubs/                    # Cached detection data
```

## 🎥 Where to Get Cricket Videos

### Free Sources
1. **YouTube** (with proper attribution):
   - Search for "cricket highlights"
   - Use youtube-dl or similar tools to download
   - Ensure compliance with copyright laws

2. **Pexels** (Free stock videos):
   - [Cricket Videos on Pexels](https://www.pexels.com/search/videos/cricket/)
   - High-quality, royalty-free cricket footage

3. **Pixabay** (Free videos):
   - [Cricket Videos on Pixabay](https://pixabay.com/videos/search/cricket/)
   - Free for commercial use

4. **Unsplash** (Some video content):
   - [Cricket content on Unsplash](https://unsplash.com/s/photos/cricket)

### Sample Video URLs (for testing):
```bash
# Download sample cricket videos
wget "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4" -O input_videos/cricket_sample.mp4
```

### Professional Sources (Paid)
1. **Getty Images Sports**
2. **Shutterstock Sports**
3. **Adobe Stock Sports**

## 🏏 Cricket-Specific Features

### Shot Classification System
The system recognizes these cricket shots:

| Shot Type | Description | Color Code |
|-----------|-------------|------------|
| **Six** | Ball crosses boundary without bouncing | Red |
| **Four** | Ball reaches boundary after bouncing | Green |
| **Drive** | Straight or off-side attacking shot | Green |
| **Cut** | Shot played square on off-side | Blue |
| **Pull** | Cross-batted shot to leg-side | Orange |
| **Sweep** | Shot played to leg-side off spin | Cyan |
| **Hook** | Shot played to leg-side off pace | Purple |
| **Flick** | Wrist shot to leg-side | Yellow |
| **Loft** | Aerial shot over fielders | Magenta |
| **Defensive** | Defensive stroke | Gray |
| **Single** | Shot resulting in one run | White |
| **Dot Ball** | No run scored | Dark Gray |

### Player Role Detection
- **Batsman** (Green): Primary striker
- **Bowler** (Blue): Ball delivery
- **Wicket Keeper** (Red): Behind stumps
- **Fielders** (Cyan): Field positions

### Cricket Statistics
- **Runs Scored**: Total runs in the analyzed period
- **Balls Faced**: Number of deliveries
- **Strike Rate**: (Runs/Balls) × 100
- **Ball Speed**: Delivery speed in km/h
- **Shot Power**: Relative power of shots (0-100%)

## 🛠️ Technical Implementation

### Cricket Ball Detection
Unlike tennis balls, cricket balls are:
- Smaller and harder to track
- Red/white colored (different from tennis yellow)
- Move faster with more unpredictable trajectories
- Often obscured by players

Our cricket ball tracker includes:
- Enhanced interpolation for fast-moving objects
- Cricket-specific event detection (deliveries, shots, boundaries)
- Improved confidence thresholds for small objects

### Cricket Pitch Detection
Cricket pitches have unique characteristics:
- 22-yard pitch length
- Creases and wickets
- Circular/oval boundary
- Different surface colors (grass field, dirt pitch)

The pitch detector identifies:
- Pitch boundaries (22-yard strip)
- Creases (batting and bowling)
- Wicket positions
- Boundary circle

### Mini Cricket Pitch Visualization
The mini pitch provides:
- Realistic cricket field appearance (green grass, brown pitch)
- Proper cricket field proportions
- Boundary circle representation
- Player positions with role-specific colors
- Ball trajectory with cricket-specific styling

## 🎯 Advanced Features

### Shot Direction Analysis
The system determines shot direction:
- **Straight**: Down the ground
- **Off Side**: Right side (for right-handed batsman)
- **Leg Side**: Left side (for right-handed batsman)  
- **Behind Wicket**: Behind the batsman

### Event Detection
Automatically detects:
- **Deliveries**: When bowler releases ball
- **Shots**: When batsman hits ball
- **Boundaries**: When ball reaches boundary
- **Wickets**: Potential dismissals (basic detection)

### Real-time Statistics
- Live updating scoreboard
- Shot-by-shot analysis
- Performance metrics
- Visual shot history (last 6 balls)

## 🔧 Customization

### Adjusting Cricket Parameters
Edit `cricket_constants/__init__.py`:
```python
# Modify field dimensions
BOUNDARY_RADIUS_MIN = 55  # Smaller grounds
BOUNDARY_RADIUS_MAX = 90  # Larger grounds

# Adjust shot classification thresholds
BOUNDARY_DISTANCE_THRESHOLD = 200  # Boundary shot distance
AGGRESSIVE_SHOT_THRESHOLD = 150    # Aggressive shot distance
```

### Custom Shot Types
Add new shot types in `utils/cricket_shot_classifier.py`:
```python
self.SHOT_TYPES = {
    'REVERSE_SWEEP': 'Reverse Sweep',
    'SWITCH_HIT': 'Switch Hit',
    # Add more shots...
}
```

### Visual Styling
Modify colors and styling in `mini_cricket_pitch.py`:
```python
# Change field colors
green_bg[:, :] = [34, 139, 34]  # Field green
pitch_color = (101, 67, 33)     # Pitch brown
```

## 📈 Performance Optimization

### For Better Accuracy
1. **Use high-resolution videos** (1080p or higher)
2. **Ensure good lighting** in cricket footage
3. **Stable camera angles** work best
4. **Clear view of pitch** improves detection

### For Faster Processing
1. **Reduce video resolution** for testing
2. **Use GPU acceleration** if available
3. **Adjust detection confidence** thresholds
4. **Enable stub caching** for development

## 🚀 Deployment Options

### Local Deployment
```bash
# Run locally
python cricket_main.py
```

### Web Deployment
The system can be adapted for web deployment using:
- **Streamlit** for quick web interface
- **Flask/Django** for full web application
- **Docker** for containerized deployment

### Cloud Deployment
Deploy on:
- **AWS EC2** with GPU instances
- **Google Cloud Platform** with AI/ML services
- **Azure** with computer vision services

## 🤝 Contributing

We welcome contributions! Areas for improvement:

1. **Enhanced Shot Classification**
   - More shot types (reverse sweep, switch hit, etc.)
   - Better accuracy for complex shots
   - Spin vs pace bowling detection

2. **Advanced Analytics**
   - Wagon wheel visualization
   - Heat maps for shot placement
   - Bowling analysis (line, length, swing)

3. **Multi-Camera Support**
   - Combine multiple camera angles
   - 3D ball trajectory reconstruction
   - Better player identification

4. **Real-time Processing**
   - Live match analysis
   - Streaming integration
   - Mobile app development

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the excellent Tennis Vision project
- YOLOv8 for object detection
- OpenCV for computer vision
- Cricket community for domain knowledge

## 📞 Support

For questions or support:
1. Open an issue on GitHub
2. Check the documentation
3. Review the tennis vision project for additional insights

---

**Ready to revolutionize cricket analysis? Let's play! 🏏**