#!/usr/bin/env python3
"""
Main runner script for Cricket Vision Analysis.
This script provides an easy way to run cricket analysis with different options.
"""

import os
import sys
import argparse
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    
    print("🔍 Checking requirements...")
    
    # Check if input video exists
    input_videos = list(Path("input_videos").glob("*.mp4"))
    if not input_videos:
        print("❌ No cricket videos found in input_videos/")
        print("💡 Run: python download_cricket_videos.py")
        return False
    
    # Check if models exist
    required_models = [
        "models/cricket_ball_model.pt",
        "models/cricket_pitch_keypoints.pth"
    ]
    
    missing_models = []
    for model in required_models:
        if not os.path.exists(model) or os.path.getsize(model) == 0:
            missing_models.append(model)
    
    if missing_models:
        print(f"⚠️  Missing or empty models: {missing_models}")
        print("💡 Run: python setup_cricket_models.py")
        print("📚 You'll need to train or obtain these models")
    
    # Check Python packages
    try:
        import cv2
        import torch
        import ultralytics
        import pandas as pd
        import numpy as np
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ Requirements check complete")
    return True

def run_cricket_analysis(video_path=None, output_path=None, enable_features=None):
    """Run cricket analysis with specified parameters"""
    
    print("🏏 Starting Cricket Vision Analysis...")
    print("=" * 50)
    
    # Set default paths
    if video_path is None:
        # Find first video in input_videos
        input_videos = list(Path("input_videos").glob("*.mp4"))
        if input_videos:
            video_path = str(input_videos[0])
        else:
            print("❌ No video found. Please add a cricket video to input_videos/")
            return False
    
    if output_path is None:
        output_path = "output_videos/cricket_analysis.avi"
    
    print(f"📹 Input video: {video_path}")
    print(f"💾 Output video: {output_path}")
    
    # Import and run cricket analysis
    try:
        # Modify the cricket_main.py to accept parameters
        import cricket_main
        
        # Update paths in cricket_main if needed
        cricket_main.input_video_path = video_path
        cricket_main.output_video_path = output_path
        
        # Run analysis
        cricket_main.main()
        
        print("🎉 Cricket analysis completed successfully!")
        print(f"📺 View results: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def list_available_videos():
    """List available cricket videos"""
    
    print("📹 Available Cricket Videos:")
    print("=" * 30)
    
    input_videos = list(Path("input_videos").glob("*.mp4"))
    
    if not input_videos:
        print("❌ No cricket videos found in input_videos/")
        print("💡 Add cricket videos or run: python download_cricket_videos.py")
        return
    
    for i, video in enumerate(input_videos, 1):
        file_size = video.stat().st_size / (1024 * 1024)  # MB
        print(f"{i}. {video.name} ({file_size:.1f} MB)")
    
    print(f"\n💡 To analyze a specific video:")
    print(f"python run_cricket_analysis.py --video input_videos/your_video.mp4")

def show_analysis_options():
    """Show available analysis options"""
    
    print("🔧 Cricket Analysis Options:")
    print("=" * 30)
    
    options = [
        ("Shot Classification", "Classify cricket shots (drive, cut, pull, etc.)"),
        ("Ball Speed Analysis", "Calculate ball delivery speed"),
        ("Player Role Detection", "Identify batsman, bowler, fielders"),
        ("Mini Pitch Visualization", "Bird's eye view of the match"),
        ("Real-time Statistics", "Live batting and bowling stats"),
        ("Shot Direction Analysis", "Off-side, leg-side, straight shots"),
        ("Boundary Detection", "Identify fours and sixes"),
        ("Event Detection", "Detect deliveries, shots, boundaries")
    ]
    
    for i, (feature, description) in enumerate(options, 1):
        print(f"{i}. {feature}")
        print(f"   {description}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Cricket Vision Analysis System")
    parser.add_argument("--video", "-v", help="Path to cricket video file")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--list", "-l", action="store_true", help="List available videos")
    parser.add_argument("--check", "-c", action="store_true", help="Check requirements")
    parser.add_argument("--options", action="store_true", help="Show analysis options")
    parser.add_argument("--setup", "-s", action="store_true", help="Run setup")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_videos()
    elif args.check:
        check_requirements()
    elif args.options:
        show_analysis_options()
    elif args.setup:
        print("🔧 Running setup...")
        os.system("python setup_cricket_models.py")
        os.system("python download_cricket_videos.py --sources")
    else:
        # Run analysis
        if not check_requirements():
            print("\n💡 Fix requirements first, then run analysis again")
            return
        
        success = run_cricket_analysis(
            video_path=args.video,
            output_path=args.output
        )
        
        if success:
            print("\n🎯 Analysis Tips:")
            print("• Check output_videos/ for the analyzed video")
            print("• Look for shot classifications and statistics")
            print("• Mini pitch shows player positions and ball trajectory")
            print("• Try different cricket videos for varied results")

if __name__ == "__main__":
    main()