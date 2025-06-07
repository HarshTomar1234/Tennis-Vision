#!/usr/bin/env python3
"""
Script to download sample cricket videos for testing the Cricket Vision system.
This script downloads free cricket videos from various sources.
"""

import os
import requests
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL to local filename"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {str(e)}")
        return False

def download_cricket_videos():
    """Download sample cricket videos for testing"""
    
    # Create input_videos directory if it doesn't exist
    os.makedirs("input_videos", exist_ok=True)
    
    # Sample cricket video URLs (these are example URLs - replace with actual working URLs)
    cricket_videos = [
        {
            "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "filename": "input_videos/cricket_sample_1.mp4",
            "description": "Sample cricket video 1"
        },
        {
            "url": "https://file-examples.com/storage/fe68c1b7c66d5b2b9c9b8b8/2017/10/file_example_MP4_1280_10MG.mp4", 
            "filename": "input_videos/cricket_sample_2.mp4",
            "description": "Sample cricket video 2"
        }
    ]
    
    print("🏏 Cricket Vision - Video Downloader")
    print("=" * 50)
    
    successful_downloads = 0
    
    for video in cricket_videos:
        print(f"\n📥 {video['description']}")
        if download_file(video["url"], video["filename"]):
            successful_downloads += 1
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {successful_downloads}")
    print(f"❌ Failed: {len(cricket_videos) - successful_downloads}")
    
    if successful_downloads > 0:
        print(f"\n🎉 Ready to analyze cricket videos!")
        print(f"Run: python cricket_main.py")
    else:
        print(f"\n⚠️  No videos downloaded successfully.")
        print(f"Please manually add cricket videos to the input_videos/ folder.")
        print(f"\nRecommended sources:")
        print(f"• Pexels: https://www.pexels.com/search/videos/cricket/")
        print(f"• Pixabay: https://pixabay.com/videos/search/cricket/")
        print(f"• YouTube (with proper attribution)")

def get_cricket_video_sources():
    """Print information about where to get cricket videos"""
    
    print("🏏 Cricket Video Sources")
    print("=" * 50)
    
    sources = [
        {
            "name": "Pexels (Free)",
            "url": "https://www.pexels.com/search/videos/cricket/",
            "description": "High-quality, royalty-free cricket videos",
            "license": "Free for commercial use"
        },
        {
            "name": "Pixabay (Free)", 
            "url": "https://pixabay.com/videos/search/cricket/",
            "description": "Free cricket video clips",
            "license": "Free for commercial use"
        },
        {
            "name": "Unsplash (Some video)",
            "url": "https://unsplash.com/s/photos/cricket",
            "description": "Some cricket video content",
            "license": "Free for commercial use"
        },
        {
            "name": "YouTube",
            "url": "https://youtube.com",
            "description": "Cricket highlights and matches",
            "license": "Check individual video licenses"
        },
        {
            "name": "Getty Images (Paid)",
            "url": "https://www.gettyimages.com",
            "description": "Professional cricket footage",
            "license": "Paid licensing"
        },
        {
            "name": "Shutterstock (Paid)",
            "url": "https://www.shutterstock.com",
            "description": "High-quality cricket videos",
            "license": "Paid licensing"
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   URL: {source['url']}")
        print(f"   Description: {source['description']}")
        print(f"   License: {source['license']}")
    
    print(f"\n💡 Tips for downloading cricket videos:")
    print(f"• Look for videos with clear view of the pitch")
    print(f"• Higher resolution (1080p+) works better")
    print(f"• Stable camera angles are preferred")
    print(f"• Good lighting conditions improve detection")
    print(f"• Videos showing batting and bowling work best")
    
    print(f"\n📁 Save downloaded videos to: input_videos/")
    print(f"📝 Rename your main video to: input_videos/cricket_match.mp4")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sources":
        get_cricket_video_sources()
    else:
        download_cricket_videos()
        print(f"\n💡 For more video sources, run:")
        print(f"python download_cricket_videos.py --sources")