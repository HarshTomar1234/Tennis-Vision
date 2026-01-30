import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats, layout_params=None):
    """
    Draw player statistics overlay on video frames.
    
    CAMERA-ROBUST: Now supports dynamic positioning via layout_params.
    
    Args:
        output_video_frames: List of video frames to draw on
        player_stats: DataFrame with player statistics
        layout_params: Optional dict from UILayoutManager.get_stats_panel_params()
                      If None, uses dynamic calculation based on frame size
    """
    # Check if shot classification data is available
    has_shot_classification = 'player_1_shot_type' in player_stats.columns or 'player_2_shot_type' in player_stats.columns

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        # Get shot types if available
        player_1_shot_type = row.get('player_1_shot_type', 'N/A')
        player_2_shot_type = row.get('player_2_shot_type', 'N/A')

        frame = output_video_frames[index]
        frame_height, frame_width = frame.shape[:2]
        
        # CAMERA-ROBUST: Dynamic positioning based on layout_params or frame size
        if layout_params is not None:
            # Use provided layout parameters
            start_x = layout_params['start_x']
            start_y = layout_params['start_y']
            width = layout_params['width']
            height = layout_params['height']
        else:
            # Dynamic calculation based on frame dimensions
            # Width 450px to accommodate Player 2 column with km/h text
            width = max(450, int(frame_width * 0.42))  # Minimum 450px
            height = 250 if has_shot_classification else 200
            
            # Position at the bottom center, with dynamic Y based on frame height
            start_x = (frame_width - width) // 2  # Center horizontally
            
            # Position stats panel in lower portion of frame (last 40%)
            start_y = int(frame_height * 0.65)
            
            # Ensure it fits within frame
            if start_y + height > frame_height - 10:
                start_y = frame_height - height - 10
        
        end_x = start_x + width
        end_y = start_y + height

        # Draw a more visible and clearer background with better contrast
        overlay = frame.copy()
        # Background panel with darker color
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        # More opacity for better readability
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add header with title
        cv2.rectangle(frame, (start_x, start_y), (end_x, start_y + 40), (40, 40, 100), -1)
        cv2.putText(frame, "PLAYER STATS", (start_x + 110, start_y + 27), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add column headers - dynamic positioning based on width
        col1_x = start_x + 15           # Metric column
        col2_x = start_x + int(width * 0.35)   # Player 1 column (~35% of width)
        col3_x = start_x + int(width * 0.65)   # Player 2 column (~65% of width)
        
        cv2.putText(frame, "Metric", (col1_x, start_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Player 1", (col2_x, start_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Player 2", (col3_x, start_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Add horizontal divider
        cv2.line(frame, (start_x, start_y + 75), (end_x, start_y + 75), (150, 150, 150), 1)
        
        # Shot Speed row
        y_pos = start_y + 100
        cv2.putText(frame, "Shot Speed", (col1_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_1_shot_speed:.1f} km/h", (col2_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_2_shot_speed:.1f} km/h", (col3_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Player Speed row
        y_pos = start_y + 130
        cv2.putText(frame, "Player Speed", (col1_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_1_speed:.1f} km/h", (col2_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_2_speed:.1f} km/h", (col3_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Avg Shot Speed row
        y_pos = start_y + 160
        cv2.putText(frame, "Avg. S. Speed", (col1_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_1_shot_speed:.1f} km/h", (col2_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_2_shot_speed:.1f} km/h", (col3_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Avg Player Speed row
        y_pos = start_y + 190
        cv2.putText(frame, "Avg. P. Speed", (col1_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_1_speed:.1f} km/h", (col2_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_2_speed:.1f} km/h", (col3_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add shot type information if shot classification is enabled
        if has_shot_classification:
            y_pos = start_y + 220
            cv2.putText(frame, "Last Shot Type", (col1_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{player_1_shot_type}", (col2_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{player_2_shot_type}", (col3_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        output_video_frames[index] = frame
    
    return output_video_frames