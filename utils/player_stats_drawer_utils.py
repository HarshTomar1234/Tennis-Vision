import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats):
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
        
        # Adjust height if we need to display shot types
        width = 350
        height = 250 if has_shot_classification else 200
        
        # Position at the bottom center (Vienna logo area)
        start_x = frame.shape[1]//2 - width//2  # Center horizontally
        start_y = 450  # Position at Vienna text area
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
        
        # Add column headers
        cv2.putText(frame, "Metric", (start_x + 15, start_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Player 1", (start_x + 150, start_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Player 2", (start_x + 250, start_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Add horizontal divider
        cv2.line(frame, (start_x, start_y + 75), (end_x, start_y + 75), (150, 150, 150), 1)
        
        # Shot Speed row
        y_pos = start_y + 100
        cv2.putText(frame, "Shot Speed", (start_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_1_shot_speed:.1f} km/h", (start_x + 150, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_2_shot_speed:.1f} km/h", (start_x + 250, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Player Speed row
        y_pos = start_y + 130
        cv2.putText(frame, "Player Speed", (start_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_1_speed:.1f} km/h", (start_x + 150, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{player_2_speed:.1f} km/h", (start_x + 250, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Avg Shot Speed row
        y_pos = start_y + 160
        cv2.putText(frame, "Avg. S. Speed", (start_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_1_shot_speed:.1f} km/h", (start_x + 150, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_2_shot_speed:.1f} km/h", (start_x + 250, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Avg Player Speed row
        y_pos = start_y + 190
        cv2.putText(frame, "Avg. P. Speed", (start_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_1_speed:.1f} km/h", (start_x + 150, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{avg_player_2_speed:.1f} km/h", (start_x + 250, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add shot type information if shot classification is enabled
        if has_shot_classification:
            y_pos = start_y + 220
            cv2.putText(frame, "Last Shot Type", (start_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{player_1_shot_type}", (start_x + 150, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{player_2_shot_type}", (start_x + 250, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        output_video_frames[index] = frame
    
    return output_video_frames