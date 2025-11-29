# run_game.py
import cv2
import mediapipe as mp
import numpy as np
from maze_generator import generate_grid_maze, maze_to_image
from game_core import FingerMazeGame

# ---------------- PARAMETERS ---------------- 
CELL_W = 31        # Smaller maze
CELL_H = 21        # Smaller maze
CELL_SIZE = 8      # pixels per maze cell
SPEED = 12.0       # pixels per frame (smooth movement through paths)
DOT_RADIUS = 3     # smaller dot radius
MAX_DISPLAY_HEIGHT = 600  # Maximum height for display window
FREE_MOVEMENT = False  # Must follow maze paths, get stuck at walls
TARGET_PROXIMITY = 80  # Only move when yellow dot is within this distance (pixels)

def create_new_game():
    """Create a new random maze game"""
    import random
    # Ensure different random seed each time
    random.seed()
    maze_grid = generate_grid_maze(CELL_W, CELL_H)             # grid with 1=path
    maze_img = maze_to_image(maze_grid, cell_size=CELL_SIZE)  # grayscale 0/255
    return FingerMazeGame(maze_img, dot_radius=DOT_RADIUS)

# ---------------- CREATE MAZE ----------------
game = create_new_game()

# Game state tracking
import time
start_time = time.time()
game_won = False

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# optional EMA smoothing
class EMAFilter:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.state = None
    def update(self, value):
        value = np.array(value, dtype=float)
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return tuple(self.state)

ema = EMAFilter(alpha=0.6)

# Create resizable window before starting the loop
cv2.namedWindow("Finger Maze - [Camera | Maze]", cv2.WINDOW_NORMAL)

with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=1,
                    static_image_mode=False) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # mirror camera for natural movement
        frame = cv2.flip(frame, 1)
        h_cam, w_cam = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # --- inside the main loop ---
        hand_detected = False
        game.current_target = None  # Reset target
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            lm8 = hand_landmarks.landmark[8]  # index fingertip

            # normalized camera coordinates (0-1, with origin at top-left)
            # MediaPipe returns coordinates relative to the flipped frame
            # Since frame is already flipped, coordinates match user's view directly
            nx, ny = lm8.x, lm8.y
            # No need to flip X - frame is already flipped, so coordinates are correct

            # map to maze pixel coordinates
            maze_h, maze_w = game.maze.shape
            mx = int(np.clip(nx * maze_w, 0, maze_w - 1))
            my = int(np.clip(ny * maze_h, 0, maze_h - 1))

            # Apply EMA smoothing to reduce jitter
            mx, my = ema.update((mx, my))
            mx, my = int(mx), int(my)

            # Ensure target is on a valid path (snap to nearest path if on wall)
            # This helps guide the player through the maze
            if game.maze[my, mx] != 255:  # If target is on a wall
                # Find nearest path position
                target_px = game.find_nearest_path((mx, my))
                mx, my = target_px
            
            # Store target for rendering
            game.current_target = (mx, my)
            
            # Only move player if yellow dot (target) is close enough (beside or behind)
            target_pos = np.array([mx, my])
            player_pos = game.player
            distance_to_target = np.linalg.norm(target_pos - player_pos)
            
            if distance_to_target <= TARGET_PROXIMITY:
                # Yellow dot is close - move player towards it
                game.move_player_towards((mx, my), speed=SPEED, free_movement=FREE_MOVEMENT)
            # If target is too far, player doesn't move

            # Draw landmarks on the flipped frame (more subtle styling)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2))
            # Optional: Remove coordinate text for cleaner display
            # cv2.putText(frame, f"Target: ({mx},{my})", (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # cv2.putText(frame, f"Player: ({int(game.player[0])},{int(game.player[1])})", (10, 90), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Show hand detection status (beautiful styling)
        if hand_detected:
            status_text = "READY"
            status_color = (0, 255, 0)
            bg_color = (0, 200, 0)
        else:
            status_text = "Show Hand"
            status_color = (0, 0, 255)
            bg_color = (0, 0, 150)
        
        # Draw background rectangle for status
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (5, 5), (text_size[0] + 15, text_size[1] + 20), bg_color, -1)
        cv2.rectangle(frame, (5, 5), (text_size[0] + 15, text_size[1] + 20), status_color, 2)
        cv2.putText(frame, status_text, (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


        # Calculate scale to fit maze on screen - ensure FULL maze is visible
        maze_h, maze_w = game.maze.shape
        
        # Get screen dimensions (use reasonable defaults if not available)
        # Target: fit both camera and maze side-by-side on screen
        screen_height = 700  # Target screen height
        screen_width = 1600  # Target screen width
        
        # Calculate available space for maze (leave room for camera on left + separator)
        # Camera will take about 40% of width, separator ~1%, maze gets ~59%
        camera_portion = 0.4
        separator_portion = 0.01
        maze_portion = 0.59
        available_maze_width = int(screen_width * maze_portion)
        available_maze_height = screen_height
        
        # Calculate scale to fit maze in available space
        scale_w = available_maze_width / maze_w
        scale_h = available_maze_height / maze_h
        # Use the smaller scale to ensure entire maze fits
        display_scale = min(scale_w, scale_h)
        
        # render maze with calculated scale
        vis = game.render(scale=display_scale)
        
        # Resize camera to match maze height (maintain aspect ratio)
        target_height = vis.shape[0]
        cam_aspect = w_cam / h_cam
        target_cam_width = int(target_height * cam_aspect)
        cam_small = cv2.resize(frame, (target_cam_width, target_height))
        
        # Draw game info on camera frame (right side of camera)
        elapsed_time = time.time() - start_time
        time_text = f"Time: {elapsed_time:.1f}s"
        
        # Calculate distance to goal
        goal_dist = np.linalg.norm(game.player - np.array(game.goal))
        dist_text = f"Distance: {int(goal_dist)}px"
        
        # Draw info box on camera (only if not won yet)
        if not game_won:
            info_y = 25
            info_x = cam_small.shape[1] - 210  # Right side of camera
            cv2.rectangle(cam_small, (info_x, 5), (info_x + 200, 60), (20, 20, 40), -1)
            cv2.rectangle(cam_small, (info_x, 5), (info_x + 200, 60), (100, 100, 150), 2)
            cv2.putText(cam_small, time_text, (info_x + 10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(cam_small, dist_text, (info_x + 10, info_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        # Check win and draw win message on vis
        if game.check_win() and not game_won:
            game_won = True
            elapsed_time = time.time() - start_time
            
            # Beautiful win celebration
            win_text = "YOU WIN!"
            time_text = f"Time: {elapsed_time:.1f}s"
            
            # Calculate text positions (on scaled maze)
            text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (vis.shape[1] - text_size[0]) // 2
            text_y = vis.shape[0] // 2
            
            # Win message background
            box_height = 100
            box_width = max(text_size[0], time_size[0]) + 40
            box_x = (vis.shape[1] - box_width) // 2
            box_y = text_y - 60
            
            cv2.rectangle(vis, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         (0, 100, 0), -1)
            cv2.rectangle(vis, (box_x, box_y), 
                         (box_x + box_width, box_y + box_height), 
                         (0, 255, 0), 3)
            
            # Win text
            cv2.putText(vis, win_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            time_x = (vis.shape[1] - time_size[0]) // 2
            cv2.putText(vis, time_text, (time_x, text_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        
        # Add a subtle border/separator between camera and maze
        separator_width = 3
        separator = np.zeros((vis.shape[0], separator_width, 3), dtype=np.uint8)
        separator[:] = [30, 30, 50]  # Dark blue-gray separator
        
        # Combine camera, separator, and maze side by side
        combined = np.hstack([cam_small, separator, vis])
        
        # Final check: if combined image is still too large, scale it down
        # This ensures the entire window fits on screen
        if combined.shape[0] > screen_height or combined.shape[1] > screen_width:
            scale_h = screen_height / combined.shape[0]
            scale_w = screen_width / combined.shape[1]
            final_scale = min(scale_h, scale_w)
            new_h = int(combined.shape[0] * final_scale)
            new_w = int(combined.shape[1] * final_scale)
            combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Display the combined image
        cv2.imshow("Finger Maze - [Camera | Maze]", combined)
        
        # Set window size to ensure full visibility (user can still resize)
        cv2.resizeWindow("Finger Maze - [Camera | Maze]", combined.shape[1], combined.shape[0])
        
        # If won, show play again/exit menu
        if game_won:
            # Draw menu on the combined image
            menu_text1 = "Press 'R' to Play Again"
            menu_text2 = "Press 'Q' to Exit"
            
            # Calculate menu position (center of screen)
            menu_y = combined.shape[0] // 2 + 80
            menu_x = combined.shape[1] // 2
            
            # Menu background
            text1_size = cv2.getTextSize(menu_text1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text2_size = cv2.getTextSize(menu_text2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            menu_width = max(text1_size[0], text2_size[0]) + 40
            menu_height = 100
            menu_box_x = menu_x - menu_width // 2
            menu_box_y = menu_y - 50
            
            cv2.rectangle(combined, (menu_box_x, menu_box_y), 
                         (menu_box_x + menu_width, menu_box_y + menu_height), 
                         (40, 40, 60), -1)
            cv2.rectangle(combined, (menu_box_x, menu_box_y), 
                         (menu_box_x + menu_width, menu_box_y + menu_height), 
                         (100, 100, 150), 2)
            
            # Menu text
            text1_x = menu_x - text1_size[0] // 2
            text2_x = menu_x - text2_size[0] // 2
            cv2.putText(combined, menu_text1, (text1_x, menu_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, menu_text2, (text2_x, menu_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
            
            cv2.imshow("Finger Maze - [Camera | Maze]", combined)
            
            # Wait for user input
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r') or key == ord('R'):
                    # Play again - create new game
                    game = create_new_game()
                    start_time = time.time()
                    game_won = False
                    ema = EMAFilter(alpha=0.6)  # Reset filter
                    break
                elif key == ord('q') or key == ord('Q'):
                    # Exit
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
                elif key == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)

        # quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
