# game_core.py
import cv2
import numpy as np

WALL_VAL = 0
PATH_VAL = 255

class FingerMazeGame:
    def __init__(self, maze_img, start_px=None, goal_px=None, dot_radius=8):
        # maze_img: grayscale binary image (0 wall, 255 path)
        self.maze = maze_img
        self.h, self.w = self.maze.shape
        # set start and goal if not provided (find top-left path and bottom-right path)
        if start_px is None:
            start_px = self.find_nearest_path((10,10))
        if goal_px is None:
            goal_px = self.find_nearest_path((self.w-20, self.h-20))
        self.start = start_px
        self.goal = goal_px
        self.player = np.array(self.start, dtype=float)
        self.dot_radius = dot_radius
        self.current_target = None  # For debug visualization
        self.player_trail = []  # Trail of player positions

    def find_nearest_path(self, px):
        x0, y0 = px
        x0 = int(np.clip(x0,0,self.w-1)); y0 = int(np.clip(y0,0,self.h-1))
        if self.maze[y0,x0] == PATH_VAL:
            return (x0,y0)
        # search radius
        for r in range(1, max(self.w, self.h)):
            for dx in range(-r, r+1):
                for dy in (-r, r+1):
                    x = x0 + dx; y = y0 + dy
                    if 0<=x<self.w and 0<=y<self.h and self.maze[y,x] == PATH_VAL:
                        return (x,y)
        return (x0,y0)

    def map_normalized_to_maze(self, nx, ny, cam_w, cam_h):
        # nx,ny are normalized 0..1; camera aspect vs maze aspect handled by simple scaling
        mx = int(nx * self.w)
        my = int(ny * self.h)
        mx = np.clip(mx, 0, self.w-1); my = np.clip(my, 0, self.h-1)
        return mx, my

    def move_player_towards(self, target_px, speed=8.0, free_movement=False):
        # target_px: (x, y) in maze pixel coords
        target = np.array(target_px, dtype=float)
        dir_vec = target - self.player
        dist = np.linalg.norm(dir_vec)
        
        if free_movement:
            # Free movement - no collision detection, move directly to target
            if dist < 0.5:
                self.player = target
            else:
                # Normalize direction and calculate step
                step = (dir_vec / dist) * min(speed, dist)
                self.player = self.player + step
                # Keep player within bounds
                self.player[0] = np.clip(self.player[0], 0, self.w - 1)
                self.player[1] = np.clip(self.player[1], 0, self.h - 1)
            return
        
        # MAZE MODE: Player must follow paths, gets stuck at walls
        if dist < 0.5:
            # If very close, only move if target is on a valid path
            if self.is_position_free(target):
                self.player = target
            # Otherwise stay stuck
            return
        
        # Calculate step towards target
        step = (dir_vec / dist) * min(speed, dist)
        new_pos = self.player + step
        
        # Check if new position is on a path (not a wall)
        if self.is_position_free(new_pos):
            # Can move - update position
            old_pos = self.player.copy()
            self.player = new_pos
            # Add to trail if moved significantly
            if np.linalg.norm(self.player - old_pos) > 2.0:
                self.player_trail.append(self.player.copy())
                # Keep trail limited
                if len(self.player_trail) > 30:
                    self.player_trail.pop(0)
        else:
            # Hit a wall - try to slide along the wall
            # Try moving only horizontally
            step_x = np.array([step[0], 0])
            attempt_x = self.player + step_x
            if self.is_position_free(attempt_x):
                self.player = attempt_x
            else:
                # Try moving only vertically
                step_y = np.array([0, step[1]])
                attempt_y = self.player + step_y
                if self.is_position_free(attempt_y):
                    self.player = attempt_y
                # If both fail, player is stuck (don't move)

    def is_position_free(self, pos):
        return self.is_position_free_with_radius(pos, self.dot_radius)
    
    def is_position_free_with_radius(self, pos, radius):
        x, y = int(pos[0]), int(pos[1])
        r = int(max(1, radius))  # Ensure at least radius 1
        # make sure indices do not go out of bounds
        x0 = max(0, x-r); x1 = min(self.w-1, x+r)
        y0 = max(0, y-r); y1 = min(self.h-1, y+r)
        if y0 > y1 or x0 > x1:
            return False
        
        # Check center point first (most important - must be on path)
        if not (0 <= x < self.w and 0 <= y < self.h):
            return False
        if self.maze[y, x] != PATH_VAL:
            return False  # Center must be on path
        
        # Check surrounding area - need most of the area to be path
        patch = self.maze[y0:y1+1, x0:x1+1]
        # Require at least 80% path for proper maze navigation (stricter)
        path_ratio = np.sum(patch == PATH_VAL) / patch.size
        return path_ratio >= 0.8

    def check_win(self, threshold_px=15):
        g = np.array(self.goal)
        p = self.player
        dist = np.linalg.norm(g - p)
        
        # First check: must be close enough
        if dist > threshold_px:
            return False
        
        # Second check: player must actually be on the goal position (or very close)
        # Check if player position overlaps with goal position
        goal_x, goal_y = int(g[0]), int(g[1])
        player_x, player_y = int(p[0]), int(p[1])
        
        # Check if player is at the goal position or adjacent to it
        dx = abs(player_x - goal_x)
        dy = abs(player_y - goal_y)
        
        # Player must be within goal radius + player radius
        goal_radius = self.dot_radius + 4
        player_radius = self.dot_radius
        max_dist = goal_radius + player_radius
        
        if dx > max_dist or dy > max_dist:
            return False
        
        # Third check: verify there's actually a path (not blocked by wall)
        # Check if the midpoint between player and goal is on a path
        mid_x = int((player_x + goal_x) / 2)
        mid_y = int((player_y + goal_y) / 2)
        
        # If midpoint is on a wall, there's a wall between them
        if 0 <= mid_x < self.w and 0 <= mid_y < self.h:
            if self.maze[mid_y, mid_x] != PATH_VAL:
                return False  # Wall between player and goal
        
        # All checks passed - player has reached the goal!
        return True

    def render(self, scale=1.0):
        # Create beautiful colored maze
        vis = cv2.cvtColor(self.maze, cv2.COLOR_GRAY2BGR)
        
        # Make walls darker blue and paths lighter (more beautiful)
        walls = self.maze == WALL_VAL
        paths = self.maze == PATH_VAL
        vis[walls] = [40, 40, 60]  # Dark blue-gray walls
        vis[paths] = [240, 240, 250]  # Light gray-white paths
        
        # Draw on original size first
        gx, gy = int(self.goal[0]), int(self.goal[1])
        px, py = int(self.player[0]), int(self.player[1])
        
        # Draw goal (beautiful green with glow effect)
        goal_radius = max(8, self.dot_radius + 4)
        # Outer glow
        cv2.circle(vis, (gx, gy), goal_radius + 2, (0, 200, 0), -1)
        # Main circle
        cv2.circle(vis, (gx, gy), goal_radius, (0, 255, 100), -1)
        # Inner highlight
        cv2.circle(vis, (gx, gy), goal_radius - 2, (150, 255, 150), -1)
        # Outline
        cv2.circle(vis, (gx, gy), goal_radius, (0, 150, 0), 2)
        
        # Draw target (beautiful yellow/orange circle with glow) if available
        if self.current_target is not None:
            tx, ty = int(self.current_target[0]), int(self.current_target[1])
            # Outer glow
            cv2.circle(vis, (tx, ty), 6, (0, 200, 255), -1)
            # Main circle
            cv2.circle(vis, (tx, ty), 4, (0, 255, 255), -1)
            # Inner highlight
            cv2.circle(vis, (tx, ty), 2, (255, 255, 200), -1)
        
        # Draw player trail (fading effect)
        for i, trail_pos in enumerate(self.player_trail):
            trail_x, trail_y = int(trail_pos[0]), int(trail_pos[1])
            # Fade effect - older positions are more transparent (smaller and darker)
            alpha = i / max(len(self.player_trail), 1)
            trail_radius = max(2, int((self.dot_radius - 1) * (1 - alpha * 0.7)))
            trail_color = (int(100 * alpha), int(50 * alpha), int(200 * alpha))
            cv2.circle(vis, (trail_x, trail_y), trail_radius, trail_color, -1)
        
        # Draw player (beautiful red with glow effect)
        player_radius = max(6, self.dot_radius + 2)
        # Outer glow
        cv2.circle(vis, (px, py), player_radius + 2, (0, 0, 200), -1)
        # Main circle
        cv2.circle(vis, (px, py), player_radius, (0, 0, 255), -1)
        # Inner highlight
        cv2.circle(vis, (px, py), player_radius - 2, (150, 150, 255), -1)
        # Outline
        cv2.circle(vis, (px, py), player_radius, (0, 0, 150), 2)
        
        # Add border around maze for polish
        cv2.rectangle(vis, (0, 0), (self.w - 1, self.h - 1), (20, 20, 40), 2)
        
        # Scale the entire image after drawing
        if scale != 1.0:
            vis = cv2.resize(vis, (int(self.w*scale), int(self.h*scale)), interpolation=cv2.INTER_NEAREST)
        
        return vis
