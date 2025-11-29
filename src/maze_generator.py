# maze_generator.py
import numpy as np
import random
import cv2
import time

def generate_grid_maze(width_cells=31, height_cells=21):
    # Use current time as seed for true randomness each game
    random.seed(int(time.time() * 1000) % (2**32))
    # width_cells and height_cells should be odd numbers (cells + walls)
    w = width_cells
    h = height_cells
    maze = np.zeros((h, w), dtype=np.uint8)  # 0 wall, 1 path
    # init cells as walls
    maze[1::2, 1::2] = 1  # mark cells
    # carve passages using DFS
    def neighbours(cx, cy):
        dirs = [(2,0),(-2,0),(0,2),(0,-2)]
        random.shuffle(dirs)
        for dx,dy in dirs:
            nx, ny = cx+dx, cy+dy
            if 0 < nx < w and 0 < ny < h and maze[ny, nx] == 1:
                yield nx, ny, dx//2, dy//2
    visited = set()
    stack = [(1,1)]
    visited.add((1,1))
    while stack:
        cx, cy = stack[-1]
        found = False
        for nx, ny, wx, wy in neighbours(cx, cy):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                # knock down wall between
                maze[cy+wy, cx+wx] = 1
                stack.append((nx, ny))
                found = True
                break
        if not found:
            stack.pop()
    return maze  # 1 = path, 0 = wall

def maze_to_image(maze, cell_size=20, wall_color=0, path_color=255):
    h, w = maze.shape
    img_h = h * cell_size
    img_w = w * cell_size
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            color = path_color if maze[y,x]==1 else wall_color
            y0, x0 = y*cell_size, x*cell_size
            img[y0:y0+cell_size, x0:x0+cell_size] = color
    return img

if __name__ == "__main__":
    m = generate_grid_maze(61,41)
    img = maze_to_image(m, cell_size=8)
    cv2.imwrite("maze.png", img)
    cv2.imshow("maze", img); cv2.waitKey(0); cv2.destroyAllWindows()
