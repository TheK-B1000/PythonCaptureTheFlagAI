import pygame
from queue import PriorityQueue
from copy import deepcopy  # For temp grid copies
import math

# Constants
ROWS = 20
COLS = 20
CELL_SIZE = 30
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Heuristic function (Manhattan distance)
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# Get valid neighbors
def get_neighbors(grid, row, col):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != 1:
            neighbors.append((nr, nc))
    return neighbors

# A* pathfinding
def a_star(grid, start, goal):
    if start == goal:
        return []
    open_set = PriorityQueue()
    count = 0
    open_set.put((h(start, goal), count, start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    while not open_set.empty():
        f, _, current = open_set.get()
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for neighbor in get_neighbors(grid, *current):
            temp_g = g_score.get(current, float('inf')) + 1
            if temp_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = temp_g
                f = temp_g + h(neighbor, goal)
                count += 1
                open_set.put((f, count, neighbor))
    return None

class Flag:
    def __init__(self, team, home_row, home_col):
        self.team = team
        self.home_pos = (home_row, home_col)
        self.pos = self.home_pos
        self.carried_by = None

    def get_pos(self):
        if self.carried_by:
            return self.carried_by.pos
        return self.pos

class Agent:
    def __init__(self, row, col, team):
        self.pos = (row, col)
        self.team = team
        self.carrying_flag = False
        self.path = []
        self.current_target_pos = None
        self.facing = 90 if team == 'blue' else 270  # degrees, 0: up, 90: right, 180: down, 270: left

    def update(self, grid, blue_flag, red_flag, all_agents):
        enemy_flag = red_flag if self.team == 'blue' else blue_flag
        home_pos = blue_flag.home_pos if self.team == 'blue' else red_flag.home_pos
        target = enemy_flag.get_pos() if not self.carrying_flag else home_pos

        # Identify enemies
        enemies = [a for a in all_agents if a.team != self.team]

        # Create temp grid with enemies (and adjacent cells) as obstacles for evasion
        temp_grid = deepcopy(grid)
        for enemy in enemies:
            er, ec = enemy.pos
            # Mark enemy pos and adjacent (Manhattan <=1) as obstacle
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = er + dr, ec + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        temp_grid[nr][nc] = 1

        # Always replan for dynamic evasion
        path = a_star(temp_grid, self.pos, target)
        if path:
            self.path = path
            self.current_target_pos = target
            # Move only one step per tick
            if self.path:
                next_pos = self.path[0]
                dr = next_pos[0] - self.pos[0]
                dc = next_pos[1] - self.pos[1]
                if dr == -1:
                    self.facing = 0
                elif dr == 1:
                    self.facing = 180
                elif dc == 1:
                    self.facing = 90
                elif dc == -1:
                    self.facing = 270
                self.path.pop(0)
                self.pos = next_pos

def draw(win, grid, blue_agents, red_agents, blue_flag, red_flag):
    win.fill(WHITE)
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == 1:
                pygame.draw.rect(win, BLACK, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # Draw base circle outlines
    blue_home_cx = blue_flag.home_pos[1] * CELL_SIZE + CELL_SIZE // 2
    blue_home_cy = blue_flag.home_pos[0] * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(win, BLUE, (blue_home_cx, blue_home_cy), CELL_SIZE // 2, width=2)
    red_home_cx = red_flag.home_pos[1] * CELL_SIZE + CELL_SIZE // 2
    red_home_cy = red_flag.home_pos[0] * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(win, RED, (red_home_cx, red_home_cy), CELL_SIZE // 2, width=2)
    # Draw flags if not carried
    if not blue_flag.carried_by:
        pygame.draw.rect(win, BLUE, (blue_flag.pos[1] * CELL_SIZE, blue_flag.pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    if not red_flag.carried_by:
        pygame.draw.rect(win, RED, (red_flag.pos[1] * CELL_SIZE, red_flag.pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # Draw agents as triangles
    all_agents = blue_agents + red_agents
    for agent in all_agents:
        color = BLUE if agent.team == 'blue' else RED
        cx = agent.pos[1] * CELL_SIZE + CELL_SIZE // 2
        cy = agent.pos[0] * CELL_SIZE + CELL_SIZE // 2
        r = CELL_SIZE // 3
        sqrt3_half = math.sqrt(3) / 2
        base_points = [
            (cx, cy - r),
            (cx - r * sqrt3_half, cy + r / 2),
            (cx + r * sqrt3_half, cy + r / 2)
        ]
        angle_deg = agent.facing
        angle_rad = math.radians(angle_deg)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        points = []
        for px, py in base_points:
            dx = px - cx
            dy = py - cy
            new_dx = dx * cos_theta - dy * sin_theta
            new_dy = dx * sin_theta + dy * cos_theta
            points.append((cx + new_dx, cy + new_dy))
        pygame.draw.polygon(win, color, points)
        if agent.carrying_flag:
            flag_color = RED if agent.team == 'blue' else BLUE
            pygame.draw.rect(win, flag_color, (cx - CELL_SIZE // 4, cy - CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
    # Draw grid lines
    for i in range(ROWS + 1):
        pygame.draw.line(win, GREY, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE))
    for j in range(COLS + 1):
        pygame.draw.line(win, GREY, (j * CELL_SIZE, 0), (j * CELL_SIZE, HEIGHT))
    pygame.display.update()

# Initialize Pygame
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Capture the Flag with Evasive A* Pathfinding")

# Grid (0: open, 1: wall)
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Add some walls (middle barrier with gaps)
for r in range(ROWS):
    if r % 5 != 0:
        grid[r][COLS // 2] = 1

# Flags
blue_flag = Flag('blue', ROWS // 2, 2)
red_flag = Flag('red', ROWS // 2, COLS - 3)

# Agents (2 per team)
blue_agents = [
    Agent(ROWS // 2 - 2, 1, 'blue'),
    Agent(ROWS // 2 + 2, 1, 'blue')
]
red_agents = [
    Agent(ROWS // 2 - 2, COLS - 1, 'red'),
    Agent(ROWS // 2 + 2, COLS - 1, 'red')
]

clock = pygame.time.Clock()
run = True
blue_score = 0
red_score = 0

while run:
    clock.tick(5)  # Slow down to see moves
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    all_agents = blue_agents + red_agents

    # Update agents (pass all_agents for enemy detection)
    for agent in all_agents:
        agent.update(grid, blue_flag, red_flag, all_agents)

    # Check for flag pickups
    for agent in all_agents:
        enemy_flag = red_flag if agent.team == 'blue' else blue_flag
        if enemy_flag.carried_by is None and agent.pos == enemy_flag.get_pos():
            enemy_flag.carried_by = agent
            enemy_flag.pos = agent.pos
            agent.carrying_flag = True

    # Check for scoring
    for agent in all_agents:
        if agent.carrying_flag:
            home_pos = blue_flag.home_pos if agent.team == 'blue' else red_flag.home_pos
            if agent.pos == home_pos:
                if agent.team == 'blue':
                    blue_score += 1
                else:
                    red_score += 1
                print(f"Scores: Blue {blue_score}, Red {red_score}")
                # Reset flag
                enemy_flag = red_flag if agent.team == 'blue' else blue_flag
                enemy_flag.carried_by = None
                enemy_flag.pos = enemy_flag.home_pos
                agent.carrying_flag = False

    # Check for tags
    for agent in all_agents:
        col = agent.pos[1]
        is_in_enemy_territory = (agent.team == 'blue' and col >= COLS // 2) or (agent.team == 'red' and col < COLS // 2)
        if is_in_enemy_territory:
            enemies = red_agents if agent.team == 'blue' else blue_agents
            for enemy in enemies:
                if abs(enemy.pos[0] - agent.pos[0]) + abs(enemy.pos[1] - agent.pos[1]) <= 1:
                    if agent.carrying_flag:
                        enemy_flag = red_flag if agent.team == 'blue' else blue_flag
                        enemy_flag.carried_by = None
                        enemy_flag.pos = agent.pos  # Drop at tag spot
                        agent.carrying_flag = False
                    # Respawn agent
                    home_col = 1 if agent.team == 'blue' else COLS - 2
                    agent.pos = (ROWS // 2, home_col)
                    agent.facing = 90 if agent.team == 'blue' else 270
                    agent.path = []
                    agent.current_target_pos = None
                    break

    draw(win, grid, blue_agents, red_agents, blue_flag, red_flag)

pygame.quit()