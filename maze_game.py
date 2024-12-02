import random
import pygame
import numpy as np

pygame.init()

# Colors (not used anymore since we're using images)
WHITE = (255, 255, 255)
GREY = (20, 20, 20)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GOLD = (255, 215, 0)

# Adjusted size and cell width for smaller maze
size = (401, 401)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Maze with Q-Learning")

done = False
clock = pygame.time.Clock()

width = 50  # Increased width for smaller grid
cols = int(size[0] / width)
rows = int(size[1] / width)

stack = []

# Q-Learning parameters
actions = ['up', 'down', 'left', 'right']
q_table_player = np.zeros((rows, cols, len(actions)))  # Player Q-values
q_table_police = np.zeros((rows, cols, len(actions)))  # Police Q-values
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Cell class for maze generation
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.walls = [True, True, True, True]  # Top, Right, Bottom, Left

    def draw(self):
        if self.visited:
            pygame.draw.rect(screen, WHITE, (self.x * width, self.y * width, width, width))
        if self.walls[0]:  # Top
            pygame.draw.line(screen, BLACK, (self.x * width, self.y * width), ((self.x + 1) * width, self.y * width), 2)
        if self.walls[1]:  # Right
            pygame.draw.line(screen, BLACK, ((self.x + 1) * width, self.y * width), ((self.x + 1) * width, (self.y + 1) * width), 2)
        if self.walls[2]:  # Bottom
            pygame.draw.line(screen, BLACK, ((self.x + 1) * width, (self.y + 1) * width), (self.x * width, (self.y + 1) * width), 2)
        if self.walls[3]:  # Left
            pygame.draw.line(screen, BLACK, (self.x * width, (self.y + 1) * width), (self.x * width, self.y * width), 2)

    def check_neighbors(self):
        neighbors = []
        if self.y > 0 and not grid[self.y - 1][self.x].visited:  # Top
            neighbors.append(grid[self.y - 1][self.x])
        if self.x < cols - 1 and not grid[self.y][self.x + 1].visited:  # Right
            neighbors.append(grid[self.y][self.x + 1])
        if self.y < rows - 1 and not grid[self.y + 1][self.x].visited:  # Bottom
            neighbors.append(grid[self.y + 1][self.x])
        if self.x > 0 and not grid[self.y][self.x - 1].visited:  # Left
            neighbors.append(grid[self.y][self.x - 1])
        return random.choice(neighbors) if neighbors else None

def remove_walls(current, next):
    dx = current.x - next.x
    dy = current.y - next.y
    if dx == 1:  # Next is to the left
        current.walls[3] = False
        next.walls[1] = False
    elif dx == -1:  # Next is to the right
        current.walls[1] = False
        next.walls[3] = False
    if dy == 1:  # Next is above
        current.walls[0] = False
        next.walls[2] = False
    elif dy == -1:  # Next is below
        current.walls[2] = False
        next.walls[0] = False

# Generate the maze
grid = [[Cell(x, y) for x in range(cols)] for y in range(rows)]
current = grid[0][0]
stack.append(current)

while stack:
    current.visited = True
    next_cell = current.check_neighbors()
    if next_cell:
        next_cell.visited = True
        stack.append(current)
        remove_walls(current, next_cell)
        current = next_cell
    elif stack:
        current = stack.pop()

# Load images
player_img = pygame.image.load('player.png')
police_img = pygame.image.load('police.png')
treasure_img = pygame.image.load('treasure.png')

# Resize images to fit the grid cell
player_img = pygame.transform.scale(player_img, (width - 10, width - 10))
police_img = pygame.transform.scale(police_img, (width - 10, width - 10))
treasure_img = pygame.transform.scale(treasure_img, (width - 10, width - 10))

# Treasure class
class Treasure:
    def __init__(self):
        self.x = cols - 1
        self.y = random.randint(0, rows - 1)

    def draw(self):
        screen.blit(treasure_img, (self.x * width + 5, self.y * width + 5))



# Player class
class Player:
    def __init__(self):
        self.x = 0
        self.y = 0

    def draw(self):
        screen.blit(player_img, (self.x * width + 5, self.y * width + 5))

    def move(self, action):
        if action == 'up' and not grid[self.y][self.x].walls[0]:
            self.y -= 1
        elif action == 'down' and not grid[self.y][self.x].walls[2]:
            self.y += 1
        elif action == 'left' and not grid[self.y][self.x].walls[3]:
            self.x -= 1
        elif action == 'right' and not grid[self.y][self.x].walls[1]:
            self.x += 1

# Police class
class Police:
    def __init__(self):
        self.x = cols - 1
        self.y = rows - 1

    def draw(self):
        screen.blit(police_img, (self.x * width + 5, self.y * width + 5))

    def move(self, action):
        if action == 'up' and not grid[self.y][self.x].walls[0]:
            self.y -= 1
        elif action == 'down' and not grid[self.y][self.x].walls[2]:
            self.y += 1
        elif action == 'left' and not grid[self.y][self.x].walls[3]:
            self.x -= 1
        elif action == 'right' and not grid[self.y][self.x].walls[1]:
            self.x += 1

# Q-learning step for the player
def q_learning_step_player(player, treasure, police):
    state = (player.y, player.x)
    
    # Exploration or exploitation
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)  # Explore
    else:
        action = actions[np.argmax(q_table_player[state])]  # Exploit
    
    # Take the action
    player.move(action)
    new_state = (player.y, player.x)

    # Calculate reward
    if new_state == (treasure.y, treasure.x):  # Reached the treasure
        reward = 20
        player.x, player.y = 0, 0  
    elif new_state == (police.y, police.x):  # Caught by police
        reward = -20
        player.x, player.y = 0, 0  # Restart player position
    else:
        # Distance calculation
        distance_before = abs(player.x - police.x) + abs(player.y - police.y)
        distance_after = abs(new_state[1] - police.x) + abs(new_state[0] - police.y)
        if distance_after > distance_before:
            reward = 5  # Moving further away from police
        elif distance_after < distance_before:
            reward = -5  # Moving closer to police
        else:
            reward = -1  # Neutral movement cost
    
    # Update Q-value
    best_future_q = np.max(q_table_player[new_state])  # Best future Q-value
    q_table_player[state][actions.index(action)] = (1 - alpha) * q_table_player[state][actions.index(action)] + alpha * (reward + gamma * best_future_q)

# Q-learning step for the police
def q_learning_step_police(police, player):
    state = (police.y, police.x)
    
    # Exploration or exploitation
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)  # Explore
    else:
        action = actions[np.argmax(q_table_police[state])]  # Exploit
    
    # Take the action
    police.move(action)
    new_state = (police.y, police.x)

    # Calculate reward
    if new_state == (player.y, player.x):  # Caught the player
        reward = 20
    else:
        reward = -1  # Neutral cost for moving
    
    # Update Q-value
    best_future_q = np.max(q_table_police[new_state])  # Best future Q-value
    q_table_police[state][actions.index(action)] = (1 - alpha) * q_table_police[state][actions.index(action)] + alpha * (reward + gamma * best_future_q)

# Main game loop
player = Player()
police = Police()
treasure = Treasure()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Q-learning steps for both player and police
    q_learning_step_player(player, treasure, police)
    q_learning_step_police(police, player)
    
    # Redraw the game
    screen.fill(GREY)
    for row in grid:
        for cell in row:
            cell.draw()
    
    player.draw()
    police.draw()
    treasure.draw()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
