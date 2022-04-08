import pygame
import random
from enum import Enum, auto

pygame.init()
# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# GLOBALS VARS
s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height


# SHAPE FORMATS

S = [['.....',
      '......',
      '..00..',
      '.00...',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

class Piece:
      y = 20
      x = 10

      def __init__(self, x, y, shape):
            self.x = x
            self.y = y
            self.shape = shape
            self.color = shape_colors[shapes.index(shape)]
            self.rotation = 0

class Curses(Enum):
      NO_CURSE = auto()
      SQUIGGLES = auto()
      FAST = auto()
      MISDIRECTION = auto()
      LIES = auto()

      def __str__(self):
            return self



#Global vars 2: Electric Boogaloo
shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# index 0 - 6 represent shape

sounds = ['Bacon', 'Explosion', 'Oof', 'Tada', 'Quack', 'Ding', 'Damn', 'Boom', 'Milk', 'Huwawa', 'Gotcha', 'Avocado']

ACTUAL_CURSES = (Curses.SQUIGGLES, Curses.FAST, Curses.MISDIRECTION, Curses.LIES)

CURSE_POINT = 1500

TERMINAL_SPEED = 0.12

FAST_SPEED = 0.08

QUICKFALL_SPEED = 10^-9

CURSE = Curses.NO_CURSE

def create_grid(locked_positions={}):
      grid = [[(0,0,0) for x in range(10)] for x in range(20)]

      for i in range(len(grid)):
            for j in range(len(grid[i])):
                  if (j, i) in locked_positions:
                        c = locked_positions[(j, i)]
                        grid[i][j] = c

      return grid

def convert_shape_format(shape):
	#turn the shape into actual useful information
      positions = []
      format = shape.shape[shape.rotation % len(shape.shape)] #gives sublist we need, % because shape_rotation will increase indefinitely (past amount of sublists)

      for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                  if column == '0': #0's are the money
                        positions.append((shape.x + j, shape.y + i)) #the 0 has to be at whatever x position you were on the grid + offset from center of shape

      for i, pos in enumerate(positions):
            positions[i] = (pos[0] - 2, pos[1] - 4) #get rid of that STUPID offset caused by those STUPID dots around our beautiful 0's
            #(centre of shape is 2 rows from top and 4 rows from left??? ) 

      return positions

def valid_space(shape, locked_positions):
      locked_pos = list(locked_positions.keys())

      formatted = convert_shape_format(shape)

      for pos in formatted:
            if pos in locked_pos:
                  return False
            if pos[0] < 0 or pos[0] >= 10 or pos[1] >= 20:
                  return False

      return True

def max_score():
      with open('./Hiscores.txt', 'r') as f:
            lines = f.readlines()
            score = lines[0].strip()  #no \n

      return score

def check_lost(positions):
	#check if the blocks have reached the top of the screen aka you lost boi
      for pos in positions:
            x, y = pos
            if y < 1:
                  return True
      return False

def get_shape(CURSE = Curses.NO_CURSE):
      #creates a random piece at the top middle of the screen
      if CURSE == Curses.SQUIGGLES:
            return Piece(5, 0, random.choice(shapes[0:2]))
      else:
	      return Piece(5, 0, random.choice(shapes)) 


def draw_text_middle(text, size, color, surface):
      font = pygame.font.SysFont('papyrus', size, bold = True)
      label = font.render(text, 1, color)

      surface.blit(label, (top_left_x + play_width/2 - label.get_width()/2, top_left_y + play_height/2 - label.get_height()/2))


def draw_grid(surface, grid):
      #draw lines of grid
      sx = top_left_x #shorten notation
      sy = top_left_y

      for i in range(len(grid)):
            pygame.draw.line(surface, (128, 128, 128), (sx, sy + i*block_size), (sx + play_width, sy + i*block_size)) 
            #draw horizontal line: surface, color, starting point line, end point line
            for j in range(len(grid[i])):
                  pygame.draw.line(surface, (128, 128, 128), (sx + j*block_size, sy), (sx + j*block_size, sy + play_height)) 


def clear_rows(grid, locked):
      inc = 0
      for i in range(len(grid)-1, -1, -1):
            row = grid[i]
            if (0, 0, 0) not in row:      # => row completely filled
                  inc += 1 #inc = amount of rows removed
                  ind = i
                  for j in range(len(row)):
                        try:
                              del locked[(j, i)]
                        except:
                              continue

      if inc > 0:
            for key in sorted(list(locked), key = lambda x: x[1])[::-1]:     #sort list based on y-value and move through it backwards (bottom to top), or else blocks moving down could overwrite existing blocks
                  x, y = key
                  if y < ind: #=> if y-value of current key is above row being removed (since only those blocks have to be moved down)
                        newKey = (x, y + inc) #move block down
                        locked[newKey] = locked.pop(key)

      return inc
      
def draw_next_shape(shape, surface, CURSE = Curses.NO_CURSE):
      font = pygame.font.SysFont('papyrus', 30)
      label = font.render('Next shape', 1, (255, 255, 255))

      sx = top_left_x + play_width + block_size * 2   #block_size * 2 is a random offset from the right of the grid, you can play around with this
      sy = top_left_y + play_height/2 - block_size * 3
                  
      format = shape.shape[shape.rotation % len(shape.shape)]

      for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                  if column == '0': #0 is one of the blocks so draw those
                        pygame.draw.rect(surface, shape.color, (sx + j*block_size, sy + i*block_size, block_size, block_size), 0)

      surface.blit(label, (sx + block_size/2, sy - block_size))


def draw_window(surface, grid, score = 0, CURSE = Curses.NO_CURSE):
      surface.fill((0, 0, 0))
      
      pygame.font.init()
      font = pygame.font.SysFont('papyrus', 60)
      label = font.render('TURBO TETRIS', 1, (255, 255, 255))

      surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 30))
      #label, coordinates of middle of screen

      #draw the blocks
      for i in range(len(grid)):
            for j in range(len(grid[i])):
                  pygame.draw.rect(surface, grid[i][j], (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0) #surface drawn on, colors drawn, position of the block we're drawing, width, height, fill
      
      #draw player
      pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 4)

      hiscore = max_score()
      font = pygame.font.SysFont('papyrus', 30)
      label_score = font.render('Score: ' + str(score), 1, (255, 255, 255))
      label_hiscore = font.render('High Score: ' + str(hiscore), 1, (255, 255, 255))
      
      sx = top_left_x - block_size * 8  #block_size * 8 is a random offset from the right of the grid, you can play around with this
      sy = top_left_y + play_height/2 - block_size * 9

      surface.blit(label_score, (sx, sy))
      surface.blit(label_hiscore, (sx, sy - block_size))

      if str(CURSE.name) != 'NO_CURSE':
            label_curse = font.render('Current curse: ', 1, (255, 255, 255))
            label_actual_curse = font.render(str(CURSE.name), 1, (255, 0, 0))
            surface.blit(label_curse, (sx, sy + block_size))
            surface.blit(label_actual_curse, (sx, sy + 2*block_size))

      draw_grid(surface, grid)

def initialize_game():
      clock = pygame.time.Clock()
      fall_time = 0
      level_time = 0

      parameters = {"run":True, "locked_positions":{}, "grid":create_grid(), "change_piece":False, "quickfall":False, \
            "current_piece":get_shape(), "next_piece":get_shape(), "fall_speed":0.27, \
            "current_speed":0.27, "score":0, "cursescore":0, "score_increased":False, "curse_turn":0, "curse":Curses.NO_CURSE}

      return clock, fall_time, level_time, parameters


def process_event(parameters, action):

      if action == 0:
            parameters["quickfall"] = True
            parameters["current_speed"] = parameters["fall_speed"]
            parameters["fall_speed"] = QUICKFALL_SPEED

      elif action == 1:
            parameters["current_piece"].x -= 1
            if not(valid_space(parameters["current_piece"], parameters["locked_positions"])):
                  parameters["current_piece"].x += 1 #if block moves outside of boundaries, put it back

      elif action == 2:
            parameters["current_piece"].x += 1
            if not(valid_space(parameters["current_piece"], parameters["locked_positions"])):
                  parameters["current_piece"].x -= 1

      elif action == 3:
            parameters["current_piece"].rotation += 1
            if not(valid_space(parameters["current_piece"], parameters["locked_positions"])):
                  parameters["current_piece"].rotation -= 1

      elif action == 4:
            parameters["current_piece"].y += 1
            if not(valid_space(parameters["current_piece"], parameters["locked_positions"])):
                  parameters["current_piece"].y -= 1
      # Else: do nothing

      return parameters

def update_state(parameters):
      parameters["grid"] = create_grid(parameters["locked_positions"]) #locked_positions may change while running

      if parameters["curse"] == Curses.FAST:
            if parameters["fall_speed"] != FAST_SPEED and parameters["fall_speed"] != QUICKFALL_SPEED:
                  parameters["current_speed"] = parameters["fall_speed"]
            parameters["fall_speed"] = FAST_SPEED

      shape_pos = convert_shape_format(parameters["current_piece"])

      #draw current piece
      for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1: #we're not showing the block before its in the visible part of the grid
                  parameters["grid"][y][x] = parameters["current_piece"].color
                  
      #if piece landed, add its location and color to locked_positions, and go on to the next piece (and create a new next piece for after that)
      if parameters["change_piece"]:
            for pos in shape_pos:
                  p = (pos[0], pos[1])
                  parameters["locked_positions"][p] = parameters["current_piece"].color
            
            inc = clear_rows(parameters["grid"], parameters["locked_positions"])
            
            if inc > 0:
                  parameters["score_increased"] = True
                  parameters["score"] += inc**2 * 500 
                  parameters["cursescore"] += inc**2 * 500
            
            if parameters["curse_turn"] > 0:
                  parameters["curse_turn"] += 1
                  if parameters["curse_turn"] > 3: 
                        parameters["curse"], parameters["curse_turn"] = inflict_curse('tick')
            
            if parameters["cursescore"] >= CURSE_POINT and parameters["curse_turn"] == 0:
                  parameters["curse"], parameters["curse_turn"] = inflict_curse('inflict')
                  parameters["cursescore"] = parameters["score"] % CURSE_POINT
            
            if parameters["quickfall"]:
                  parameters["quickfall"] = False
                  parameters["fall_speed"] = parameters["current_speed"]

            parameters["current_piece"] = parameters["next_piece"]
            parameters["next_piece"] = get_shape(parameters["curse"])
            parameters["change_piece"] = False
      
      return parameters


import matplotlib.pyplot as plt

def visualize_state(grid):
      plt.imshow(grid, cmap='hot', interpolation='nearest')
      plt.show()

clock, fall_time, level_time, parameters = initialize_game()

parameters["current_piece"].y += 10

parameters = update_state(parameters)
visualize_state(parameters["grid"])

action = 1 # Possible actions = [0, 1, 2, 3, 4] = [quickfall, left, right, rotate, down]

parameters = process_event(parameters, action) # Let action change state
parameters = update_state(parameters)
visualize_state(parameters["grid"]) # State
print(parameters["score"]) # Reward