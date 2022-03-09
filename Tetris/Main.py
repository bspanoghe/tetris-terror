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


def valid_space(shape, grid):
      accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)] #all possible positions within the boundaries and are white aka not already occupied
      accepted_pos = [j for sub in accepted_pos for j in sub] #take all positions in list and slap them into a one dimensional list
      #e.g.: [[(2, 5)], [(0, 3)]] -> [(2,5), (0,3)]

      formatted = convert_shape_format(shape)

      for pos in formatted:
            if pos not in accepted_pos:
                  if pos[1] > -1: 
                        return False
                        #if the y-value is within boundaries (but x isnt), return false 
                        # => block will start above visible grid aka in negative y boundaries but thats allowed, we don't return false for that
      
      return True


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

def max_score():
      with open('./Hiscores.txt', 'r') as f:
            lines = f.readlines()
            score = lines[0].strip()  #no \n

      return score


def update_score(nscore):
      score = max_score()

      with open('./Hiscores.txt', 'w') as f:
            if nscore > int(score):
                  f.write(str(nscore))
            else:
                  f.write(str(score))


def play_sound(file, type):

      if type == 'song':
            pygame.mixer.music.load('./Sounds/{}.mp3'.format(file))
            pygame.mixer.music.play(-1)
            pygame.mixer.music.set_volume(0.15)

      else:
            global effect
            effect = pygame.mixer.Sound('./Sounds/{}.wav'.format(file))
            effect.play()
            effect.set_volume(0.2)


def inflict_curse(effect = None):
      if effect == 'inflict':
            CURSE = random.choice(ACTUAL_CURSES)
            #CURSE = Curses.LIES
            turn = 1

      if effect == 'tick':
            CURSE = Curses.NO_CURSE
            turn = 0
                  
      return CURSE, turn



def main(win):
	
      locked_positions = {}
      grid = create_grid(locked_positions)

      change_piece = False
      run = True
      quickfall = False
      current_piece = get_shape()
      next_piece = get_shape()
      clock = pygame.time.Clock()
      fall_time = 0
      fall_speed = 0.27
      level_time = 0
      current_speed = 0.27
      score = 0
      cursescore = 0
      turn = 0
      CURSE = Curses.NO_CURSE

      while run: #so you can quit
            grid = create_grid(locked_positions) #locked_positions may change while running
            fall_time += clock.get_rawtime() #gets amount of time since clock.tick() clicked, starts at 0
            level_time += clock.get_rawtime()
            clock.tick()

            if CURSE == Curses.FAST:
                  if fall_speed != FAST_SPEED and fall_speed != QUICKFALL_SPEED:
                        current_speed = fall_speed
                  fall_speed = FAST_SPEED

            if level_time/1000 >= 10:
                  level_time = 0
                  if fall_speed > TERMINAL_SPEED:   #minimum value for fall_speed so you don't start going super fucking fast after 2 minutes
                        fall_speed -= 0.005
                  
            if fall_time/1000 >= fall_speed:
                  #fall_time is given in ms, speed in s
                  fall_time = 0
                  current_piece.y += 1
                  #move your piece down every tick
                  if not(valid_space(current_piece, grid)) and current_piece.y > 0:
                        current_piece.y -= 1
                        change_piece = True #the piece has landed, time to get to the next one (and lock current piece's pos)
                  if CURSE != Curses.FAST and fall_speed == FAST_SPEED:
                        fall_speed = current_speed

            for event in pygame.event.get():
                  if event.type == pygame.QUIT:
                        run = False
                        pygame.display.quit()
                        quit()
                  
                  if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                              score += CURSE_POINT
                              cursescore += CURSE_POINT
                        if event.key == pygame.K_SPACE:
                              quickfall = True
                              current_speed = fall_speed
                              fall_speed = QUICKFALL_SPEED

                        if str(CURSE.name) == 'MISDIRECTION':
                              if event.key == pygame.K_LEFT:
                                    current_piece.x += 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.x -= 1
                                    
                              elif event.key == pygame.K_RIGHT:
                                    current_piece.x -= 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.x += 1 #if block moves outside of boundaries, put it back

                              elif event.key == pygame.K_UP:
                                    current_piece.y += 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.y -= 1      

                              elif event.key == pygame.K_DOWN:
                                    current_piece.rotation += 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.rotation -= 1

                        else:
                              if event.key == pygame.K_LEFT:
                                    current_piece.x -= 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.x += 1 #if block moves outside of boundaries, put it back

                              elif event.key == pygame.K_RIGHT:
                                    current_piece.x += 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.x -= 1

                              elif event.key == pygame.K_UP:
                                    current_piece.rotation += 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.rotation -= 1

                              elif event.key == pygame.K_DOWN:
                                    current_piece.y += 1
                                    if not(valid_space(current_piece, grid)):
                                          current_piece.y -= 1


            shape_pos = convert_shape_format(current_piece)

            #draw current piece
            for i in range(len(shape_pos)):
                  x, y = shape_pos[i]
                  if y > -1: #we're not showing the block before its in the visible part of the grid
                        grid[y][x] = current_piece.color
                        
            #if piece landed, add its location and color to locked_positions, and go on to the next piece (and create a new next piece for after that)
            if change_piece:
                  for pos in shape_pos:
                        p = (pos[0], pos[1])
                        locked_positions[p] = current_piece.color
                  
                  inc = clear_rows(grid, locked_positions)

                  if inc > 0:
                        play_sound(random.choice(sounds), 'effect')

                  score += inc**2 * 500 
                  cursescore += inc**2 * 500

                  if turn > 0:
                        turn += 1
                        if turn > 3: 
                              CURSE, turn = inflict_curse('tick')
                  
                  if cursescore >= CURSE_POINT and turn == 0:
                        CURSE, turn = inflict_curse('inflict')
                        cursescore = score % CURSE_POINT
                  
                  if quickfall:
                        quickfall = False
                        fall_speed = current_speed

                  current_piece = next_piece
                  next_piece = get_shape(CURSE)
                  false_shape = get_shape(CURSE)
                  while false_shape == next_piece:
                        false_shape = get_shape(CURSE)
                  change_piece = False
                                          
            
            draw_window(win, grid, score, CURSE)

            if str(CURSE.name) == "LIES": 
                  draw_next_shape(false_shape, win, CURSE)
            else:
                  draw_next_shape(next_piece, win, CURSE)

            pygame.display.update()

            if check_lost(locked_positions):
                  draw_text_middle("YOU LOST AHAHAHA", 80, (255, 255, 255), win)
                  pygame.mixer.music.stop()
                  play_sound('Failed', 'effect')
                  pygame.display.update()
                  pygame.time.delay(2000)
                  effect.stop()
                  run = False
                  update_score(score)


def main_menu(win):
      run = True
      while run:
            win.fill((0, 0, 0))
            draw_text_middle('Press any key to begin', 60, (255, 255, 255), win)

            pygame.display.update()
            for event in pygame.event.get():
                  if event.type == pygame.QUIT:
                        run = False
                  if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                              run = False
                        else:
                              play_sound('Vento', 'song')
                              main(win)
      
      
      pygame.display.quit()
      main(win)
	
win = pygame.display.set_mode((s_width, s_height)) #define pygame window
pygame.display.set_caption('Tetris')
main_menu(win)  # start game