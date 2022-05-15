# -*- coding: utf-8 -*-
# Imports & environment creation

import numpy as np
from PIL import Image
import Tetris.Main_computerman as Tetris
import pygame
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import initializers

logbook_dir = "log.txt"
"""# Functions"""

"""## Visualization"""


def showstate(state):
  return Image.fromarray(state).show()


"""## Network"""

def create_model(convneurons, denseneurons, etha, input_shape, num_outputs):
 
  n = len(convneurons)
  m = len(denseneurons)

  # create linear model
  model = Sequential()
  # we start with a first convolutional block of layers
  model.add(Conv2D(filters = convneurons[0], kernel_size = (4, 4), padding='same', input_shape = input_shape, strides = 1, kernel_initializer = initializers.variance_scaling(scale=2)))
  model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # middle layers

  model.add(Conv2D(filters = convneurons[1], kernel_size = (3, 3), padding='same', strides = 1, kernel_initializer = initializers.variance_scaling(scale=2)))
  model.add(Activation('relu'))
  model.add(Conv2D(filters = convneurons[2], kernel_size = (3, 3), padding='same', strides = 1, kernel_initializer = initializers.variance_scaling(scale=2)))
  model.add(Activation('relu'))
  

  # end of convolutional layers, start of 'hidden' dense layers (can be more than 1 if necessary)
  model.add(Flatten())
  for i in range(m):
    model.add(Dense(denseneurons[i], kernel_initializer = initializers.variance_scaling(scale=2)))
    model.add(Activation('relu'))

  # Final dense layer = linear classifier
  model.add(Dense(num_outputs, kernel_initializer = initializers.variance_scaling(scale=2)))
  model.add(Activation('linear'))

  opt = tf.keras.optimizers.Adam(learning_rate=etha)

  model.compile(loss='mse', optimizer=opt, metrics=["mse"],)

  return model

def get_Q_values(state, model): # For one input
  state = np.reshape(state, (1,) + state.shape) # Specify to model batch size of 1
  return model(state).numpy()[0] # Returns list of lists, but we only use one input (for which model(x) is faster than model.predict(x))

def get_Q_values_multiple_states(state, model): # For multiple input states
  return model(state).numpy()

def choose_action(state, EPS, model, legal_actions):
  if np.random.rand() < EPS: # With probability epsilon:
    #return env.action_space.sample() # Return random action
    return random_action(legal_actions)
  else:
    q_values = get_Q_values(state, model)
    return np.argmax(q_values) # Return index of action with highest q-value

def update_network(train_network, target_network, memory, state_memory, endstate_memory, BATCH_SIZE, MEMORY_SIZE, GAMMA):
  batch_ids = np.random.randint(MEMORY_SIZE, size=BATCH_SIZE) # Pick a list of random IDs within MEMORY_SIZE of size BATCH_SIZE

  actions = memory[batch_ids, 0].astype("int") # We'll be using the action as an index, so no floating numbers
  rewards = memory[batch_ids, 1]
  dones = memory[batch_ids, 2]

  state_chains = np.array([np.dstack([state_memory[id] for id in range(batch_id+1-STATE_HISTORY_LEN, batch_id+1)]) for batch_id in batch_ids])
  endstate_chains = np.array([np.dstack([endstate_memory[id] for id in range(batch_id+1-STATE_HISTORY_LEN, batch_id+1)]) for batch_id in batch_ids])

  targets = train_network.predict(state_chains)
  target_Q_values = get_Q_values_multiple_states(endstate_chains, target_network)

  for i in range(BATCH_SIZE):
    if dones[i]: # value of done => If it's done, the target is just the reward
      targets[i, actions[i]] = rewards[i]
    else: # If it wasn't dead, predict max Q-value of follow-up state
      targets[i, actions[i]] = rewards[i] + GAMMA * np.amax(target_Q_values[i])
  
  #hist = 
  hist = train_network.fit(state_chains, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)

  ##MSE
  MSE = hist.history['loss']
  return MSE

def preprocess_state(state):
  # turn list of lists into np array
  img = np.array(state)
  # Collapse 3rd dimension (color) through summing & reshaping
  img = np.sum(img, axis = 2)
  # Binarize
  img = img.astype("bool")

  return img

def random_action(legal_actions):
  """Returns random index of a legal action"""
  return np.random.randint(len(legal_actions))

def generate_startingstate():
  _, _, _, parameters = Tetris.initialize_game() # geeft clock, fall_time, level_time, parameters 
  state = parameters["grid"]
  state = preprocess_state(state)
  
  return state, parameters

def update_memory(memorycounter, state, action, reward, done, endstate, state_memory, endstate_memory, memory):
  state_memory[memorycounter, ] = state
  endstate_memory[memorycounter, ] = endstate
  memory[memorycounter, ] = np.array((action, reward, done))

def update_state_chain(state_chain, endstate):
  endstate = np.reshape(endstate, endstate.shape + (1,))
  state_chain[:,:,0:-1], state_chain[:,:,-1:] = state_chain[:,:,1:], endstate # Pop out endstate on position 1 and add the most recent endstate
  return state_chain

def test_game(train_network, EPS, visualized = False):
  time_played = 0
  done = False
  endstate, parameters = generate_startingstate()
  state_chain = np.dstack([endstate for _ in range(STATE_HISTORY_LEN)])

  total_reward = 0
  max_steps = 100000

  if visualized:
    win = pygame.display.set_mode((800, 700)) #define pygame window
    pygame.display.set_caption('Tetris')
    win.fill((0, 0, 0))
    Tetris.draw_text_middle('Press any key to begin', 60, (255, 255, 255), win)
    pygame.display.update()
    waiting = True

    while waiting:
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          waiting = False

  while not done and time_played < max_steps: # Keep going until you're dead/done or the memory is full

    if visualized:
      Tetris.draw_window(win, parameters["grid"], parameters["score"], parameters["curse"])
      pygame.display.update()

    action = choose_action(state_chain, EPS, train_network, legal_actions)
    endstate, reward, done, _, parameters = Tetris.game_step(parameters, legal_actions[action], time_played)
    endstate = preprocess_state(endstate)
    state_chain = update_state_chain(state_chain, endstate)

    if done: # the game is over
        reward -= DEATH_PENALTY
    
    time_played += 1
    total_reward += reward

  if visualized:
    pygame.display.quit()
  
  return total_reward, parameters["score"], time_played

def visualize_memory(state):
    plt.imshow(state, cmap='hot', interpolation='nearest')
    plt.show()


"""# Environment"""

legal_actions = [1, 2, 3, 4]

"""# Parameters"""

TRAINING_STEPS = 4 # Amount of in game-steps before network updates (steps in a session)
LEN_EPISODE = 1000 # Amount of sessions until target NN (frozen NN) gets updated with trained NN
NUM_EPISODES = 3000 # Amount of episodes the NN trains
NUM_EPISODES_LOG = 1 # log data for visualization every specified number of episodes 
EXPLORE_LEN = 30 # Amount of episodes in the exploration phase (EPS goes to EPS_min in this time period) ##np.ceil((np.log(EPS_MIN)-np.log(EPS))/np.log(EPS_DECAY))

MEMORY_SIZE = 10000 # Amount of actions remembered at one time
INITIAL_MEMORY_SIZE = 2000 # Part of memory that gets populated by a random agent
EPS = 1 # Chance of taking a random action instead of the one with highest Q-value
EPS_MIN = 0.1 # Minimum value of EPS
EPS_DECAY = (EPS_MIN/EPS)**(1/EXPLORE_LEN) # Amount EPS changes every time an episode has passed
REWARD_EPS_DECAY = 1 # Amount of EPS changes every time a threshold is passed

GAMMA = 0.99 # Weight of rewards further in future
STATE_HISTORY_LEN = 2 # Amount of frames used as input for network
DEATH_PENALTY = 10
REWARD_THRESHOLD = 1000
REWARD_INCREASE = 1000

CONVNEURONS = [64, 64, 64]
DENSENEURONS = [256, 64]
ETHA = 0.00025
BATCH_SIZE = 32
BATCHNORM = False

state, _ = generate_startingstate()

state_shape = state.shape
input_shape = state.shape + (STATE_HISTORY_LEN, )
action_size = len(legal_actions) # Amount of possible actions to take

"""# Initialization"""

# We split up the memory into 3 part to account for the variable dimensionality of the states depending on the chosen game
memory = np.zeros((MEMORY_SIZE, 3), dtype = "int32") # store the chosen action, the reward and helpther you were done
state_memory = np.zeros((MEMORY_SIZE,) + state_shape, dtype = "uint8") # state_shape is a tuple, so we turn MEMORY_SIZE into a tuple before concatenating
endstate_memory = np.zeros((MEMORY_SIZE,) + state_shape, dtype = "uint8")

train_network = create_model(CONVNEURONS, DENSENEURONS, ETHA, input_shape, action_size)
train_network.summary()

train_network.load_weights("latest_weights_new.h5")

# Simply using target_network = train_network will make target_network identical to train_network at all times!
# This makes freezing the target network impossible as it will be updated together with train_network
# Create a separate target_network and copy train_network's weights instead to avoid this
target_network = create_model(CONVNEURONS, DENSENEURONS, ETHA, input_shape, action_size)
target_network.set_weights(train_network.get_weights())

with open(logbook_dir, 'a') as writefile:
  writefile.write(f"\n------------------------------------------------------------------------------------------------------------------")
  writefile.write(f"\nTRAINING_STEPS:{TRAINING_STEPS},LEN_EPISODE:{LEN_EPISODE},NUM_EPISODES:{NUM_EPISODES}")
  writefile.write(f"\nMEMORY_SIZE:{MEMORY_SIZE},EPS:{EPS},EPS_DECAY:{EPS_DECAY},EPS_MIN:{EPS_MIN},GAMMA:{GAMMA},STATE_HISTORY_LEN:{STATE_HISTORY_LEN}")
  writefile.write(f"\nCONVNEURONS:{CONVNEURONS},DENSENEURONS:{DENSENEURONS},ETHA:{ETHA},BATCH_SIZE:{BATCH_SIZE},BATCHNORM:{BATCHNORM}")
  writefile.write(f"\nEXTRA NOTES: Actually lets go bigger dense layer again. More power, baby!")  
  writefile.write(f"\n------------------------------------------------------------------------------------------------------------------")
  writefile.close()

# Initialize memory

memorycounter = 0
history_reward = 0 # Remember the reward of the last STATE_HISTORY_LEN frames
done = False
next_piece = 0

while memorycounter < INITIAL_MEMORY_SIZE: # Keep going until the memory is full
  done = False
  time_played = 0
  endstate, parameters = generate_startingstate()

  while not done and memorycounter < INITIAL_MEMORY_SIZE: # Keep going until you're dead/done or the memory is full
    state = endstate
    action = random_action(legal_actions) # Initialization of memory done with random steps
    endstate, reward, done, next_piece, parameters = Tetris.game_step(parameters, legal_actions[action], time_played)
    endstate = preprocess_state(endstate)

    if done: # the game is over
      reward -= DEATH_PENALTY
    time_played += 1

    update_memory(memorycounter, state, action, reward, done, endstate, state_memory, endstate_memory, memory)
    memorycounter += 1

    action = random_action(legal_actions)

"""# Training"""

memory_full = False
stepcounter = 0 # Keep track of in-game steps
traincounter = 0 # Keep track of amount of training sessions (every time it updates)
episodecounter = 0 # Keep track of episodes (every time target network updates)
time_played = 0 # Keep track how long you stay alive
history_reward = 0 # Keep track of the reward over the last frames until you save data to the memory
score = 0
next_piece = 0 # This is the index of the next_piece in the list of shapes (see Piece class in Main.py)

endstate, parameters = generate_startingstate()
state_chain = np.dstack([endstate for _ in range(STATE_HISTORY_LEN)])
action = choose_action(state_chain, EPS, train_network, legal_actions)

MSE_list = [] # Store score, time_played and MSE
best_score = 0

while episodecounter <= NUM_EPISODES:
  print("Now running episode", episodecounter)
  while traincounter <= LEN_EPISODE:
    while stepcounter <= TRAINING_STEPS: # Keep going until training sesh is done
      if done: # You died
        # Reset the environment, a new beginning
        time_played = 0
        done = False
        endstate, parameters = generate_startingstate()
        state_chain = np.dstack([endstate for _ in range(STATE_HISTORY_LEN)])

      while not done and stepcounter <= TRAINING_STEPS: # Keep going until you're dead/done or the episode is done
        
        state = endstate
        action = choose_action(state_chain, EPS, train_network, legal_actions)
        endstate, reward, done, next_piece, parameters = Tetris.game_step(parameters, legal_actions[action], time_played)
        endstate = preprocess_state(endstate)
        state_chain = update_state_chain(state_chain, endstate)

        if done: # the game is over
          reward -= DEATH_PENALTY
        
        stepcounter += 1
        time_played += 1

        update_memory(memorycounter, state, action, reward, done, endstate, state_memory, endstate_memory, memory)
        history_reward = 0
        
        memorycounter += 1
        if memorycounter == MEMORY_SIZE:
            memorycounter = 0
            memory_full = True

    # Training session done

    traincounter += 1

    stepcounter = 0  
    if memory_full: # Don't sample from your entire memory before it's full
      MSE = update_network(train_network, target_network, memory, state_memory, endstate_memory, BATCH_SIZE, MEMORY_SIZE, GAMMA)
    else:
      MSE = update_network(train_network, target_network, memory, state_memory, endstate_memory, BATCH_SIZE, memorycounter, GAMMA)
    MSE_list.append(MSE)
    
  # Episode done
  episodecounter += 1
  traincounter = 0
  target_network.set_weights(train_network.get_weights())

  test_score, true_score, step_counter = test_game(train_network, EPS)
  print("Training score: {}\nTrue score: {}\nTime survived: {}".format(test_score, true_score, step_counter))

  if episodecounter-1 <= EXPLORE_LEN and EPS > EPS_MIN:
    EPS = EPS * EPS_DECAY
  elif test_score >= REWARD_THRESHOLD:
    EPS = EPS * REWARD_EPS_DECAY
    REWARD_THRESHOLD += REWARD_INCREASE
  
  if episodecounter % NUM_EPISODES_LOG == 0:
    # test_score = Tetris.play_game_visualized(train_network, EPS)
    with open(logbook_dir, 'a') as writefile:
      #writefile.write(f"\n{np.mean(scores[0])},{np.mean(scores[1])},{np.mean(scores[2])},{EPS}")
      writefile.write(f"\n{test_score},{true_score},{step_counter},{np.mean(MSE_list)},{EPS},{episodecounter}")
      writefile.close()
    MSE_list = []
  
  train_network.save_weights("latest_weights_new.h5")
  if test_score > best_score:
    train_network.save_weights("best_weights_new.h5")
    best_score = np.copy(test_score)