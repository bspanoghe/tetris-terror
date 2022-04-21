# -*- coding: utf-8 -*-
# Imports & environment creation

import numpy as np
from PIL import Image
import Tetris.Main_computerman as Tetris
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
  # middle layers

  model.add(Conv2D(filters = convneurons[1], kernel_size = (4, 4), padding='same', strides = 1, kernel_initializer = initializers.variance_scaling(scale=2)))
  model.add(Activation('relu'))
  model.add(Conv2D(filters = convneurons[2], kernel_size = (4, 4), padding='same', strides = 1, kernel_initializer = initializers.variance_scaling(scale=2)))
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

def get_Q_values(state, model):
  state = np.reshape(state, (1,) + state.shape) # Specify to the model it's getting one state
  return model(state).numpy()[0] # Returns list of lists, but we only use one input (for which model(x) is faster than model.predict(x))

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
  next_pieces = memory[batch_ids, 3]

  states = state_memory[batch_ids]/256 # Normalize images at the last moment
  endstates = endstate_memory[batch_ids]/256

  targets = train_network.predict(states)

  for i in range(BATCH_SIZE):
    if dones[i]: # value of done => If it's done, the target is just the reward
      targets[i, actions[i]] = rewards[i]
    else: # If it wasn't dead, predict max Q-value of follow-up state
      target_Q_values = get_Q_values(endstates[i, :], target_network)
      targets[i, actions[i]] = rewards[i] + GAMMA * np.amax(target_Q_values)
  
  #hist = 
  hist = train_network.fit(states, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)

  ##MSE
  MSE = hist.history['loss']
  return MSE

def preprocess_state(state):
  # Crop and resize the image
  img = np.array(state)
  img = img[1:171:2, ::2]
  # Convert to grayscale
  img = (0.3*img[:, :, 0] + 0.59*img[:, :, 1] + 0.11*img[:, :, 2]).astype("uint8")
  # Improve image contrast
  #img[img == 31] = 0 #31 is bg color
  # Next we DONT normalize the image from 0 to 1
  #img = img/256 ## Don't chance datatype from uint8 to float until the last moment to conserve memory!
  img = img.reshape(img.shape + (1,))
  return img

def random_action(legal_actions):
  """Returns random index of a legal action"""
  return np.random.randint(len(legal_actions))

def generate_startingstate(STATE_HISTORY_LEN):
  _, _, _, parameters = Tetris.initialize_game() # geeft clock, fall_time, level_time, parameters 
  state = parameters["grid"]
  state = preprocess_state(state)
  state = np.concatenate([state for _ in range(STATE_HISTORY_LEN)], axis = 2)
  return state, parameters

def update_state(state, frame):
  #k = 3 ## for RGB
  k = 1 ## graystate
  frame = preprocess_state(frame)
  state[:,:,0:-k], state[:,:,-k:] = state[:,:,k:], frame
  return state

"""# Environment"""

legal_actions = [0, 1, 2, 3, 4, 5]

"""# Parameters"""

TRAINING_STEPS = 4 # Amount of in game-steps before network updates (steps in a session)
LEN_EPISODE = 1000 # Amount of sessions until target NN (frozen NN) gets updated with trained NN
NUM_EPISODES = 1000 # Amount of episodes the NN trains
NUM_EPISODES_LOG = 1 # log data for visualization every specified number of episodes 
EXPLORE_LEN = int(NUM_EPISODES/50) # Amount of episodes in the exploration phase (EPS goes to EPS_min in this time period) ##np.ceil((np.log(EPS_MIN)-np.log(EPS))/np.log(EPS_DECAY))

MEMORY_SIZE = int(TRAINING_STEPS*LEN_EPISODE*NUM_EPISODES / 50) # Amount of actions remembered at one time
INITIAL_MEMORY_SIZE = int(MEMORY_SIZE / 20) # Part of memory that gets populated by a random agent
EPS = 1 # Chance of taking a random action instead of the one with highest Q-value
EPS_MIN = 0.1 # Minimum value of EPS
EPS_DECAY = (EPS_MIN/EPS)**(1/EXPLORE_LEN) # Amount EPS changes every time an episode has passed
REWARD_EPS_DECAY = 1 # Amount of EPS changes every time a threshold is passed

GAMMA = 0.99 # Weight of rewards further in future
STATE_HISTORY_LEN = 1 # Amount of frames that are skipped then added to memory
DEATH_PENALTY = 50
REWARD_THRESHOLD = 1000
REWARD_INCREASE = 1000

CONVNEURONS = [32, 64, 64]
DENSENEURONS = [256]
ETHA = 0.00025
BATCH_SIZE = 32
BATCHNORM = False

state, _ = generate_startingstate(STATE_HISTORY_LEN)

state_shape = state.shape
action_size = len(legal_actions) # Amount of possible actions to take

"""# Initialization"""

# We split up the memory into 3 part to account for the variable dimensionality of the states depending on the chosen game
memory = np.zeros((MEMORY_SIZE, 4), dtype = "int32") # store the chosen action, the reward, whether you were done and the next piece
state_memory = np.zeros((MEMORY_SIZE,) + state_shape, dtype = "uint8") # state_shape is a tuple, so we turn MEMORY_SIZE into a tuple before concatenating
endstate_memory = np.zeros((MEMORY_SIZE,) + state_shape, dtype = "uint8")

train_network = create_model(CONVNEURONS, DENSENEURONS, ETHA, state_shape, action_size)
train_network.summary()

# Simply using target_network = train_network will make target_network identical to train_network at all times!
# This makes freezing the target network impossible as it will be updated together with train_network
# Create a separate target_network and copy train_network's weights instead to avoid this
target_network = create_model(CONVNEURONS, DENSENEURONS, ETHA, state_shape, action_size)
target_network.set_weights(train_network.get_weights())

with open(logbook_dir, 'a') as writefile:
  writefile.write(f"\n------------------------------------------------------------------------------------------------------------------")
  writefile.write(f"\nTRAINING_STEPS:{TRAINING_STEPS},LEN_EPISODE:{LEN_EPISODE},NUM_EPISODES:{NUM_EPISODES}")
  writefile.write(f"\nMEMORY_SIZE:{MEMORY_SIZE},EPS:{EPS},EPS_DECAY:{EPS_DECAY},EPS_MIN:{EPS_MIN},GAMMA:{GAMMA},STATE_HISTORY_LEN:{STATE_HISTORY_LEN}")
  writefile.write(f"\nCONVNEURONS:{CONVNEURONS},DENSENEURONS:{DENSENEURONS},ETHA:{ETHA},BATCH_SIZE:{BATCH_SIZE},BATCHNORM:{BATCHNORM}")
  writefile.write(f"\nEXTRA NOTES: ")  
  writefile.write(f"\n------------------------------------------------------------------------------------------------------------------")
  writefile.close()

# Initialize memory

memorycounter = 0
stepcounter = 0
history_reward = 0 # Remember the reward of the last STATE_HISTORY_LEN frames
done = False
next_piece = 0

while memorycounter < INITIAL_MEMORY_SIZE: # Keep going until the memory is full
  if done: # You died
    # Remembering your death is the only way to grow
    state = all_states[:,:,0:state_shape[-1]]
    endstate = all_states[:,:,state_shape[-1]:]

    memory[memorycounter, ] = [np.copy(action), np.copy(history_reward), np.copy(done), np.copy(next_piece)]
    state_memory[memorycounter, ] = np.copy(state)
    endstate_memory[memorycounter, ] = np.copy(endstate)
    history_reward = 0
    memorycounter += 1

  done = False
  endstate, parameters = generate_startingstate(STATE_HISTORY_LEN)
  all_states = np.concatenate((endstate, endstate), axis = 2) # Keeps track of states + endstates
  action = random_action(legal_actions) # Initialization of memory done with random steps
  time_played = 0

  while not done and memorycounter < INITIAL_MEMORY_SIZE: # Keep going until you're dead/done or the memory is full
    history_counter = time_played % STATE_HISTORY_LEN

    state, reward, done, next_piece, parameters = Tetris.game_step(parameters, legal_actions[action]) 

    if done: # the game is over
      history_reward -= DEATH_PENALTY
    history_reward += reward
    stepcounter += 1
    time_played += 1
    all_states = update_state(all_states, state)

    if history_counter == STATE_HISTORY_LEN-1:
      state = all_states[:,:,0:state_shape[-1]]
      endstate = all_states[:,:,state_shape[-1]:]

      memory[memorycounter, ] = [np.copy(action), np.copy(history_reward), np.copy(done), np.copy(next_piece)]
      state_memory[memorycounter, ] = np.copy(state)
      endstate_memory[memorycounter, ] = np.copy(endstate)
      history_reward = 0

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

endstate, parameters = generate_startingstate(STATE_HISTORY_LEN)
all_states = np.concatenate((endstate, endstate), axis = 2)
action = choose_action(endstate, EPS, train_network, legal_actions)

MSE_list = [] # Store score, time_played and MSE

while episodecounter <= NUM_EPISODES:
  print("Now running episode", episodecounter)
  while traincounter <= LEN_EPISODE:
    while stepcounter <= TRAINING_STEPS: # Keep going until training sesh is done
      if done: # You died
        # Remembering your death is the only way to grow
        state = all_states[:,:,0:state_shape[-1]]
        endstate = all_states[:,:,state_shape[-1]:]

        memory[memorycounter, ] = [np.copy(action), np.copy(history_reward), np.copy(done), np.copy(next_piece)]
        state_memory[memorycounter, ] = np.copy(state)
        endstate_memory[memorycounter, ] = np.copy(endstate)
        history_reward = 0
        memorycounter += 1
        if memorycounter == MEMORY_SIZE:
          memorycounter = 0
          memory_full = True

        # Reset the environment, a new beginning
        done = False

        endstate, parameters = generate_startingstate(STATE_HISTORY_LEN)
        all_states = np.concatenate((endstate, endstate), axis = 2)
        action = choose_action(endstate, EPS, train_network, legal_actions)

        time_played = 0
        score = 0
        next_piece = 0

      while not done and stepcounter <= TRAINING_STEPS: # Keep going until you're dead/done or the episode is done
        history_counter = time_played % STATE_HISTORY_LEN

        state, reward, done, next_piece, parameters = Tetris.game_step(parameters, legal_actions[action]) 
        if done: # the game is over
          history_reward -= DEATH_PENALTY
        history_reward += reward 
        score += reward
        stepcounter += 1
        time_played += 1
        all_states = update_state(all_states, state)

        if history_counter == STATE_HISTORY_LEN-1: # Only store data to memory and choose an action every STATE_HISTORY_LEN times (action thus gets repeated STATE_HISTORY_LEN times)
          state = all_states[:,:,0:state_shape[-1]]
          endstate = all_states[:,:,state_shape[-1]:]

          memory[memorycounter, ] = [np.copy(action), np.copy(history_reward), np.copy(done), np.copy(next_piece)]
          state_memory[memorycounter, ] = np.copy(state)
          endstate_memory[memorycounter, ] = np.copy(endstate)
          history_reward = 0
          
          memorycounter += 1
          if memorycounter == MEMORY_SIZE:
            memorycounter = 0
            memory_full = True

          action = choose_action(endstate, EPS, train_network, legal_actions)
          #print(action)

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

  _, _, _, parameters_test =  Tetris.initialize_game()
  state_test = parameters_test["grid"]
  state_test = preprocess_state(state_test)
  state_test = np.concatenate([state_test for _ in range(STATE_HISTORY_LEN)], axis = 2)

  done2 = False
  stepcounter2 = 0
  score2 = 0

  while not done2:
    action2 = choose_action(state_test, 0, train_network, legal_actions)
    state2, reward_test, done2, next_piece2, parameters2 = Tetris.game_step(parameters_test, legal_actions[action2])
    stepcounter2 += 1
    score2 += reward_test
    state_test = update_state(state_test, state2)

  if episodecounter-1 <= EXPLORE_LEN and EPS > EPS_MIN:
    EPS = EPS * EPS_DECAY
  elif score2 >= REWARD_THRESHOLD:
    EPS = EPS * REWARD_EPS_DECAY
    REWARD_THRESHOLD += REWARD_INCREASE
  
  if episodecounter % NUM_EPISODES_LOG == 0:
    print("Lasted", stepcounter2, " frames and got a score of", score2, "!")
    with open(logbook_dir, 'a') as writefile:
      #writefile.write(f"\n{np.mean(scores[0])},{np.mean(scores[1])},{np.mean(scores[2])},{EPS}")
      writefile.write(f"\n{score2},{stepcounter2},{np.mean(MSE_list)},{EPS},{episodecounter}")
      writefile.close()
    MSE_list = []
  
  train_network.save_weights("latest_weights.h5")