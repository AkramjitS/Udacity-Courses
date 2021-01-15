[//]: # (Image References)

[Trained_model]: (index1.png) "Training process of MADDPG model"

# Report: Project 3 (Tennis) 

### Learning algorithm

#### Overview
In this project, we use a MADDPG method to train two tennis playing agents for project 3 (Tennis). This version is built similiar to DDPG with two individual agents that both are trained with the global state for training their actor and their own actions and global state for training their critic. They both have their own actors and critics, both having a local and target version for TD(temporal difference) learning. We use a seperate replay buffer for each agent and add noise to the actions and use gradient clipping to help model robustness and stability. We saved the local actors and critics for both agents as checkpoint_actor0.pth, checkpoint_actor1.pth and checkopint_critic0.pth, checkpoint_critic1.pth.

#### Models
The actor model is a feed forward neural network with relu activation and tanh on the output. 
The architecture is: 48 input dimensions -> 400 hidden dimensions -> 400 hidden dimensions -> 2 output dimensions. 

The critic model is a feed forward neural network with relu that takes states and actions as inputs and no function on the output. We feed the input layer with the states and send it to the first hidden layer where it is concatenated with the actions and then sent to the second hidden layer. 
The architecture is: 48 input dimensions -> 400 hidden dimensions + 4 input dimensions -> 400 hidden dimensions -> 1 output dimension.

#### Replay buffer
The replay buffer allows the agent to train from any situation in the enviroment by not correlating sequential actions with their states to one another. This is done by randomly training on previous experiences with the enviroment and help make the model robust to new situation by disallowing it to become reliant on correlated situation. 

#### Actors and Critics
The reason we have both actors and critics is so that we can train our agent continously even when there is no meaningful response from the enviroment to use. The critic makes up for that by leveraging the actions and response from the enviroment to see if the agent would make a good reward in the future based on current actions in the current state. 

#### Local and Target models
We have both a local and target model for both actor and critic. The reason for doing this is the same as it is in standard deep Q-learning. This is so we have perform temporal difference learning between the current model and the target model to help ensure that the models converges and to reduce volitility. 

#### Updating and Gradient Clipping
As was provided in the notes for the project, both updating after every 10 timesteps with only 10 updates instead of the 20 for the 20 agents and gradient clipping on the critic model. This was done to help ensure convergence of the model and reduce gradient explosion. 

#### Noise
As was speficied in the DDPG paper, we included Ornstein-Uhlenbeck process noise to model random pertubations that is reset after every learning iteration. This makes the model more robust to small changes in the enviroment and the actions taken. 

#### Hyperparameters
HIDDEN_LAYERS = \[400, 400\] # Hidden layer sizes of the actor and critic networks

LR_A = 1e-4              # learning rate for the actor
LR_C = 1e-4              # learning rate for the critic
DISCOUNT_RATE = 0.99     # discount factor
TAU = 1e-3               # soft update factor
BUFFER_SIZE = int(1e6)   # buffer size for replay buffer
BATCH_SIZE = 128         # batch size
WEIGHT_DECAY = 0.0       # L2 weight decay
NOISE_INITIAL = 1.0      # Initial noise factor
NOISE_DECAY = 0.999999   # Noise factor
NOISE_MIN = 0.005        # Minimum noise factor
T_UPDATE = 10            # Number of timesteps between updating the target network
NUM_UPDATES = 10         # Amount of times to update the network every T_UPDATE timesteps


EPS_MAX = 4000           # Number of max episodes to train to TARGET_MEAN_SCORE
T_MAX = 1000             # Number of max timesteps per episode
TARGET_MEAN_SCORE = 0.65 # target score for the enviroment for 'success'

### Training process
We trained our model to achieve a moving average score over the last 100 episodes of atleast 0.65 to ensure "success" of our model. We achieved success in 2541 episodes with a final moving average score of 0.6568. 
The training process graph for the score for each episode, the current moving average per episode, and the score to ensure beating the enviroment as is required by the project of 0.5, below my own requirement of the model, is as follows:

![Trained model]

#### Testing
We tested our model on the enviroment for 5 iterations and these are the score we got back for each iteration:
-  2.600000038743019
-  2.600000038743019
-  1.095000016503036
-  2.600000038743019
- -0.004999999888241291

### Future work
In the future, we could make the number of layers in the network deeper and wider or use mode advanced layers like lstm to keep track of the previous sequence of moves to improve temporal understanding of the enviroment. Beyond that, we could train by using more than just the next step to estimate reward and instead use a sequence of future steps properly weighted to help update reward during the training process. Beyond that, we could train our critics using both the agents actions instead of just their own and look into using more sophisticated experience replay methods like priority experience replay. 