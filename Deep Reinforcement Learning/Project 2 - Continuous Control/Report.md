[//]: # (Image References)

[image1]: index.png "Trained model"

# Report: Project 2(Continous Control) 

### Learning algorithm

#### Overview
The learning algorithm that we use for Continuous Control(Project 2) is DDPG modified to work with 20 agents at once. It is similiar to standard deep Q-learning with a replay buffer. The modifications that are done is that we also have a critic that lets us know how good our actions are. Beyond that, for both actor and critic we have their local versions and target versions to perform temporal difference on to reduce training volitility. An extra step taken for model training is only updating the models once every 20 timesteps and only updating it 10 times and performed gradient clipping. To help ease training, we also added some noise to the actions that we took. We saved the trained actor and critic in "checkpoint_actor.pth" and "checkpoint_critic.pth" respectively. 

#### Replay buffer
The replay buffer allows the agent to train from any situation in the enviroment by not correlating sequential actions with their states to one another. This is done by randomly training on previous experiences with the enviroment and help make the model robust to new situation by disallowing it to become reliant on correlated situation. 

#### Actors and Critics
The reason we have both actors and critics is so that we can train our agent continously even when there is no meaningful response from the enviroment to use. The critic makes up for that by leveraging the actions and response from the enviroment to see if the agent would make a good reward in the future based on current actions in the current state. 

#### Local and Target models
We have both a local and target model for both actor and critic. The reason for doing this is the same as it is in standard deep Q-learning. This is so we have perform temporal difference learning between the current model and the target model to help ensure that the models converges and to reduce volitility. 

#### Updating and Gradient Clipping
As was provided in the notes for the project, both updating after every 20 timesteps with only 10 updates instead of the 20 for the 20 agents and gradient clipping on the critic model. This was done to help ensure convergence of the model and reduce gradient explosion. 

#### Noise
As was speficied in the DDPG paper, we included Ornstein-Uhlenbeck process noise to model random pertubations that is reset after every learning iteration. This makes the model more robust to small changes in the enviroment and the actions taken. 

#### Hyperparameters
HIDDEN_LAYERS = \[400, 300\] # Hidden layer sizes of the actor and critic networks

LR_A = 1e-4              # learning rate for the actor
LR_C = 1e-3              # learning rate for the critic
DISCOUNT_RATE = 0.99     # discount factor
TAU = 1e-3               # soft update factor
BUFFER_SIZE = int(1e6)   # buffer size for replay buffer
BATCH_SIZE = 128         # batch size
WEIGHT_DECAY = 0.0       # L2 weight decay
NOISE_INITIAL = 1.0      # Initial noise factor
NOISE_DECAY = 0.999999   # Noise factor
NOISE_MIN = 0.0          # Minimum noise factor
T_UPDATE = 20            # Number of timesteps between updating the target network
NUM_UPDATES = 10         # Amount of times to update the network every T_UPDATE timesteps


EPS_MAX = 2000           # Number of max episodes to train to TARGET_MEAN_SCORE
T_MAX = 1000             # Number of max timesteps per episode
TARGET_MEAN_SCORE = 30.0 # target score for the enviroment for 'success'

### Training process
We trained our model to achieve a moving average score over the last 100 episodes of atleast 30.0 to ensure "success" of our model. We achieved success in 25 episodes with a final moving average score of 30.13. 
The training process graph of the score from each episode is given below:

![Trained model][image1]

The start of the training process seems odd but we observed that the model achieved a score in the first few episodes that was close to 30. The rest of the training was to bring the moving average up to 30 and to ensure robustness of the model. 

#### Testing
We tested our model on the enviroment and recieved a score of 33.1694992586039

### Future work
In the future, we could make the number of layers in the network deeper and wider or use mode advanced layers like lstm to keep track of the previous sequence of moves to improve temporal understanding of the enviroment. Beyond that, we could train by using more than just the next step to estimate reward and instead use a sequence of future steps properly weighted to help update reward during the training process