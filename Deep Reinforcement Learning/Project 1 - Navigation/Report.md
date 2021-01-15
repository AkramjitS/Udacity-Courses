[//]: # (Image References)

[image1]: index.png "Trained model"


# Report

### Learning algorithm

#### Overview
The learning algorithm that we employ in project 1(Navigation) is that of standard deep Q-learning. This includes the usage of a Q-function that is modelled by a neural network as well as using two neural networks and a replay buffer in the training of the agent. The trained model for the agent was saved to the file "checkpoint.pth".

#### Two Q-functions
We use the two neural networks, that represent two seperate copies of the Q-function so that we can learn better ensure that the model that we are training will converge to a good agent. Specifically, the two Q-functions are the target Q-function that keeps track of the previous iterations internal weights will the other active Q-function keeps updates the model activelly. Both models are weighted averaged based on parameter tau periodically to help ensure the ability to learn but also to make sure the new model doesn't try to learn too fast. 

#### Replay buffer
The replay buffer allows the model to learn from previous episodes that it records. This allows the model to lose any state action coherence that it builds up that isn't intrinsic to the underlying problem while building this up where it still hasn't yet. 

#### Q-function design
The Q-function is a sequential feed-forward neural network the input state vector and return a logits vector corresponding to an action in the action space. 
The model takes in the dimensionality of the state space and action space, being 37 and 4 respectively, and takes in a list of the hidden layers number of internal nodes. As default, we have this set as 128, 64, 32. We also include a ReLU activation function to each hidden layer. Thus, the resulting network that we use to model the Q-function is: 32 node input -> 128 node hidden layer with ReLU -> 64 node hidden layer with ReLU -> 32 node hidden layer with ReLU -> 4 node output which are logits.

#### Hyperparameters
The following hyperparameters were used along with their respective values and usage
* BUFFER_SIZE = int(1e5)     # replay buffer size
* BATCH_SIZE = 64            # minibatch size
* GAMMA = 0.99               # discount factor
* TAU = 1e-3                 # to adjust soft update of target parameters
* LR = 5e-4                  # learning rate
* UPDATE_EVERY = 4           # how often to update the network
* fc_units = \[128, 64, 32\] # number of units in the respective hidden layers of the Q-function
* n_episodes = 2000          # maximum number of training episodes
* max_t = 1000               # maximum number of timesteps per episode
* eps_start = 1.0            # starting value of epsilon, for epsilon-greedy action selection
* eps_end = 0.01             # minimum value of epsilon
* eps_decay = 0.995          # multiplicative factor (per episode) for decreasing epsilon

### Training process

#### Target rewards
We trained our network to get a mean reward of 16.0 over the last 100 episodes to complete training which it did in 855 episodes. The graph of the rewards vs episode is: 

![Trained model][image1]

#### Test rewards
During a test of the trained model, we received a reward of 15.0 from the enviroment. 

### Future work
In the future, we could improve the model by replacing the Q-function with a deeper network or use more advanced layers like transformers rather than ReLU to model the enviroment and our interactions with it. 

Beyond that, we could also input to the Q-function a short history of state action pairs that it could use to more intelligently understand the enviroment. Extending this idea, the model could instead be a RNN and have the current state update the model and have it train over sequences of states and actions in this manner. 