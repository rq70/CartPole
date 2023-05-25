# CartPole
In this tutorial, we aim to provide readers with an overview of RL principles as well as sample code in Python by introducing the OpenAI Gym library. We begin by developing intuition about what is considered an RL problem and introduce formal definitions as well as key terms used to describe and model an RL program.
The Cartpole project is one of the famous problems in the field of reinforcement learning. In this problem, a simple cart system with a pole is attached to the ground plane. The goal is to keep the pole balanced on the card for the maximum amount of time.
To solve this problem, the agent must control the card by moving the card in such a way that the balance of the pole is maintained. To replace the agent with a reinforcement learning algorithm, the learning process is performed in such a way that the agent randomly performs an operation and after evaluating the result, updates a set of rules to achieve the best performance in the next steps. In this model, the number of steps to learn and perform operations is determined randomly.
This problem is one of the classical problems in the field of reinforcement learning, and many reinforcement learning algorithms, including Q-learning and Deep Q-Networks (DQN), have been used to solve it. Model training using deep learning methods and advanced reinforcement learning algorithms can help improve the agent's performance in this problem.
The purpose of this report is to introduce and explain the Q-learning algorithm for solving reinforcement learning problems using a simple example.

![download](https://github.com/rq70/CartPole/assets/68390542/ea69c634-3323-4fb6-8a04-d3cc79e7ec44)

# Game rules:
The goal of this example/game is for the operator to keep the bar standing on the cart as long as possible without losing balance by moving the cart left or right. The more steps the agent can balance for them, the better the number of steps actually represents a reward for playing the game well. If the angle of the pole tilts more than 15 degrees to the left or right or the cart moves more than 2.4 units from the center, the agent loses the game. The learning environment is the play area and the position of the cart as well as the pole. The current position of the pole and cart indicates the current state of the game. Given the current state of the game, the agent must decide whether to move the cart to the right or left at a given time step.

# Reinforcement learning:
![download1](https://github.com/rq70/CartPole/assets/68390542/4bae71e2-e42e-4b0f-bdf3-e8992da2aeb4)

In the picture above, you can see the main parts of the reinforcement learning algorithm of this game.
 A : Move left or right
 S : in time (t), rod angle and cart position in the environment or game area.
 R : reward for the action performed.
Each action is associated with a reward, which is represented by Rt+1. In this example, the reward for each move is +1 fixed, which is given to the agent when it remains balanced.
Each independent attempt to play the game, from the beginning to the end, is called an episode, and each episode consists of several time steps.
This algorithm needs to pass +100 episodes to learn the effective action. As humans need such training.

# Q-Learning:
We need to use a specific learning algorithm so that the agent can learn from experiences and perform the right action at the right time.

![download](https://github.com/rq70/CartPole/assets/68390542/5e60c25c-d1c4-4db9-9dbe-5fbb3412f563)

When the agent first starts searching, the values in the above table are unknown. The only thing the agent can do is to randomly roam around the environment to search the environment and this creates a better understanding of where to go, which we call the exploration phase.
The goal of Q-Learning is to continuously exit the discovery phase and record the exact values (as possible) in the table.
Most of the time, the agent chooses an action that maximizes its reward, but sometimes it acts randomly to ensure discovery.


# def epsilon_greedy_policy(state, env, Q_table, exploration_rate):
    if (np.random.random() < exploration_rate):
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state])

In the code above, a function is written that has four inputs, inside it there is a condition that creates random numbers between 01 and is matched with the search rate. If the condition is met, it samples a random operation. If the condition is not met, Greedy performs the action that has the highest reward given the current situation.

def get_rate(e):
    return max(0.1, min(1., 1. - np.log10((e + 1) / 25.)))
    
We use an arbitrary function to set the amount of learning and searching as you can see above. According to the current episode, it gives us the rate, and with the increase in the number of episodes, the rate tends to decrease.
This code produces a non-negative number calculated based on the input e. More precisely, log10((e + 1) / 25) is calculated, then it is put from a right arm and finally it is min 1 and max 0.1 and the final output is returned as the output number.

In fact, this code is an activation function that may be used in neural networks and machine learning. The initial function uses min and max to limit the output and uses log10 to decrease the value.

import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(1, 1000, 1000)
y = np.array([get_rate(i) for i in x])
plt.plot(x, y)
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.show()

This code allows you to create an unsigned array named x that contains 1000 numbers, with a difference of 1 in the range 1 to 1000. More precisely, the linspace function in namespace creates an array of length 1000 that starts at 1 and ends at 1000, with the crossing points between these two ends evenly distributed. That is, the distance between two consecutive numbers is equal.
This code takes a numpy array named x and calls the get_rate function for each element of the array and returns it as a new member to the y array. In other words, this code calculates a rate for each member of the x array using the get_rate function and stores its return in a new array named y.

![download (1)](https://github.com/rq70/CartPole/assets/68390542/e9fef992-96cc-4805-9b78-2d3114ecc658)


# Algorithm:
The mathematical form of Q-Learning is as follows, which updates the Q-Table.

![image](https://github.com/rq70/CartPole/assets/68390542/7a4a64e5-7583-49ca-8515-6fcdbe98455a)

def update_q(Q_table, state, action, reward, new_state, alpha, gamma):
    Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state][action])
    return Q_table

# OpenAI Gym Environment:

!pip install gym >/dev/null
import gym
env = gym.make('CartPole-v0')
for e in range(1):
    state = env.reset()
    for step in range(200):
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        state = new_state
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break
env.close()

# Update the package references repository
!apt update -y >/dev/null
# Install packages for visualisation
!apt-get install xvfb python3-opengl ffmpeg -y >/dev/null
!pip install pyvirtualdisplay imageio >/dev/null

# Do all the necessary imports, based on the installations we just did
import gym, math, imageio, os, time
import numpy as np
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display

# Fix the random number seed
np.random.seed(42)
fake_display = False

# Create a fake display which will help us to render the animations
if fake_display is False:
    display = Display(visible=0, size=(700, 450))
    display.start()
    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    fake_display=True
def Q_learning(env, num_episodes):
    # Discount factor gamma represents how much does the agent value future rewards as opposed to immediate rewards
    gamma = 0.98

    # (1, 1, 6, 12) represents the discretisation buckets
    # Initialise the q-table as full of zeros at the start
    Q_table = np.zeros((1, 1, 6, 12) + (env.action_space.n,))

    # Create a list to store the accumulated reward per each episode
    total_reward = []
    for e in range(num_episodes):
        # Reset the environment for a new episode, get the default state S_0
        state = env.reset()
        state = discretize_state(state, env)

        # Adjust the alpha and the exploration rate, it is a coincidence they are the same
        alpha = exploration_rate = get_rate(e)
        
        # Initialize the current episode reward to 0 
        episode_reward = 0
        done = False
        while done is False:
            # Choose the action A_{t} based on the policy
            action = epsilon_greedy_policy(state, env, Q_table, exploration_rate)

            # Get the new state (S_{t+1}), reward (R_{t+1}), end signal
            new_state, reward, done, _ = env.step(action)
            new_state = discretize_state(new_state, env)

            # Update Q-table via update_q(Q_table, S_{t}, A_{t}, R_{t+1}, S_{t+1}, alpha, gamma) 
            Q_table = update_q(Q_table, state, action, reward, new_state, alpha, gamma)

            # Update the state S_{t} = S_{t+1}
            state = new_state
            
            # Accumulate the reward
            episode_reward += reward
        
        total_reward.append(episode_reward)
    print('Finished training!')
    return Q_table, total_reward

Note that the pole angle or cart coordinate is continuous. To explain Q-learning in its default form using a table, we need to discretize the situation with respect to the position of the pole and the cart. In order to do this, we introduce the following function that converts the continuous space into discrete bins based on predefined bounds.

def discretize_state(state, env, buckets=(1, 1, 6, 12)):
    # The upper and the lower bounds for the discretization
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]

    # state is the native state representations produced by env
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
   
    # state_ is discretized state representation used for Q-table later
    state_ = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    state_ = [min(buckets[i] - 1, max(0, state_[i])) for i in range(len(state))]
    
    return tuple(state_)

balance the pole!

# OpenAI Gym builds the environment for us inclusing all the rules, dynamics etc.
env = gym.make('CartPole-v0')
# How long do we want the agent to explore and learn
num_episodes = 1000        

# Let us use Q-learning to learn the game!
Q_table, total_reward = Q_learning(env, num_episodes)

plt.plot(range(num_episodes), total_reward)
plt.xlabel('Episode')
plt.ylabel('Training cumulative reward')
plt.show()

![image](https://github.com/rq70/CartPole/assets/68390542/b57835ca-d6a4-4ac3-80c5-85286d30c8b7)

Final Test and Visualisation
# Initialise the reward
episode_reward = 0
# Count how many times the agent went right and how many times it went left
right = 0
left = 0

# Initialise empty buffer for the images that will be stiched to a gif
# Create a temp directory
filenames = []
try:
    os.mkdir("./temp")
except:
    pass

# Test the trained agent in a completely frest start
state = env.reset()
state = discretize_state(state, env)

# Run for maximum of 200 steps which is the limit of the game
for step in range(200):
    # Plot the previous state and save it as an image that 
    # will be later patched together sa a .gif
    img = plt.imshow(env.render(mode='rgb_array'))
    plt.title("Step: {}".format(step))
    plt.axis('off')
    plt.savefig("./temp/{}.png".format(step))
    plt.close()
    filenames.append("./temp/{}.png".format(step))
    
    # Here we set the exploration rate to 0.0 as we want to avoid any random exploration
    action = epsilon_greedy_policy(state, env, Q_table, exploration_rate=0.0)
    right+=1 if action == 1 else 0
    left+=1 if action == 0 else 0
    new_state, reward, done, _ = env.step(action)
    new_state = discretize_state(new_state, env)
    state = new_state
    episode_reward += reward

    # At the end of the episode print the total reward
    if done:
        print(f'Test episode finished at step {step+1} with a total reward of: {episode_reward}')
        print(f'We moved {right} times right and {left} times left')
        break        
# Stitch the images together to produce a .gif
with imageio.get_writer('test.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
# Cleanup the images for the next run
for f in filenames:
    os.remove(f)

# Close the environment
env.close()

![image](https://github.com/rq70/CartPole/assets/68390542/f00ea74d-a323-4803-a578-cf6f9f271afa)

![download (1)](https://github.com/rq70/CartPole/assets/68390542/fec3160c-0762-495f-8101-02ebe945d91b)

