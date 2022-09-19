# TAURA V3S SOFT ACTOR CRITIC

## What is a Soft Actor Critic?
Is a Deep Reinforcement Learning composed by two main neural networks. One of those is
the agent, which will learn to chose on actions given a certain state, considering the probability of the state
that will result from that action, and the possible states that can occur giving this new state. The other DNN
is the critic, which will specialize in judging the agent on its actions (calculating the reward function), given
the probability of the state that will be caused by tha t action, and the possible states that will come after.
There will also be a third network, used for controlling given a state, which actions are possible (valuable).

## What do we want?
Get robots to learn with stability in a continuous space environment

##Why is SAC a good option for our V3S robots?
It can learn way faster than the other conventional
reinforcement learning deep neural networks, and since we could not find a way to accelerate the
simulator, this fits perfectly. Other than that, actor critic networks seem to achieve very good
results, since the cost function used is complemented with a high entropy (stochasticity), which gives us a better
chance to reach the global optima (encourages exploration), for example. It maximizes the reward over time

## How does SAC differ from its pairs?
Other than its max-entropy model, there is a very important difference in the
SAC network. The output from the other models will be direct, while SAC outputs an average and a standard deviation,
which will form a normal distribution, which will be sampled to choose the action.
