# Report

## Learning Algorithm

To solve Continuous Control project, I used as a start point the code from the Udaciy deep reinforcement learning repository that implements Deep Deterministic Policy Gradients (DDPG) algorithm.
[https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) 

In *Fixed Target* algorithm improvement we are using a separate network with a fixed parameter w_ for estimating the TD target and at *UPDATE_EVERY* step, we copy the parameters from our DQN network to update the target network.

In *Experience Replay* we are storing agent’s experiences, and then randomly drawing batches of them to train the network. By keeping the experiences we draw random, we prevent the network from only learning from immediate experiences, and allow it to learn from a more varied array of past experiences. The Experience Replay buffer stores a fixed number of recent memories, and as new ones come in, old ones are removed.

Original code is in first step customized to use * Unity ML-Agents * environment. In this project I am using Reacher environment with 20 agens. The main part of the program is in following files

```Python
ddpg_agent.py
model.py
Continuous_Control.ipynb
```

*ddpg_agent.py* code implements an environment-aware agent, while in *model.py* is a neural network models of Actor and Critic.
*Continuous_Control.ipynb* sadrzi kod koji trenira agenta za resavanje problema i prikazuje rezultate.

#### DDPG with Actor-Critic target network and Experience Replay
The model of the neural network used is shown in the picture:

![Network model](./Images/dqn+ft+rb.png  "Network model")
Za Actor koristio sam mrezu sa dva lejera

The size of the hidden layers are:
FC1 size = 256
FC2 size = 256
The input parameter is the state of the environment (size 33), and the output is  action (size 4).
Za Critic sam takodje koristi mresu sa dva lejera dimenzija
FCS1 size = 256
FC2 size + action_size= 256 + 4
Input parmeters are state in first layer and action in second layer, and output is Q  value.
I used batch normalization on the state input and all layers of the Actor network and all layers of the Critic network prior to the action input as in paper https://arxiv.org/pdf/1509.02971.pdf
Aktivaciona funkcija je leakyReLU u svim lejerima osim u poslednjem lejeru Actor gde sm koristio tanh jer su vednosti akcije u rasponu [-1,1], i poslednjem lejeru Critic nisam koristio aktivacionu funkciju jer izlaz pretstavla Q value.

The hyper parameters are: 

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
WARMUP_TIME = 10000
PLAY_TIME = 45
WORK_TIME = 10
play_time_decay = 0.999

Hyperparameters koje sam uveo su WARMUP_TIME koji predstavlja vreme popunjavanja Buffera pre nego sto agent krene da uci. PLAY_TIME pretsavlja koliko dugo ce agent prikupljati uzorke pre nego sto krene da uci, WORK_TIME pretstavlja broj ciklusa ucenja kada agent krene da uci,. play_time_decay predstavlja faktor koim opada PlaY_Time slicno kao GAMMA.

Kod u ddpg_agent.py je promenjen da podrzava input 20 agenata. Ideja je bila prilikom ovih promena da se zadrzi model sa jednim actor i jednim critic a da se buffer puniiskustvom 20 agenata. Iskustva se u buffer dodaju na uniormly slucajan tako da je verovatnoca da ce iskustvo svakog od agena bit i dodato u buffer je 50%, na taj nacin cinimo buffer more random i razijamo korelacije jos vise( nesto slicno kao dropout layer).
Druga znacajna promena je da se uvodi vreme uzimanja uzoraka PLAY_TIME u kontinuitetu i vreme ucenja WORK_TIME  u kontinuitetu, cime se utice na odnos exploration/expliatation
Treca promena je uvodjenje
torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
u Critic.
Parametri OUNoise su blago povecani u odnosu na bazni kod i iznose theta = .25 i sigma = .3


By changing the hyper parameter GAMMA, I did not get a significant improvement in the results, decreasing gives worse results, and the increase does not make a significant improvement.
Changing the TAU also did not give me a significant improvement.
By increasing BATCH_SIZE = 128, better results are obtained, as is the reduction of learning rate LR = 5e-5.
Other hyper parameters have the same values as in the original project.

Results after 1000 episodes are:

	Episode 100	Average Score: 0.98
	Episode 200	Average Score: 3.68
	Episode 300	Average Score: 8.53
	Episode 400	Average Score: 11.56
	Episode 468	Average Score: 13.02
	Environment solved in 368 episodes!	Average Score: 13.02
	Episode 500	Average Score: 13.71
	Episode 600	Average Score: 14.71
	Episode 700	Average Score: 15.21
	Episode 800	Average Score: 15.99
	Episode 900	Average Score: 16.27
	Episode 1000	Average Score: 16.32

![Plot of rewards](./Images/dqn+ft+rb-results.png  "Plot of rewards")
Agent solves the problem after 368 episodes.

Version 2 DQN with epsilon learning "likes" smaller batches and larger learninig rates then version 1 DQN. 
In the case of using the Dueling DQN improvement, the output of the network is replaced by the model in the image:
![Dueling network model](./Images/duelin_dqn.png  "Dueling network model")

## Conclusion

The best results are obtained with the model
DQN + Fixed Target + Replay Buffer + Epsilon learning and the sizes of the hidden leyers FC1 = 137 and FC2 = 64. With this network configuration and hyper parameters, the agent solves the problem for less than 200 episodes.

## Ideas for Future Work 	
The next step that would, I belive, further improve the algorithm in each of the tested variants is the implementation of the Prioritized Experience Replay algorithm [https://arxiv.org/pdf/1511.05952](https://arxiv.org/pdf/1511.05952), as well as experiments with the size and number of hidden layers.

## References

1. https://arxiv.org/abs/1509.02971
2.  [https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df) 