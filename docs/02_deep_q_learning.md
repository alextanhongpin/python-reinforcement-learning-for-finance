```python
import gymnasium as gym
```


```python
env = gym.make("CartPole-v1")
env.action_space
```




    Discrete(2)




```python
[env.action_space.sample() for _ in range(10)]
```




    [np.int64(0),
     np.int64(1),
     np.int64(1),
     np.int64(0),
     np.int64(0),
     np.int64(0),
     np.int64(0),
     np.int64(0),
     np.int64(0),
     np.int64(0)]




```python
env.observation_space
```




    Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32)




```python
env.observation_space.shape
```




    (4,)




```python
env.reset(seed=100)
```




    (array([ 0.03349816,  0.0096554 , -0.02111368, -0.04570484], dtype=float32),
     {})




```python
env.step(0)
```




    (array([ 0.03369127, -0.18515752, -0.02202777,  0.24024247], dtype=float32),
     1.0,
     False,
     False,
     {})



## A Random Agent


```python
class RandomAgent:
    def __init__(self):
        self.env = gym.make("CartPole-v1")

    def play(self, episodes=1):
        self.total_rewards = []
        for e in range(episodes):
            self.env.reset()

            for step in range(1, 100):
                action = self.env.action_space.sample()
                state, reward, done, trunc, info = self.env.step(action)
                if done:
                    self.total_rewards.append(step)
                    break
```


```python
ra = RandomAgent()
ra.play(15)
average_reward = round(sum(ra.total_rewards) / len(ra.total_rewards), 2)
average_reward
```




    24.6




```python
import random
from collections import deque

import numpy as np
import torch
from torch import nn
```


```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")
```

    Using mps device



```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
model
```




    NeuralNetwork(
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=4, out_features=24, bias=True)
        (1): ReLU()
        (2): Linear(in_features=24, out_features=24, bias=True)
        (3): ReLU()
        (4): Linear(in_features=24, out_features=2, bias=True)
      )
    )




```python
class DQLAgent:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=2000)
        self.batch_size = 12
        self.gamma = 0.9
        self.max_reward = 0
        self.model = NeuralNetwork().to(device)
        self.env = gym.make("CartPole-v1")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        return torch.argmax(self.model(state).detach(), dim=1).item()

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            if not done:
                expect = torch.max(self.model(next_state).detach(), dim=1)[0]
                reward += self.gamma * expect
            target = self.model(state)
            loss = self.criterion(target[:, action], torch.Tensor([reward]).to(device))
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            observation, _ = self.env.reset()
            state = torch.Tensor(observation).view(-1, 4).to(device)
            for f in range(1, 5000):
                action = self.act(state)
                observation, reward, done, trunc, _info = self.env.step(action)
                next_state = torch.Tensor(observation).view(-1, 4).to(device)
                self.memory.append([state, action, next_state, reward, done])
                state = next_state

                if done or trunc:
                    self.max_reward = max(self.max_reward, f)
                    templ = f"episode={e:4d} | total_reward={f:4d} | max={self.max_reward:4d}"
                    if e % 100 == 0:
                        print(templ, end="\r")
                    break
            if len(self.memory) > self.batch_size:
                self.replay()

    def test(self, episodes):
        for e in range(1, episodes + 1):
            observation, _ = self.env.reset()
            for f in range(1, 5000):
                state = torch.Tensor(observation).view(-1, 4).to(device)
                with torch.no_grad():
                    logits = self.model(state)
                action = torch.argmax(logits, dim=1)[0].item()
                observation, reward, done, trunc, _info = self.env.step(action)
                if done or trunc:
                    print(f, end=" ")
                    break
```


```python
agent = DQLAgent()
agent.learn(2000)
```

    episode=2000 | total_reward= 150 | max= 500


```python
%time agent.test(10)
```

    198 97 202 336 118 178 190 142 186 160 CPU times: user 319 ms, sys: 196 ms, total: 515 ms
    Wall time: 700 ms

