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
     np.int64(0),
     np.int64(0),
     np.int64(0),
     np.int64(0),
     np.int64(0),
     np.int64(1),
     np.int64(0),
     np.int64(1),
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




    28.47


