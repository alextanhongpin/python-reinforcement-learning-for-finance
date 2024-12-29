# Learning through Interaction

## Tossing a biased coin


```python
import numpy as np

rng = np.random.default_rng(seed=100)
```


```python
def simulate(epoch, state_space=None, action_space=None, n=250, seed=100):
    rng = np.random.default_rng(seed=seed)
    return np.array([epoch() for _ in range(n)]).mean()
```


```python
def epoch():
    total = 0
    for _ in range(100):
        action = rng.choice(action_space)
        state = rng.choice(state_space)
        if action == state:
            total += 1
    return total
```


```python
state_space = [1, 0]
action_space = [1, 0]

simulate(epoch, state_space, action_space)
```




    np.float64(49.968)



If the coin is unbiased, we will end up with 50% chances of getting half-head and half-tail.


```python
state_space = [1, 1, 1, 1, 0]  # 80% head
action_space = [1, 0]

simulate(epoch, state_space, action_space)
```




    np.float64(49.924)



Even if the coin is biased, if we pick head or tail 50% of the time, we still get half correct. The calculation is as follow:

```
percentage_head = 0.8
percentage_tail = 0.2
probability_correct = 0.5
times = 100
total = probability_correct * (percentage_head + percentage_tail) * times
total = 50
```


```python
state_space = [1, 1, 1, 1, 0]  # 80% head
action_space = [1, 0]


def epoch():
    total = 0
    for _ in range(100):
        action = rng.choice(action_space)
        state = rng.choice(state_space)
        if action == state:
            total += 1
        action_space.append(state)
    return total


simulate(epoch, state_space, action_space)
```




    np.float64(68.492)



This time, we got more than 50% right. This is calculated as follow:

```
percentage_head = 0.8
percentage_tail = 0.2
probability_head_correct = 0.8
probability_tail_correct = 0.2
times = 100
total = (percentage_head * probability_head_correct + percentage_tail * probability_tail_correct) * times
total = 68
```


```python
(0.8**2 + 0.2**2) * 100
```




    68.00000000000001




```python
from collections import Counter

state_space = [1, 1, 1, 1, 0]  # 80% head
action_space = [1, 0]


def epoch():
    total = 0
    for _ in range(100):
        action = Counter(action_space).most_common()[0][0]
        state = rng.choice(state_space)
        if action == state:
            total += 1
        action_space.append(state)
    return total


simulate(epoch, state_space, action_space)
```




    np.float64(80.068)



In the last example, assuming we already know what is the actual probability of the biased coin, we can just pick head 100% of the time.

```
percentage_head = 1.0
percentage_tail = 0.0
probability_head_correct = 0.8
probability_tail_correct = 0.2
times = 100
total = (percentage_head * probability_head_correct + percentage_tail * probability_tail_correct) * times
total = (percentage_head * probability_head_correct + 0) * times
total = 80
```

## Rolling a biased die


```python
# The probability of the outcome 4 is 5 times as likely as other number.
state_space = [1, 2, 3, 4, 4, 4, 4, 4, 5, 6]
action_space = [1, 2, 3, 4, 5, 6]


def epoch():
    total = 0
    for _ in range(600):
        action = rng.choice(action_space)
        state = rng.choice(state_space)
        if action == state:
            total += 1
    return total


simulate(epoch, state_space, action_space)
```




    np.float64(100.452)




```python
prob_1 = 0.1 * 1 / 6
prob_2 = 0.1 * 1 / 6
prob_3 = 0.1 * 1 / 6
prob_4 = 0.5 * 1 / 6
prob_5 = 0.1 * 1 / 6
prob_6 = 0.1 * 1 / 6
(prob_1 + prob_2 + prob_3 + prob_4 + prob_5 + prob_6) * 600
```




    100.0




```python
state_space = [1, 2, 3, 4, 4, 4, 4, 4, 5, 6]
action_space = [1, 2, 3, 4, 5, 6]


def epoch():
    total = 0
    for _ in range(600):
        action = rng.choice(action_space)
        state = rng.choice(state_space)
        if action == state:
            total += 1
        action_space.append(state)
    return total


# This is taking too long
# simulate(epoch, state_space, action_space)
```


```python
prob_1 = 0.1 * 1 / 10
prob_2 = 0.1 * 1 / 10
prob_3 = 0.1 * 1 / 10
prob_4 = 0.5 * 5 / 10
prob_5 = 0.1 * 1 / 10
prob_6 = 0.1 * 1 / 10
(prob_1 + prob_2 + prob_3 + prob_4 + prob_5 + prob_6) * 600
```




    180.00000000000003



If the actual probablity of the die is known, we can just guess the number 4 100% of the time, which will yield 50% of the total reward.


```python
prob_1 = 0.0 * 1 / 10
prob_2 = 0.0 * 1 / 10
prob_3 = 0.0 * 1 / 10
prob_4 = 1.0 * 5 / 10
prob_5 = 0.0 * 1 / 10
prob_6 = 0.0 * 1 / 10
(prob_1 + prob_2 + prob_3 + prob_4 + prob_5 + prob_6) * 600
```




    300.0


