# Learning through interaction

Code is optimized.


```python
import numpy as np
```


```python
def simulate(epoch, state_space=None, action_space=None, n=250, seed=100):
    np.random.seed(seed)
    return np.array([epoch() for _ in range(n)]).mean()
```

## Tossing a biased coin


```python
state_space = [1, 0]
action_space = [1, 0]


def epoch():
    total = 0
    for i in range(100):
        action = np.random.choice(action_space)
        state = np.random.choice(state_space)
        if action == state:
            total += 1
    return total


simulate(epoch, state_space, action_space)
```




    np.float64(50.236)



When we guess it head half of the time, we will get 50% chance of guessing it right.


```python
is_head = 0.5 * 0.5 * 100  # prob is head x prob of guessing head x times
is_tail = 0.5 * 0.5 * 100  # prob is tail x prob of guessing tail x times
is_head + is_tail
```




    50.0




```python
state_space = [1, 1, 1, 1, 0]  # 80% head
action_space = [1, 0]
action_weights = [0.5, 0.5]


def epoch():
    total = 0
    for i in range(100):
        action = np.random.choice(action_space, p=action_weights)
        state = np.random.choice(state_space)
        if action == state:
            total += 1
    return total


simulate(epoch, state_space, action_space)
```




    np.float64(50.26)



Even if the coin is biased, but we still take equal actions (guessing half head and half tail), the max reward is just 50% of total rewards.


```python
state_space = [1, 1, 1, 1, 0]  # 80% head
action_space = [0, 1]
action_weights = [0.2, 0.8]


simulate(epoch, state_space, action_space)
```




    np.float64(67.628)



When we start adding more weights to our decision by picking heads 80% of the time, our reward increases.
The theoretical reward is $0.8^2 + 0.2^2 = 0.68$


```python
is_head = 0.8 * 0.8 * 100  # prob is head x prob of guessing head x times
is_tail = 0.2 * 0.2 * 100  # prob is tail x prob of guessing tail x times
is_head + is_tail
```




    68.00000000000001




```python
state_space = [1, 1, 1, 1, 0]  # 80% head
action_space = [0, 1]
action_weights = [0, 1]  # 100% head


simulate(epoch, state_space, action_space)
```




    np.float64(79.7)




```python
is_head = 0.8 * 1.0 * 100  # prob is head x prob of guessing head x times
is_tail = 0.2 * 0.0 * 100  # prob is tail x prob of guessing tail x times
is_head + is_tail
```




    80.0



If we already know the coin is biased, we can simply choose head 100% of the time, and end up with the maximum reward of 80$.

## Rolling a biased die


```python
# The probability of the outcome 4 is 5 times as likely as other number.
state_space = [1, 2, 3, 4, 4, 4, 4, 4, 5, 6]
action_space = [1, 2, 3, 4, 5, 6]
action_weights = [1, 1, 1, 1, 1, 1]


def epoch():
    total = 0
    for _ in range(600):
        action = np.random.choice(action_space)
        state = np.random.choice(state_space)
        if action == state:
            total += 1
    return total


simulate(epoch, state_space, action_space)
```




    np.float64(99.8)




```python
# probablity of getting number n x probability of guessing number n
sum(
    [
        0.1 * 1 / 6,
        0.1 * 1 / 6,
        0.1 * 1 / 6,
        0.5 * 1 / 6,
        0.1 * 1 / 6,
        0.1 * 1 / 6,
    ]
) * 600
```




    100.0




```python
# The probability of the outcome 4 is 5 times as likely as other number.
state_space = [1, 2, 3, 4, 4, 4, 4, 4, 5, 6]
action_space = [1, 2, 3, 4, 5, 6]
# Choose number 4 five times more than the rest.
action_weights = [
    0.1,
    0.1,
    0.1,
    0.5,
    0.1,
    0.1,
]


simulate(epoch, state_space, action_space)
```




    np.float64(99.8)




```python
# The probability of the outcome 4 is 5 times as likely as other number.
state_space = [1, 2, 3, 4, 4, 4, 4, 4, 5, 6]
action_space = [1, 2, 3, 4, 5, 6]
action_weights = [0, 0, 0, 1, 0, 0]  # Choose number 4 every time.


simulate(epoch, state_space, action_space)
```




    np.float64(99.8)


