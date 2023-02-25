#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from gym import Env
import math
import numpy as np

from typing import Optional, Union
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from gym import logger, spaces


# In[2]:


def sumTon(a, d, n):
    return ((n/2)*(2*a+((n-1)*d)))


# In[3]:


metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

class cartpole(gym.Env):
    def __init__(self):
        self.n = 2
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        
        self.polemass_length = self.masspole * self.length
        
        self.force_mag = 10.0
        
        self.tau = 0.02  # seconds between state updates

        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        
        self.x_threshold = 2.4
        
        high = np.array(
            [
                np.arange(4.8, sumTon(4.8, 5.8, self.n), 5.8),
                np.full(self.n, np.finfo(np.float32).max),
                np.full(self.n, self.theta_threshold_radians * 2),
                np.full(self.n, np.finfo(np.float32).max),
            ],
            dtype=object,
        ).flatten()

        low = np.array(
            [
                np.arange(-4.8, sumTon(-4.8, -5.8, self.n), -5.8),
                np.full(self.n, np.finfo(np.float32).max),
                np.full(self.n, self.theta_threshold_radians * 2),
                np.full(self.n, np.finfo(np.float32).max),
            ],
            dtype=object,
        ).flatten()
        

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        
        
        self.action_space = MultiDiscrete(np.full((self.n), 2))
        
        self.observation_space = spaces.Box(low, high, shape=(4, self.n), dtype=object)

        self.render_mode = "human"

        self.screen_width = 6000 * self.n
        self.screen_height = 400 * self.n
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4, self.n))
        for i in range(0, self.n):
            self.state[0][i] = self.state[0][i] + 5.8

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        x = self.state[0]
        x_dot = self.state[1]
        theta = self.state[2]
        theta_dot = self.state[3]

        force = []
        for i in range(0, self.n):
            force.append(self.force_mag if action[i] == 1 else -self.force_mag)
#             force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = np.divide(
            np.add(force, (self.polemass_length * np.multiply(np.square(theta_dot), sintheta))),
            self.total_mass
        )
        thetaacc = np.divide(
                np.subtract((self.gravity * sintheta),
                 np.multiply(costheta, temp)
                ),
            (
            self.length * (4.0 / 3.0 - (self.masspole * (np.square(costheta) / self.total_mass)
                                       )
                          )
            )
        )

        xacc = np.subtract(temp, (self.polemass_length * np.multiply(thetaacc, costheta))) / self.total_mass

        if self.kinematics_integrator == "euler":
            x = np.add(x, (self.tau * x_dot))
            x_dot = np.add(x_dot, (self.tau * xacc))
            theta = np.add(theta, (self.tau * theta_dot))
            theta_dot = np.add(theta_dot, (self.tau * thetaacc))

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        reward = []
        for i in range(0, self.n):
            terminated = bool (x[i] < -self.x_threshold or x[i] > self.x_threshold or theta[i] < -self.theta_threshold_radians or theta[i] > self.theta_threshold_radians)
            if not terminated:
                reward.append(1.0)
            elif self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
                reward.append(1.0)
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1
                reward.append(0.0)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state), reward, terminated, False, {}
    
    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4, self.n))
        for i in range(0, self.n):
            self.state[0][i] = self.state[0][i] + 5.8
        return self.state
    
    def render(self):
        pass
    
    def close(self):
        pass


# In[4]:


env = cartpole()


# In[ ]:


episodes = 5
for episodes in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
#         env.render()
        action = env.action_space.sample()
        print(action)
        print(env.step(action))
#         obs, reward, done, info = env.step(action)
#         score += reward
    print("Episode:{}, Score:{}".format(episode, score))
env.close()


# In[ ]:




