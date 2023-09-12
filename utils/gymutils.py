import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
import gymnasium as gym

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0]/20, im.size[1]/18),
                f'Episode: {episode_num+1}', fill=text_color)

    return im


def save_random_agent_gif(env: gym.Env):
    frames = []
    for i in range(5):
        state = env.reset()
        while True:
            action = env.action_space.sample()

            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=i))
            state, reward, truncated, terminal, info = env.step(action)
            if truncated or terminal:
                break

    env.close()

    imageio.mimwrite(os.path.join(
        './videos/', 'random_agent.gif'), frames, fps=60)


env = gym.make('CartPole-v1', render_mode='rgb_array')
save_random_agent_gif(env)
