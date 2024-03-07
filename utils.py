import base64
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import io
from IPython.display import HTML
from IPython import display as ipythondisplay
import glob

def create_video_random(filename, env, fps=30):
    
    image_size = (600, 400)
    
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'H264'), fps, image_size)

    done = False
    state = env.reset()[0]
    frame = env.render()
    video.write(frame)
    while not done:
        state = np.expand_dims(state, axis=0)
        action = np.random.randint(2)
        state, _, done, _, _ = env.step(action)
        frame = env.render()
        video.write(frame)
    video.release()

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")

