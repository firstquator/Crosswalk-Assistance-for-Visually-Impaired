import time
import threading
from multiprocessing import Process, Queue
from gtts import gTTS
from playsound import playsound
import pygame


class Voice:
    t = None
    t2 = None

    def __init__(self):
        self.set_thread("../sound/left.mp3")

    def voice(self, file):
        music_file = file  # mp3 or mid file

        freq = 16000  # sampling rate, 44100(CD), 16000(Naver TTS), 24000(google TTS)
        bitsize = -16  # signed 16 bit. support 8,-8,16,-16
        channels = 1  # 1 is mono, 2 is stereo
        buffer = 2048  # number of samples (experiment to get right sound)

        pygame.mixer.init(freq, bitsize, channels, buffer)
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()

        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(30)

    def set_thread(self, file):
        self.t = threading.Thread(target=self.voice, kwargs={"file": file})

    def run(self):
        self.t.start()


v = Voice()
v.run()
for i in range(5000):
    print(i)
    if i == 1100:
        v.set_thread("./sound/right.mp3")
        v.run()
