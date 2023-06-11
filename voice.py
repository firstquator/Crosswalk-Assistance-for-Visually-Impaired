import time
import threading
from multiprocessing import Process, Queue
from gtts import gTTS
from playsound import playsound
import pygame


class Voice:
    t = None

    def __init__(self):
        self.set_thread()

    def voice(self):
        music_file = "sample.mp3"  # mp3 or mid file

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

    def set_thread(self):
        self.t = threading.Thread(target=self.voice)

    def run(self):
        self.t.start()


v = Voice()

for i in range(1000):
    v.run()
