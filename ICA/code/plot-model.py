import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

combinedMusic = 'combinedMusic.wav'
path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/CS191ML/ICA/resources/'
# music 1 is ed-sheeran
# music 2 is ben&ben 
music_1 = 'ben.wav'
music_2 = 'ed.wav'
audio_1_fig = 'audio_1_wave.png'
audio_2_fig = 'audio_2_wave.png'


if __name__ == "__main__":
    for dirName, subdirList, fileList in os.walk(path):
        print('Found directory: %s' % dirName)
        if(os.path.exists(os.path.join(dirName, music_1)) and os.path.exists(os.path.join(dirName, music_2))):
            
            mix_1_wave = wave.open(os.path.join(dirName, music_1),'r')
            mix_2_wave = wave.open(os.path.join(dirName, music_2),'r')


            #Extract Raw Audio from Wav File
            audio_signal_1_raw = mix_1_wave.readframes(-1)
            audio_signal_1 = np.fromstring(audio_signal_1_raw, 'Int16')
            audio_signal_2_raw = mix_2_wave.readframes(-1)
            audio_signal_2 = np.fromstring(audio_signal_2_raw, 'Int16')

            fs = mix_1_wave.getframerate()
            timing = np.linspace(0, len(audio_signal_1)/fs, num=len(audio_signal_1))
            fs2 = mix_2_wave.getframerate()
            timing2 = np.linspace(0, len(audio_signal_2)/fs2, num=len(audio_signal_2))

            # Save the figures in a png file
            plt.figure(figsize=(12,2))
            plt.title('Recording 1')
            plt.plot(timing,audio_signal_1, c="#3ABFE7")
            plt.ylim(-35000, 35000)
            # plt.show()
            plt.savefig(os.path.join(dirName, audio_1_fig))

            plt.figure(figsize=(12,2))
            plt.title('Recording 2')
            plt.plot(timing2,audio_signal_2, c="#df8efd")
            plt.ylim(-35000, 35000)
            #plt.show()
            plt.savefig(os.path.join(dirName, audio_2_fig))
 

            del mix_1_wave,mix_2_wave,audio_signal_1_raw,audio_signal_2_raw,audio_signal_1,audio_signal_2,fs,fs2,timing,timing2
            plt.close("all")
            gc.collect()
            

            
    
