import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/CS191ML/ICA/resources/'
separated_1 = 'separated_audio_1.wav'
separated_2 = 'separated_audio_2.wav'
# music 1 is ed-sheeran
# music 2 is ben&ben 
music_1 = 'ben.wav'
music_2 = 'ed.wav'
sep_audio_1_fig = 'separated_audio_1.png'
sep_audio_2_fig = 'separated_audio_2.png'
plt.rcParams['agg.path.chunksize'] = 10000

if __name__ == "__main__":
    for dirName, subdirList, fileList in os.walk(path):
        print('Found directory: %s' % dirName)
        if(os.path.exists(os.path.join(dirName, separated_1)) and os.path.exists(os.path.join(dirName, separated_2))):

            sep_mix_1_wave = wave.open(os.path.join(dirName, separated_1),'r')
            sep_mix_2_wave = wave.open(os.path.join(dirName, separated_2),'r')
            sep_audio_signal_1_raw = sep_mix_1_wave.readframes(-1)
            sep_audio_signal_1 = np.fromstring(sep_audio_signal_1_raw, 'Int16')
            sep_audio_signal_2_raw = sep_mix_2_wave.readframes(-1)
            sep_audio_signal_2 = np.fromstring(sep_audio_signal_2_raw, 'Int16')

            sep_fs = sep_mix_1_wave.getframerate()
            sep_timing = np.linspace(0, len(sep_audio_signal_1)/sep_fs, num=len(sep_audio_signal_1))
            sep_fs2 = sep_mix_1_wave.getframerate()
            sep_timing2 = np.linspace(0, len(sep_audio_signal_2)/sep_fs2, num=len(sep_audio_signal_2))

            # Plot Independent Component #1
            plt.figure(figsize=(12,2))
            plt.title('Independent Component #1')
            plt.plot(sep_timing,sep_audio_signal_1, c="#f65e97")
            plt.ylim(-20000, 20000)
            # plt.show()
            plt.savefig(os.path.join(dirName, sep_audio_1_fig))

            # Plot Independent Component #2
            plt.figure(figsize=(12,2))
            plt.title('Independent Component #2')
            plt.plot(sep_timing2,sep_audio_signal_2, c="#87de72")
            plt.ylim(-20000,20000)
            # plt.show()
            plt.savefig(os.path.join(dirName, sep_audio_2_fig))

            del sep_mix_1_wave,sep_mix_2_wave,sep_audio_signal_1_raw,sep_audio_signal_1,sep_audio_signal_2_raw,sep_audio_signal_2,sep_fs,sep_fs2,sep_timing,sep_timing2
            plt.close("all")
            gc.collect()
            