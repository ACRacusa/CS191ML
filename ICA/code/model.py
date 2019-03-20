import sys, os
import wave as wv
import gc
import numpy as np
from sklearn import decomposition as dc
from scipy.io import wavfile as wf
from scipy import signal
from matplotlib import pyplot as plt


path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/CS191ML/ICA/resources/'
separated_1 = 'separated_audio_1.wav'
separated_2 = 'separated_audio_2.wav'
# music 1 is ed-sheeran
# music 2 is ben&ben 
music_1 = 'ben.wav'
music_2 = 'ed.wav'


if __name__ == "__main__":
    for dirName, subdirList, fileList in os.walk(path):
        print('Found directory: %s' % dirName)
        if(os.path.exists(os.path.join(dirName, music_1)) and os.path.exists(os.path.join(dirName, music_2))):
            # Read the wave file
            wav_file_1 = wv.open(os.path.join(dirName, music_1),'r')
            wav_file_2 = wv.open(os.path.join(dirName, music_2),'r')

            # Extract Raw Audio from Wav File
            audio_signal_1_raw = wav_file_1.readframes(-1)
            audio_signal_1 = np.fromstring(audio_signal_1_raw, 'Int16')
            audio_signal_2_raw = wav_file_2.readframes(-1)
            audio_signal_2 = np.fromstring(audio_signal_2_raw, 'Int16')

            # Get the framerate and timing
            frame_rate_1 = wav_file_1.getframerate()
            audio_timing_1 = np.linspace(0, len(audio_signal_1)/frame_rate_1, num=len(audio_signal_1))
            frame_rate_2 = wav_file_2.getframerate()
            audio_timing_2 = np.linspace(0, len(audio_signal_2)/frame_rate_2, num=len(audio_signal_2))

            data_mixed_list = list(zip(audio_signal_1,audio_signal_2))
            
            # Initialize FastICA with n_components=2
            ICA_model = dc.FastICA(n_components=2)  

            # Run the FastICA algorithm using fit_transform on dataset data_mixed_list
            ICA_model_result = ICA_model.fit_transform(data_mixed_list)
            
            result_audio_signal_1 = ICA_model_result[:,0]
            result_audio_signal_2 = ICA_model_result[:,1]
            
            # Map the values to the appropriate range for int16 audio. That range is between -32768 and +32767.
            # The sounds will be a little faint, we can increase the volume by multiplying by a value like 100
            result_audio_signal_1_int = np.int16(result_audio_signal_1*32767*100)
            result_audio_signal_2_int = np.int16(result_audio_signal_2*32767*100)

            
            # Write wave files
            wf.write(os.path.join(dirName, separated_1), 2*frame_rate_1, result_audio_signal_1_int)
            wf.write(os.path.join(dirName, separated_2), 2*frame_rate_2, result_audio_signal_2_int)

            #increment the counter
            # ctr += 1
            del wav_file_1,wav_file_2,audio_signal_1, audio_signal_2 , audio_signal_1_raw,audio_signal_2_raw,frame_rate_1,frame_rate_2, audio_timing_1,audio_timing_2, data_mixed_list, ICA_model, ICA_model_result, result_audio_signal_1,result_audio_signal_2, result_audio_signal_1_int,result_audio_signal_2_int
            gc.collect()
