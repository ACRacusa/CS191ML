import sys, os
import wave
import gc
import numpy as np
from sklearn import decomposition as dc
from scipy.io import wavfile
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
            mix_1_wave = wave.open(os.path.join(dirName, music_1),'r')
            mix_2_wave = wave.open(os.path.join(dirName, music_2),'r')

            # Extract Raw Audio from Wav File
            audio_signal_1_raw = mix_1_wave.readframes(-1)
            audio_signal_1 = np.fromstring(audio_signal_1_raw, 'Int16')
            audio_signal_2_raw = mix_2_wave.readframes(-1)
            audio_signal_2 = np.fromstring(audio_signal_2_raw, 'Int16')

            # Get the framerate and timing
            fs = mix_1_wave.getframerate()
            timing = np.linspace(0, len(audio_signal_1)/fs, num=len(audio_signal_1))
            fs2 = mix_2_wave.getframerate()
            timing2 = np.linspace(0, len(audio_signal_2)/fs2, num=len(audio_signal_2))

            X = list(zip(audio_signal_1,audio_signal_2))
            
            # Initialize FastICA with n_components=2
            ica = dc.FastICA(n_components=2)  

            # Run the FastICA algorithm using fit_transform on dataset X
            ica_result = ica.fit_transform(X)
            
            result_audio_signal_1 = ica_result[:,0]
            result_audio_signal_2 = ica_result[:,1]
            
            # Map the values to the appropriate range for int16 audio. That range is between -32768 and +32767.
            # The sounds will be a little faint, we can increase the volume by multiplying by a value like 100
            result_audio_signal_1_int = np.int16(result_audio_signal_1*32767*100)
            result_audio_signal_2_int = np.int16(result_audio_signal_2*32767*100)

            
            # Write wave files
            wavfile.write(os.path.join(dirName, separated_1), fs, result_audio_signal_1_int)
            wavfile.write(os.path.join(dirName, separated_2), fs2, result_audio_signal_2_int)

            #increment the counter
            # ctr += 1
            del mix_1_wave,mix_2_wave,audio_signal_1, audio_signal_2 , audio_signal_1_raw,audio_signal_2_raw,fs,fs2, timing,timing2, X, ica, ica_result, result_audio_signal_1,result_audio_signal_2, result_audio_signal_1_int,result_audio_signal_2_int
            gc.collect()


    # ##########################################################################

    # # Read the wave file
    # mix_1_wave = wave.open(os.path.join(path, music_1),'r')
    
    # #Get the parameters
    # # print(mix_1_wave.getparams())

    # # Extract Raw Audio from Wav File
    # audio_signal_1_raw = mix_1_wave.readframes(-1)
    # audio_signal_1 = np.fromstring(audio_signal_1_raw, 'Int16')

    # fs = mix_1_wave.getframerate()
    # timing = np.linspace(0, len(audio_signal_1)/fs, num=len(audio_signal_1))
    # #prints out the length of the audio signal
    # print(len(audio_signal_1))
    # plt.figure(figsize=(12,2))
    # plt.title('Recording 1')
    # plt.plot(timing,audio_signal_1, c="#3ABFE7")
    # plt.ylim(-35000, 35000)
    # plt.show()
    
    # ############### get music 2 #########################
    # mix_2_wave = wave.open(os.path.join(path, music_2),'r')
    
    # #Get the parameters
    # # print(mix_2_wave.getparams())

    # # Extract Raw Audio from Wav File
    # audio_signal_2_raw = mix_2_wave.readframes(-1)
    # audio_signal_2 = np.fromstring(audio_signal_2_raw, 'Int16')

    # fs2 = mix_2_wave.getframerate()
    # timing = np.linspace(0, len(audio_signal_2)/fs2, num=len(audio_signal_2))
    # #prints out the length of the audio signal
    # print(len(audio_signal_1))
    # plt.figure(figsize=(12,2))
    # plt.title('Recording 2')
    # plt.plot(timing,audio_signal_2, c="#df8efd")
    # plt.ylim(-35000, 35000)
    # plt.show()

    # X = list(zip(audio_signal_1,audio_signal_2))
    
    # # Initialize FastICA with n_components=3
    # ica = dc.FastICA(n_components=2)  
    # #ica.fit(X)
    # # Run the FastICA algorithm using fit_transform on dataset X
    # ica_result = ica.fit_transform(X)
    

    # result_audio_signal_1 = ica_result[:,0]
    # result_audio_signal_2 = ica_result[:,1]
    
    # # Map the values to the appropriate range for int16 audio. That range is between -32768 and +32767.
    # # The sounds will be a little faint, we can increase the volume by multiplying by a value like 100
    # result_audio_signal_1_int = np.int16(result_audio_signal_1*32767*100)
    # result_audio_signal_2_int = np.int16(result_audio_signal_2*32767*100)

    # # Plot Independent Component #1
    # plt.figure(figsize=(12,2))
    # plt.title('Independent Component #1')
    # plt.plot(result_audio_signal_1_int, c="#f65e97")
    # plt.ylim(-0.010, 0.010)
    # plt.show()

    # # Plot Independent Component #2
    # plt.figure(figsize=(12,2))
    # plt.title('Independent Component #2')
    # plt.plot(result_audio_signal_2_int, c="#87de72")
    # plt.ylim(-0.010, 0.010)
    # plt.show()

    # # Write wave files
    # wavfile.write(os.path.join(path, separated_1), fs, result_audio_signal_1_int)
    # wavfile.write(os.path.join(path, separated_2), fs, result_audio_signal_2_int)

# if __name__ == "__main__":
#     main()