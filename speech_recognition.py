#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave

#For opening files in the folder
import os
import glob

sample_rate = 44100
sample_rate_new = 0

#Opening audio files
def open_audio (folder_name, genre_list, genre_list_time):
	files = os.listdir(folder_name)

	for filename in glob.glob(os.path.join(folder_name, '*.wav')):
	    rate, audio_signal = wave.read(filename)
	    genre_list.append(audio_signal)

	    time = np.linspace(0, len(audio_signal)/rate, num=len(audio_signal))
	    genre_list_time.append(time)

def plot_signal(time, signal):
	plt.plot(time, signal)
	plt.show()

def find_max(signal_array):
	_max = max (signal_array[0])

	for i in range(1, len(signal_array)):
		if max(signal_array[i]) > _max:
			_max = max(signal_array[i])
	return _max

def normalize(signal):
	_abs = [np.abs(i) for i in signal]

	max_amplitude = find_max(_abs)
	
	for i in range(0, len(signal)):
		signal[i] = signal[i]/max_amplitude

def warp(signal_time, signal):
	global sample_rate_new

	avg = get_average_length(signal_time)
	sample_rate_new = int(avg)
	for i in range (0, len(signal_time)):
		signal_time[i] = np.linspace(0, avg/sample_rate, num=len(signal[i]))
	

def get_average_length(lst):
	_sum = 0
	for i in range (0, len(lst)):
		_sum += len(lst[i])
	avg = _sum/len(lst)

	return avg

def get_average(lst):
	_sum = 0
	for i in range (0, len(lst)):
		_sum += (lst[i])
	avg = _sum/len(lst)

	return avg

def average_energy(no_segment, signal_array, signal_array_new):
	indices_per_segment = len(signal_array)/no_segment
	indices_per_segment = int(indices_per_segment)
	stop_index = 0

	for i in range (0, no_segment):
		signal_array_new.append(get_average(signal_array[stop_index:stop_index+indices_per_segment]))
		stop_index += indices_per_segment+1

def find_euclidean_similarity(signal_new):
	global rock_avg_e
	global blues_avg_e
	global jazz_avg_e
	global soul_avg_e
	global pop_avg_e

	diff_with_rock = []
	diff_with_blues = []
	diff_with_jazz = []
	diff_with_soul = []
	diff_with_pop = []

	for i in range (0, len(rock)):
		for j in range (0, len(rock_avg_e)):
			diff_with_rock.append(abs(rock_avg_e[i][j] - signal_new[j]))
			diff_with_blues.append(abs(blues_avg_e[i][j] - signal_new[j]))
			diff_with_jazz.append(abs(jazz_avg_e[i][j] - signal_new[j]))
			diff_with_soul.append(abs(soul_avg_e[i][j] - signal_new[j]))
			diff_with_pop.append(abs(pop_avg_e[i][j] - signal_new[j]))

	_min = []

	_min.append(min(diff_with_rock))
	_min.append(min(diff_with_blues))
	_min.append(min(diff_with_jazz))
	_min.append(min(diff_with_soul))
	_min.append(min(diff_with_pop))

	index = _min.index(min(_min))

	return index


#Opening and saving samples to lists
rock = []
blues = []
jazz = []
soul = []
pop = []

rock_time = []
blues_time = []
jazz_time = []
soul_time = []
pop_time = []

rock_path = '/Users/arnes/Desktop/PR Assignment 3/Sample audio data/Rock'
blues_path = '/Users/arnes/Desktop/PR Assignment 3/Sample audio data/Blues'
jazz_path = '/Users/arnes/Desktop/PR Assignment 3/Sample audio data/Jazz'
soul_path = '/Users/arnes/Desktop/PR Assignment 3/Sample audio data/Soul'
pop_path = '/Users/arnes/Desktop/PR Assignment 3/Sample audio data/Pop'

open_audio(rock_path, rock, rock_time)
open_audio(blues_path, blues, blues_time)
open_audio(jazz_path, jazz, jazz_time)
open_audio(soul_path, soul, soul_time)
open_audio(pop_path, pop, pop_time)


#STEP 1 - ENHANCE SIGNAL

#Normalization
plot_signal(rock_time[0], rock[0])
normalize(rock)
plot_signal(rock_time[0], rock[0])
normalize(blues)
normalize(jazz)
normalize(soul)
normalize(pop)

#Warping
warp(rock_time, rock)
warp(blues_time, blues)
warp(jazz_time, jazz)
warp(soul_time, soul)
warp(pop_time, pop)

#STEP 2 - SEGMENTATION
no_segments = 10

#STEP 3 - FEATURE EXTRACTION
#Average energy
rock_avg_e = [[] for i in range (no_segments)]
blues_avg_e = [[] for i in range (no_segments)]
jazz_avg_e = [[] for i in range (no_segments)]
soul_avg_e = [[] for i in range (no_segments)]
pop_avg_e = [[] for i in range (no_segments)]

for i in range(0, len(rock)):
	average_energy(no_segments, rock[i], rock_avg_e[i])
	average_energy(no_segments, blues[i], blues_avg_e[i])
	average_energy(no_segments, jazz[i], jazz_avg_e[i])
	average_energy(no_segments, soul[i], soul_avg_e[i])
	average_energy(no_segments, pop[i], pop_avg_e[i])


#Getting input audio
testfile1 = "/Users/arnes/Desktop/PR Assignment 3/Test/rock1.wav"
testfile2 = "/Users/arnes/Desktop/PR Assignment 3/Test/blues1.wav"
testfile3 = "/Users/arnes/Desktop/PR Assignment 3/Test/jazz1.wav"
testfile4 = "/Users/arnes/Desktop/PR Assignment 3/Test/soul1.wav"
testfile5 = "/Users/arnes/Desktop/PR Assignment 3/Test/pop1.wav"

test_audio_signal1 = []
test_audio_signal2 = []
test_audio_signal3 = []
test_audio_signal4 = []
test_audio_signal5 = []

rate1, test_audio_signal = wave.read(testfile1)
test_audio_signal1.append(test_audio_signal)

rate2, test_audio_signal = wave.read(testfile2)
test_audio_signal2.append(test_audio_signal)

rate3, test_audio_signal = wave.read(testfile3)
test_audio_signal3.append(test_audio_signal)

rate4, test_audio_signal = wave.read(testfile4)
test_audio_signal4.append(test_audio_signal)

rate5, test_audio_signal = wave.read(testfile5)
test_audio_signal5.append(test_audio_signal)

test_time1 = []
test_time2 = []
test_time3 = []
test_time4 = []
test_time5 = []

test_time1.append(np.linspace(0, len(test_audio_signal1)/rate1, num=len(test_audio_signal1)))
test_time2.append(np.linspace(0, len(test_audio_signal2)/rate2, num=len(test_audio_signal2)))
test_time3.append(np.linspace(0, len(test_audio_signal3)/rate3, num=len(test_audio_signal3)))
test_time4.append(np.linspace(0, len(test_audio_signal4)/rate4, num=len(test_audio_signal4)))
test_time5.append(np.linspace(0, len(test_audio_signal5)/rate5, num=len(test_audio_signal5)))


#Processing input audio
normalize(test_audio_signal1)
normalize(test_audio_signal2)
normalize(test_audio_signal3)
normalize(test_audio_signal4)
normalize(test_audio_signal5)

warp(test_time1, test_audio_signal1)
warp(test_time2, test_audio_signal2)
warp(test_time3, test_audio_signal3)
warp(test_time4, test_audio_signal4)
warp(test_time5, test_audio_signal5)

test_1_avg_e = [[] for i in range (no_segments)]
test_2_avg_e = [[] for i in range (no_segments)]
test_3_avg_e = [[] for i in range (no_segments)]
test_4_avg_e = [[] for i in range (no_segments)]
test_5_avg_e = [[] for i in range (no_segments)]

average_energy(no_segments, test_audio_signal1[0], test_1_avg_e[0])
average_energy(no_segments, test_audio_signal2[0], test_2_avg_e[0])
average_energy(no_segments, test_audio_signal3[0], test_3_avg_e[0])
average_energy(no_segments, test_audio_signal4[0], test_4_avg_e[0])
average_energy(no_segments, test_audio_signal5[0], test_5_avg_e[0])

#STEP 5 - FEATURE MATCHING
#Calculating similarity with Euclidean distance
sim = []
sim.append(find_euclidean_similarity(test_1_avg_e[0]))
sim.append(find_euclidean_similarity(test_2_avg_e[0]))
sim.append(find_euclidean_similarity(test_3_avg_e[0]))
sim.append(find_euclidean_similarity(test_4_avg_e[0]))
sim.append(find_euclidean_similarity(test_5_avg_e[0]))

#Give output
choices = ["Rock", "Blues", "Jazz", "Soul", "Pop"]

for i in range(0, len(sim)):
	answer = choices[sim[i]]
	print("Speech", i+1, "recognized as", answer)

