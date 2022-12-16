import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob

speechFileList = glob.glob("/hdd3/snoring_stridor_original_data/Snoring_AudioClips/SS_043_snoring.wav")

for speechFile in tqdm(speechFileList):
  y, sr = librosa.load(speechFile,sr=22050)

  filename = speechFile.split('/')[-1].replace('.wav','')
  path = '/hdd3/stridor_visualization/supple/test/SS_snoring_043'
  print(path)
  if not os.path.isdir(path):
    os.makedirs(path)
  max_db = librosa.amplitude_to_db(y).max()
  threshold_db = librosa.amplitude_to_db(y).max() - 15 #15 is average dB
  max_amp = librosa.db_to_amplitude(max_db)
  threshold_amp = librosa.db_to_amplitude(threshold_db)

  plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(y) / sr, len(y))
  plt.plot(t, y, color='black', label = 'Entire waveform')
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude") 
  plt.title("Waveform") 
  plt.legend(loc='upper right')
  
  plt.savefig('{}/waveform_origin_large.png'.format(path), bbox_inches='tight', pad_inches=0)
  plt.clf()
  
  clips = librosa.effects.split(y, top_db=15)
  
  wav_data = []
  for c in clips:
      data = y[c[0]: c[1]]
      wav_data.extend(data)

  wav_data_overlap = np.zeros(y.shape)
  for i in clips:
    start,end = i
    wav_data_overlap[start:end] = y[start:end]

  wav_data_overlap[wav_data_overlap==0.] = np.nan

  plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(y) / sr, len(y))
  plt.plot(t, y, color = 'black',label = 'Entire waveform')
  plt.plot(t, wav_data_overlap,label = 'New waveform with larger than threshold')
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude") 
  plt.title("Waveform") 
  plt.axhline(y = threshold_amp, color='r', linestyle = '-', linewidth= '2', label = 'Threshold')
  plt.axhline(y = -threshold_amp, color='r', linestyle = '-', linewidth= '2')
  plt.legend(loc='upper right')
  
  plt.savefig('{}/waveform_threshold_with_pre_large.png'.format(path), bbox_inches='tight', pad_inches=0)
  plt.clf()
  
  plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(wav_data) / sr, len(wav_data))
  plt.plot(t, wav_data, label = 'Pre-processed waveform')
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude")
  plt.title("Waveform")
  plt.legend(loc='upper right')
  plt.savefig('{}/perprocessed_waveform_large.png'.format(path), bbox_inches='tight', pad_inches=0)
  plt.clf()

  wav_data = np.array(wav_data)
  snoring_wav = np.zeros(wav_data.shape)
  stridor_wav = np.zeros(wav_data.shape)

  plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(wav_data) / sr, len(wav_data))
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude")
  plt.title("Waveform")

  snoring_wav = wav_data

  stridor_wav[stridor_wav==0.]=np.nan 

  plt.plot(t,snoring_wav, color = 'tab:orange', label = 'Predicted as snoring')
  plt.plot(t,stridor_wav, color = 'tab:purple', label = 'Predicted as stridor')
  

  for x_coor in list(range(0,int(len(wav_data) / sr),5)):
    plt.axvline(x = x_coor, color='black', linestyle = '-', linewidth= '2')

  plt.legend(loc='upper right', fontsize = 12)
  
  plt.savefig('{}/perprocessed_waveform_large_overlap_2.png'.format(path), bbox_inches='tight', pad_inches=0)
  plt.clf()

  plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(wav_data) / sr, len(wav_data))
  plt.plot(t, wav_data, label = 'Pre-processed waveform')
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude")
  plt.title("Waveform")
  for x_coor in list(range(0,int(len(wav_data) / sr),5)):
    plt.axvline(x = x_coor, color='black', linestyle = '-', linewidth= '2')

  plt.legend(loc='upper right', fontsize = 12)
  
  plt.savefig('{}/perprocessed_patch_split.png'.format(path), bbox_inches='tight', pad_inches=0)
  plt.clf()


  wav_data_idx = []
  for i in clips:
    wav_data_idx.append(i)
  
  wav_data_idx_all = []  
  for i in wav_data_idx:
    idx_list = list(range(i[0], i[1], 1))
    wav_data_idx_all.extend(idx_list)
  
  stridor_wav_entire = np.zeros(y.shape)
  
  stridor_wav_entire[stridor_wav_entire==0.] = np.nan

  plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(y) / sr, len(y))
  plt.plot(t, y, color = 'black',label = 'No ROI')
  plt.plot(t, wav_data_overlap, color = 'tab:orange', label = 'Predicted as snoring')
  plt.plot(t, stridor_wav_entire, color = 'tab:purple', label = 'Predicted as stridor')
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude") 
  plt.title("Waveform") 
  plt.legend(loc='upper right')
  
  plt.savefig('{}/waveform_final_diagnosis.png'.format(path), bbox_inches='tight', pad_inches=0)
  plt.clf()