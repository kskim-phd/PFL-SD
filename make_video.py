import librosa
import librosa.display
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

def zeropad(num): #get file name
    num=str(num)
    return (4-len(num))*'0'+num

#Snoring
speechFileList = glob.glob("/hdd3/snoring_stridor_original_data/Snoring_AudioClips/SS_031_snoring.wav")
for speechFile in tqdm(speechFileList):
  y, sr = librosa.load(speechFile,sr=22050)
  filename = speechFile.split('/')[-1].replace('.wav','')
  
  clips = librosa.effects.split(y, top_db=15)

  wav_data_overlap = np.zeros(y.shape)
  for i in clips:
    start,end = i
    wav_data_overlap[start:end] = y[start:end]

  wav_data_overlap[wav_data_overlap==0.] = np.nan

  wav_data_idx = []
  for i in clips: # i = entire_waveform's idx (meeting the threshold) 
    wav_data_idx.append(i)
  
  wav_data_idx_all = [] 
  for i in wav_data_idx:
    idx_list = list(range(i[0], i[1], 1))
    wav_data_idx_all.extend(idx_list)
  
  stridor_wav_entire = np.zeros(y.shape)

  #031
  for i in wav_data_idx_all[21*5*sr:22*5*sr]:
    stridor_wav_entire[i] = y[i]
  
  stridor_wav_entire[stridor_wav_entire==0.] = np.nan
  
  fig=plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(y) / sr, len(y))
  plt.plot(t, y, color = 'black',label = 'No ROI')
  plt.plot(t[:1], wav_data_overlap[:1], color = 'tab:orange', label = 'Predicted as snoring')
  plt.plot(t[:1], stridor_wav_entire[:1], color = 'tab:purple', label = 'Predicted as stridor')
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude") 
  plt.title("Waveform") 
  plt.legend(loc='upper right')

  x=0
  idx=0
  for idx in tqdm(range(0,len(stridor_wav_entire),sr)):
    plt.figure(figsize=(24, 4))
    t = np.linspace(0, len(y) / sr, len(y))
    
    plt.plot(t, y, color = 'black',label = 'No ROI')
    plt.plot(t[:idx], wav_data_overlap[:idx], color = 'tab:orange', label = 'Predicted as snoring')
    plt.plot(t[:idx], stridor_wav_entire[:idx], color = 'tab:purple', label = 'Predicted as stridor')
    
    minval = idx-(sr*1) if idx-(sr*1)>0 else 0
    maxval = len(stridor_wav_entire)-1 if idx+(sr*1)>len(stridor_wav_entire)-1 else idx+(sr*1)
    if (~np.isnan(stridor_wav_entire[minval:maxval])).sum()>0:
        label = 'Stridor'
    elif (~np.isnan(wav_data_overlap[minval:maxval])).sum()>0:
        label = 'Snoring'
    else:
        label = 'No ROI'

    if label == 'Stridor':
      color = 'tab:purple'
    elif label == 'Snoring':
      color = 'tab:orange'
    else:
      color = 'black'

    plt.plot([t[idx],t[idx]],[-0.5,0.5],color='black',linewidth='3')   #waveplot amplitude range
    plt.text(t[idx]+4,0.48,label,fontsize=14, color=color)#,rotation=90)  # +(text 가로위치), () text 세로 위치
    plt.margins(x=0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.legend(loc='lower right',fontsize=12)
    
    plt.savefig('/hdd4/stridor_visualization/31_2/waveform_final_diagnosis_'+zeropad(int(idx/sr))+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

#make video 
paths = sorted(glob.glob('/hdd4/stridor_visualization/31_2/*.png'))

frame_array = []
for idx , path in tqdm(enumerate(paths)) : 
    asd=np.ones((186, 1006, 3))*255
    img_ori = cv2.imread(path)
    img = cv2.resize(img_ori, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    height, width, layers = img.shape
    asd[:height,:width,:] = img
    height, width, layers = asd.shape
    size = (width,height)
    frame_array.append(asd.astype('uint8'))
out = cv2.VideoWriter('/hdd3/stridor_visualization/snoring_031_final_video_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()


#Stridor 
speechFileList = glob.glob("/hdd3/snoring_stridor_original_data/raw_data_for_figure/stridor/annotation_segment_wave/SS_007_segment_2.wav")
for speechFile in tqdm(speechFileList):
  y, sr = librosa.load(speechFile,sr=22050)
  filename = speechFile.split('/')[-1].replace('.wav','')
  
  clips = librosa.effects.split(y, top_db=15)  #average_dB * 0.3 (standard dB)

  wav_data_overlap = np.zeros(y.shape)
  for i in clips:
    start,end = i
    wav_data_overlap[start:end] = y[start:end]

  wav_data_overlap[wav_data_overlap==0.] = np.nan

  wav_data_idx = []
  for i in clips: 
    wav_data_idx.append(i)
  
  wav_data_idx_all = []  
  for i in wav_data_idx:
    idx_list = list(range(i[0], i[1], 1))
    wav_data_idx_all.extend(idx_list)
  
  snoring_wav_entire = np.zeros(y.shape)
  
  #007
  for i in wav_data_idx_all[8*5*sr:9*5*sr]:
    snoring_wav_entire[i] = y[i]
  
  snoring_wav_entire[snoring_wav_entire==0.] = np.nan

  fig=plt.figure(figsize=(24, 4))
  t = np.linspace(0, len(y) / sr, len(y))
  plt.plot(t, y, color = 'black',label = 'No ROI')
  plt.plot(t[:1], wav_data_overlap[:1], color = 'tab:purple', label = 'Predicted as stridor')
  plt.plot(t[:1], snoring_wav_entire[:1], color = 'tab:orange', label = 'Predicted as snoring')
  plt.margins(x=0)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude") 
  plt.title("Waveform") 
  plt.legend(loc='upper right',fontsize=12)

  x=0
  idx=0
  for idx in tqdm(range(0,len(snoring_wav_entire),sr)):
    plt.figure(figsize=(24, 4))
    t = np.linspace(0, len(y) / sr, len(y))
    
    plt.plot(t, y, color = 'black',label = 'No ROI')
    plt.plot(t[:idx], wav_data_overlap[:idx], color = 'tab:purple', label = 'Predicted as stridor')
    plt.plot(t[:idx], snoring_wav_entire[:idx], color = 'tab:orange', label = 'Predicted as snoring')
    minval = idx-(sr*1) if idx-(sr*1)>0 else 0
    maxval = len(snoring_wav_entire)-1 if idx+(sr*1)>len(snoring_wav_entire)-1 else idx+(sr*1)
    if (~np.isnan(snoring_wav_entire[minval:maxval])).sum()>0:
        label = 'Snoring'
    elif (~np.isnan(wav_data_overlap[minval:maxval])).sum()>0:
        label = 'Stridor'
    else:
        label = 'No ROI'

    if label == 'Stridor':
      color = 'tab:purple'
    elif label == 'Snoring':
      color = 'tab:orange'
    else:
      color = 'black'

    plt.plot([t[idx],t[idx]],[-0.5,0.5],color='black',linewidth='3')   #waveplot amplitude range
    plt.text(t[idx]+8,0.48,label,fontsize=14, color=color)
    plt.margins(x=0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.legend(loc='lower right',fontsize=12)
    
    plt.savefig('/hdd4/stridor_visualization/7_2/waveform_final_diagnosis_'+zeropad(int(idx/sr))+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

#make video
paths = sorted(glob.glob('/hdd4/stridor_visualization/7_2/*.png'))

frame_array = []
for idx , path in tqdm(enumerate(paths)) : 
    asd=np.ones((186, 1006, 3))*255
    img_ori = cv2.imread(path)
    img = cv2.resize(img_ori, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    height, width, layers = img.shape
    asd[:height,:width,:] = img
    height, width, layers = asd.shape
    size = (width,height)
    frame_array.append(asd.astype('uint8'))
out = cv2.VideoWriter('/hdd3/stridor_visualization/stridor_007_video_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()