import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import make_chunks
import  os
from tqdm import tqdm
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
import os

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root_current = '/'.join(currentpath)

# split audio patch 
fold = 1
diseases = ['snoring', 'stridor']
modes = ['train','test','val']

for mode in modes:
    for disease in diseases:
        audio_path = root_current+'/data/{}fold/{}/{}/'.format(fold,mode,disease)
        audio_files = glob.glob(audio_path + '*.wav')

        for audio_file in tqdm(audio_files):
            audio, sr = librosa.load(audio_file, sr= 22050)
            print(audio.shape, sr)

            clips = librosa.effects.split(audio, top_db=15)  
            print(clips)

            wav_data = []
            for c in clips:
                print(c)
                data = audio[c[0]: c[1]]
                wav_data.extend(data)
            save_dir = root_current+'/preprocessed_data/{}fold/waveform/{}/{}/'.format(fold,mode,disease)
            os.makedirs(save_dir,exist_ok=True)
            sf.write(save_dir+audio_file.split('/')[-1], wav_data, sr)


#save 5sec patch (30patch per patient)

for disease in diseases:
    dir = root_current+'/preprocessed_data/{}fold/waveform/'.format(fold)
    waveplot_train_name_list = []
    waveplot_test_name_list = []
    test_path = dir + 'test/{}/*.wav'.format(disease)
    train_path = dir + 'train/{}/*.wav'.format(disease)
    val_path = dir + 'val/{}/*.wav'.format(disease)

    train_files = glob.glob(train_path) 
    test_files = glob.glob(test_path)
    val_files = glob.glob(val_path)

    for file_path in tqdm(train_files):
        audio = AudioSegment.from_wav(file_path)
        
        waveplot_name = file_path.split("/")[-1].split(".")[0]
        splicing_resolution = 1000*5  # pydub calculates in millisec 

        chunks = make_chunks(audio, splicing_resolution)
        k = 0
        for i,chunk in enumerate(chunks):
            if k == 30:
                break
            if len(chunk) == 1000*5:
                save_folder = root_current+'/preprocessed_data/{}fold/waveform_chunk/train/'.format(fold)+str(disease)+'/'
                save_folder2 = root_current+'/preprocessed_data/{}fold/waveform_chunk/test/'.format(fold)+str(disease)+'/'
                os.makedirs(save_folder,exist_ok=True)
                os.makedirs(save_folder2,exist_ok=True)
                chunk.export(save_folder + str(1)+ "{}_chunk_combined_ver_{}.wav".format(waveplot_name, i),format="wav")
                chunk.export(save_folder2 + str(1)+ "{}_chunk_combined_ver_{}.wav".format(waveplot_name, i),format="wav")
                k+=1
            else:
                pass

    for file_path in tqdm(test_files):
        audio = AudioSegment.from_wav(file_path)
        waveplot_name = file_path.split("/")[-1].split(".")[0]
        splicing_resolution = 1000*5  # pydub calculates in millisec   

        chunks = make_chunks(audio, splicing_resolution)
        k = 0
        for i,chunk in enumerate(chunks):
            if k == 30:
                break
            if len(chunk) == 1000*5:
                save_folder = root_current+'/preprocessed_data/{}fold/waveform_chunk/test/'.format(fold)+str(disease)+'/'
                os.makedirs(save_folder,exist_ok=True)
                chunk.export(save_folder + "{}_chunk_combined_ver_{}.wav".format(waveplot_name, i),format="wav")
                k+=1
            else:
                pass

    for file_path in tqdm(val_files):
        audio = AudioSegment.from_wav(file_path)
        waveplot_name = file_path.split("/")[-1].split(".")[0]
        splicing_resolution = 1000*5  # pydub calculates in millisec  

        chunks = make_chunks(audio, splicing_resolution)
        k = 0
        for i,chunk in enumerate(chunks):
            if k == 30:  
                break
            if len(chunk) == 1000*5:
                save_folder = root_current+'/preprocessed_data/{}fold/waveform_chunk/train/'.format(fold)+str(disease)+'/'
                os.makedirs(save_folder,exist_ok=True)
                chunk.export(save_folder + str(2)+ "{}_chunk_combined_ver_{}.wav".format(waveplot_name, i),format="wav")
                k+=1
            else:
                pass



# <Log-Mel spectrogram transformation>

window_size=2048  
hop_size=512     
mel_bins=128       
fmin=20           
fmax=14000
window = 'hann'
sample_rate = 22050  
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None


spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
            
logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

fold = 1
modes2 = ['train','test']
for mode in modes2:
    data_dir = root_current+'/preprocessed_data/{}fold/waveform_chunk/{}'.format(fold,mode)
    data_list = glob.glob(data_dir + "/*/*.wav")

    for data in tqdm(data_list):
        sig , sr = librosa.core.load(data, sr=None, mono=True)
        input = sig
        input = torch.from_numpy(input)
        input.cuda()
        input = input.unsqueeze(0)

        x = spectrogram_extractor(input.float())   # (batch_size, 1, time_steps, freq_bins)
        x = logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze()
        savefolder = root_current+'/preprocessed_data/{}fold/logmel_spectrogram/'.format(fold)+str(data.split('/')[-3])+'/'+str(data.split('/')[-2])+'/'
        os.makedirs(savefolder,exist_ok=True)
        plt.imsave(savefolder+str(data.split('/')[-1].split('.')[-2])+'.png',x)