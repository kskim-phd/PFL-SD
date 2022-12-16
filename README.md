# Fully-automatic and Reliable Stridor Detection with Small Training Voice Samples via Patch-wise Few-shot Learning in Multiple System Atrophy
This repository contains the reference Pytorch source code for the following paper:
<br/>
Fully-automatic and Reliable Stridor Detection with Small Training Voice Samples via Patch-wise Few-shot Learning in Multiple System Atrophy
<br/>
<br/>
Jong Hyeon Ahn, MD*, Ju Hwan Lee*, Chae Yeon Lim, Eun Yeon Joo, MD, PhD, Jinyoung Youn, MD, PhD, Myung Jin Chung, MD, PhD, Jin Whan Cho, MD, PhD, and Kyungsu Kim, PhD (* denotes equal contribution)

# Preparation
1. PFL.yaml (change prefix dir)
2. conda env create -f PFL.yaml
3. conda activate PFL
4. PFL-SD pre-trained weights - https://drive.google.com/file/d/1TYMQj8LIkPgFHBePzAOu9UZyvonSCuO7/view?usp=share_link

# Train and Inference 
1. Preprocessing - python audio_preprocessing.py (extract only snoring/stridor part -> split patch -> log-Mel transformation) 

2. Train - bash train.sh 

3. inference - bash test.sh 

4. Evaluation - python evaluation.py (majority voting per patient, save ROC_curve, print confusion matrix)

# Visualization
1. Visualization - bash visualization.sh    
    Download video (supplementary file S2, S3 in paper) - https://drive.google.com/file/d/1sc2TILt--4renzkWjo_Q6RqsZbKgx-JG/view?usp=share_link
    
2. Combine audio file & video file - ffmpeg -i input_video.avi -i input_audio.mp3 -c copy output_video.avi

# PFL-SD results
![acc](https://user-images.githubusercontent.com/93506254/208030122-d13a6d01-3960-4878-9370-99231bdb2db9.PNG)

![confusion](https://user-images.githubusercontent.com/93506254/208030226-5f433fab-e4ef-4f9f-9463-2fe99904cddd.PNG)

![ROC](https://user-images.githubusercontent.com/93506254/208030239-6f3d1a10-c18c-4be3-84be-d700b191be45.PNG)

![waveplot](https://user-images.githubusercontent.com/93506254/208030253-9d8f3360-e085-4728-b02c-1ae05e3287de.PNG)


# Data Availability
The main data supporting the results of this study are reported within the paper. The raw datasets from Samsung Medical Center are protected to preserve patient privacy, but they can be made available upon reasonable request if approval is obtained from the corresponding Institutional Review Board.

# reference
We developed the proposed code based on original code referenced below.
<br/>
Original few-shot code is stated in these following references:
<br/>
Few-shot: https://github.com/Tsingularity/FRN



