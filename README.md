# Fully-automatic and Reliable Stridor Detection with Small Training Voice Samples via Patch-wise Few-shot Learning in Multiple System Atrophy
This repository contains the reference Pytorch source code for the following paper:
\\
Fully-automatic and Reliable Stridor Detection with Small Training Voice Samples via Patch-wise Few-shot Learning in Multiple System Atrophy
Jong Hyeon Ahn, MD*, Ju Hwan Lee*, Chae Yeon Lim, Eun Yeon Joo, MD, PhD, Jinyoung Youn, MD, PhD, Myung Jin Chung, MD, PhD, Jin Whan Cho, MD, PhD, and Kyungsu Kim PhD (* denotes equal contribution)

# Preparation
1. PFL.yaml (change prefix dir)
2. conda env create -f PFL.yaml
3. conda activate PFL
4. PFL-SD pre-trained weights
https://drive.google.com/file/d/1vW4AUWetAJc_m-uWaSF_eMJ51vpAp8hN/view?usp=sharing

# Train & Inference 
1. Preprocessing - python audio_preprocessing.py (extract only snoring/stridor part -> patch split -> log-Mel transformation) 

2. Train - bash train.sh 

3. Test - bash test.sh 

4. Evaluation - python evaluation.py (majority voting per patient, save ROC_curve, print confusion matrix)

5. Visualization - bash visualization.sh
Download video (supplementary file S2, S3 in paper) - https://drive.google.com/file/d/1vW4AUWetAJc_m-uWaSF_eMJ51vpAp8hN/view?usp=sharing

6. Combine audio file & video file - ffmpeg -i input_video.avi -i input_audio.mp3 -c copy output_video.avi

