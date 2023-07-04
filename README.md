# Automatic stridor detection using small training set via patch-wise few-shot learning for diagnosis of multiple system atrophy
This repository contains the reference Pytorch source code for the following paper:
<br/>
Automatic stridor detection using small training set via patch-wise few-shot learning for diagnosis of multiple system atrophy
<br/>
<br/>
Jong Hyeon Ahn, MD*, Ju Hwan Lee*, Chae Yeon Lim, Eun Yeon Joo, MD, PhD, Jinyoung Youn, MD, PhD, Myung Jin Chung, MD, PhD, Jin Whan Cho, MD, PhD+, and Kyungsu Kim, PhD+ (* contributed equally to this work as co-first authors and + contributed equally to this work as the co-corresponding authors.)

# Preparation
1. PFL.yaml (change prefix dir)
2. conda env create -f PFL.yaml
3. conda activate PFL
4. PFL-SD pre-trained weights - https://drive.google.com/file/d/1TYMQj8LIkPgFHBePzAOu9UZyvonSCuO7/view?usp=share_link

# Training and Inference 
1. Preprocessing - python audio_preprocessing.py (extract only snoring/stridor part -> split patch -> log-Mel transformation) 

2. Training - bash train.sh 

3. Inference - bash test.sh 

4. Evaluation - python evaluation.py (majority voting per patient, save ROC_curve, print confusion matrix)

# Data Availability
The main data supporting the results of this study are reported within the paper. The raw datasets from Samsung Medical Center are protected to preserve patient privacy, but they can be made available upon reasonable request if approval is obtained from the corresponding Institutional Review Board.
<br/>
Please address all correspondence concerning this code to our corresponding author Kyungsu Kim at kskim.doc@gmail.com.

# reference
We developed the proposed code based on original code referenced below.
<br/>
Original few-shot code is stated in these following references:
<br/>
Few-shot: https://github.com/Tsingularity/FRN



