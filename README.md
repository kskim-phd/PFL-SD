# Preparation
1. FRN.yaml (change prefix dir)
2. conda env create -f FRN.yaml
3. conda activate FRN
4. FRN pre-trained weights
https://drive.google.com/file/d/1vW4AUWetAJc_m-uWaSF_eMJ51vpAp8hN/view?usp=sharing

# Train & Inference 
1. Preprocessing - python audio_preprocessing.py (extract only snoring/stridor part -> patch split -> log-Mel transformation) 

2. Train - bash train.sh 

3. Test - bash test.sh 

4. Evaluation - python evaluation.py (majority voting per patient, save ROC_curve, print confusion matrix)

5. Visualization - bash visualization.sh

6. Combine audio file & video file - ffmpeg -i input_video.avi -i input_audio.mp3 -c copy output_video.avi

