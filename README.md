# QAIST

# Model Architecture
* Two independent speech-emotion prediction models are used. 
* They act as independent classifiers.
* Then, by ensemble method, we combine the predictions and output the final prediction.

# Code 
* Ensemble.ipynb:
	* code for executing ensemble function
	* receives prediction csv files from each model (each row = probs for each emotion for that audio file)
	* computes ensemble by selecting from different ensemble functions.
	* outputs csv file to submit to Eval AI. (but need to manually add column names: fileID, Emotion afterwards)
* speech_emotion_recognition_XJHe:
	* this code is implemented based on https://ieeexplore.ieee.org/document/8421023 this paper.
	* to execute, run train.ipynb and execute test.ipynb (be careful about the path of train and val data in extract_mel.py)
	* Dependencies
		* tensorflow == 1.5.0
		* sklearn
		* matplotlib
		* python_speech_features
		* wave
	
