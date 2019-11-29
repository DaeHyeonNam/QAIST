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
	