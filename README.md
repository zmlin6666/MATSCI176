# MATSCI176
MATSCI176 Final Project

This repository includes six files. To use the machine learning models, users must first download the battery dataset from the following link, unzip the file, and rename the extracted folder as "nasa_batteries" (this specific name is required for data preprocessing).  

Dataset Download Link: 
[Battery Data Set (NASA)](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)  

Included Files:  
1. Final_Project.ipynb 
   - This notebook can be uploaded to Google Colab to reproduce the results presented in the report.  
   - It includes instructions for data preprocessing and model training using four models: XGBoost, Random Forest, LSTM, and Bi-LSTM.  
   - The purpose of each step is documented in cell comments.  

2. Hyperparameter_Tuning.ipynb  
   - This notebook is used for hyperparameter tuning of the four models (XGBoost, Random Forest, LSTM, and Bi-LSTM) across both single and multiple feature sets (8 configurations in total).  

3. convert_nasa_dataset_discharge.py 
   - This script preprocesses the raw battery dataset to generate the processed dataset used for machine learning.  
   - In Final_Project.ipynb, it is executed using the command: %run convert_nasa_dataset_discharge.py    
   - The script processes the "nasa_batteries"*folder (containing raw data) and outputs a new directory called "nasa_batteries_processed". Within this directory, a "data.scv" file with extracted disharge data will be used for training models.  
   - It also relies on the following three files for data preprocessing:  

4. enums.py
5. preprocessing.py  
6. nasa_battery_metadata.yml 

Files 2–6 are referenced from SambaMixer:
- Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10818656)  
- GitHub: [SambaMixer Repository](https://github.com/sascha-kirch/samba-mixer)  
