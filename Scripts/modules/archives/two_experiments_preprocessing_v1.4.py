"""
Title: Pre_processing Two Sequence learning experiments
Author: Elyès Tabbane
Date: June 18, 2024
Code Version of Experiment: code_ext_V3.3
Version of the processing pipeline: V1.1
Python Version: 3.9.19

Description:
This file processes data from data/preprocessed and data/experiment1-processed to concatenate the results and save them into a new CSV file
in a csv file in data/all_experiment_processed

Dependencies: see functions.py
"""
import sys
import os

# Adjust sys.path to include the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.functions import *
from modules.params import *
from modules.complexity_measures import *


date='2024-06_18'
# --- 
# Paths of preprocessed datasets
# ---
# experiment_1 is the base experiment and experiment_2 is the extension
root_path='/Users/et/Documents/UNICOG/2-Experiments/memocrush'
path_processed_data=os.path.join(root_path,'Data/processed')
path_experiment_base=os.path.join(path_processed_data,'experiment-1/experiment1_processed_10h05_31032023_data.csv')
path_experiment_ext=os.path.join(path_processed_data,'experiment-2/processed_20240416_15h50_memocrush_extension_pilote2_data.csv')
save_path=os.path.join(path_processed_data,'both')

# --- 
# Read CSV
# ---
df_base=pd.read_csv(path_experiment_base)
df_ext=pd.read_csv(path_experiment_ext)

# --- 
# Specify experiment
# ---
df_base['which_exp']='base'
df_ext['which_exp']='extension'

# --- 
# Mark complete / incomplete experiments for extension
# ---
# Determine the maximum number of trials
max_trial = df_ext['counter'].unique()[-1]
print('max_trial= ',max_trial)

# Create a mask for incomplete experiments
participant_counts = df_ext['participant_ID'].value_counts()
incomplete_mask = df_ext['participant_ID'].map(participant_counts) != max_trial

# Mark the incomplete experiments
df_ext.loc[incomplete_mask, 'which_exp'] = 'incomplete'

# drop Training
# Drop training
df_ext=df_ext[df_ext['state']=='main_experiment']
df_ext.reset_index(drop=True,inplace=True)


# --- 
# Formating datasets
# ---
# The following reformat datasets to make the 
# concatenation possible.

df_ext = df_ext.drop(['counter', 'click_timings_before', 'interclick_timings_before', 'sequences_response_before', 'survey_age',
 'survey_degree',
 'survey_music',
 'survey_math', 'dl_structure',], axis=1)

# --- 
# Concatenating and saving dataset
# ---
concatenated_df=pd.concat([df_base, df_ext], ignore_index=True)
concatenated_df.to_csv(os.path.join(save_path,f'two_experiment_datasets_{date}.csv'), index=False)

# --- 
# Creating a dataset for complexity comparisons
# ---
comp_columns=['participant_ID','which_exp','seq_name','confidence','distance_dl','performance','LoT Complexity']
data_comp=concatenated_df[comp_columns].copy()
all_dictionary_complexities=[
    complexities_allOperations_noChunk,
    dict_shannon_entropy ,
    dict_lz_complexity,
    dict_change_complexity,
    dict_algorithmic_complexity,
    dict_subsymetrie,
    dict_chunk_complexity
]


for dictionary,name in zip(all_dictionary_complexities,name_complexities):
    matched_values = [dictionary.get(value, -1) for value in data_comp['seq_name']]
    data_comp[name]=matched_values
    
data_comp.to_csv(os.path.join(save_path,f'{date}_complexity_dataset.csv'), index=False)

# Also save the base and extension dataset separately
data_comp_base=data_comp[data_comp['which_exp']=='base']
data_comp_base.reset_index(drop=True,inplace=True)
data_comp_base.to_csv(os.path.join(save_path,f'{date}_complexity_dataset_BASE_only.csv'), index=False)


data_comp_ext=data_comp[data_comp['which_exp']=='extension']
data_comp_ext.reset_index(drop=True,inplace=True)
data_comp_ext.to_csv(os.path.join(save_path,f'{date}_complexity_dataset_EXT_only.csv'), index=False)
