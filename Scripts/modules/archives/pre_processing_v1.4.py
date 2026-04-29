"""
Title: Pre_processing Memocrush Extension Pilot n2
Author: Elyès Tabbane
Date: April 15, 2024
Code Version of Experiment: code_ext_V3.3
Version of the processing pipeline: V1.1
Python Version: 3.9.17 

Description:
This file pre-processes raw data from data/raw and saves the results pandas dataframe
in a csv file in data/processed

Dependencies: see functions.py
"""
import sys
import os

# Adjust sys.path to include the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.functions import *
from modules.params import *

# Col_names changes from pilot n1. Columns were added in second pilot.


file_name="20240416_15h50_memocrush_extension_pilote2_data.csv"
col_names=['participant_ID', 'sequences_structure', 'seq', 'performance','element_selectors', 'confidence', 'counter', 'presentation_time',
       'start_time', 'click_timings_before', 'click_timings_after',
       'interclick_timings_before', 'interclick_time',
       'response_sequences_before', 'response_sequences_after', 'score',
       'current_score', 'screen_width', 'screen_height', 'set_delay',
       'break_time', 'SOA', 'last_click', 'state', 'survey', 'problem']
raw=pd.read_csv(f"/Users/et/Documents/UNICOG/2-Experiments/memocrush/Data/raw/experiment-2/{file_name}", names=col_names, header=None, index_col=False,delimiter=',',quotechar='"',skipinitialspace=True,encoding='utf-8')

# Create a new dataset with only the pure data. This can be removed if we want to check hardware of participants or demographics
saved_columns=["participant_ID","sequences_structure","seq","performance","confidence","counter",'start_time',"click_timings_before","click_timings_after","interclick_timings_before","interclick_time","response_sequences_before","response_sequences_after","score","state","survey"]
clean_raw=raw[saved_columns]

# Drop duplicates
clean_raw.drop_duplicates(subset='click_timings_after',inplace=True)
clean_raw.reset_index(drop=True,inplace=True)
clean_raw.drop(1,axis=0,inplace=True)
clean_raw.reset_index(drop=True,inplace=True)

# Droping trials from experimenters
clean_raw.drop(clean_raw[clean_raw['participant_ID'].isin(experimenter_id)].index, inplace=True)
clean_raw.reset_index(drop=True,inplace=True)

clean_raw.drop(0,axis=0,inplace=True)
clean_raw.reset_index(drop=True,inplace=True)

# Excluse training trials


# ---------------------------------------
# ********** Formating dataset **********
# ---------------------------------------

clean_raw=str2int_dataset(clean_raw, label_int_col=label_int_col_old,processed=True)



# -- Organizing Survey Answers
clean_raw['survey'].fillna("no_data,no_data,no_data,no_data", inplace=True)
survey=[x.split(',') for x in clean_raw['survey'].to_numpy()]
clean_raw=clean_raw.drop('survey',axis=1)
clean_raw['survey_age']=[x[0] for x in survey]
clean_raw['survey_degree']=[x[1] for x in survey]
clean_raw['survey_music']=[x[2] for x in survey]
clean_raw['survey_math']=[x[3] for x in survey]

# ---------------------------------------
# ************* DL-Distance *************
# ---------------------------------------
processed_clean=clean_raw.copy()

# -- Add a column to my dataframe for distance DL. Same for dl-distance that compares the structure and not the true response
processed_clean["distance_dl"]=0


# -- Compute all the distance of Damerau-Levenshtein and add them to our dataframe
for i, row in processed_clean.iterrows():
    processed_clean.at[i, "distance_dl"] = dl_distance(row["seq"], row["response_sequences_after"])

    
    

# ---------------------------------------
# ************* Token Error *************
# ---------------------------------------
# explanation: a response is a token error if it misses or adds to the base set of positions used to build 
# the stimuli

for i in range(len(processed_clean)):
    test=is_token_err(processed_clean.at[i,"seq"], processed_clean.at[i,"response_sequences_after"])
    processed_clean.at[i,"TokenErr"]=test
    processed_clean.at[i,"TokenErr_forg"]=False
    processed_clean.at[i,"TokenErr_add"]=False
    if test:   
    # -- Adding (not in precedent experiment): token_forg and token_add
    # if True: token_forg => case of TokenErr where at least one of the tokens is in missing (has been forgotten)
    # if False: token_add => case of TokenErr where at least one of the tokens is missing
        test_forg=is_token_forg(processed_clean.at[i,"seq"], processed_clean.at[i,"response_sequences_after"])
        processed_clean.at[i,"TokenErr_forg"]=test_forg
        processed_clean.at[i,"TokenErr_add"]=not test_forg

# Create a column in processed_clean that allows comparing different answers with the same temporal structure
column_holder=[]
for i in range(len(processed_clean)):
    column_holder.append(compare_tokens(processed_clean.iloc[i]["seq"],processed_clean.iloc[i]["response_sequences_after"]))
processed_clean["comparable_temp"]=column_holder



# ---------------------------------------
# *********** Legacy Support ************
# ---------------------------------------

# == Change the name of the response_sequences_after to sequences_response to match the old script
processed_clean.rename(columns={'response_sequences_after': 'sequences_response'}, inplace=True)
processed_clean.rename(columns={'response_sequences_before': 'sequences_response_before'}, inplace=True)

# == Have all the structures expressions in the same format (starting at 0)
adjust_expression(processed_clean)

# -- Tagging sequences
processed_clean["seq_name"]=processed_clean["sequences_structure"].apply(lambda x: which_seq(x))

# ---------------------------------------
# ************* DL-Structure *************
# ---------------------------------------

holder=[]
for index, row in processed_clean.iterrows():
    holder.append(dl_distance(row['sequences_structure'],row['comparable_temp']))
processed_clean['dl_structure']=holder

# ---------------------------------------
# *********** LoT Complexity ************
# ---------------------------------------

# -- We will assign a complexity of -1 for 'training' sequencess
matched_values = [complexities_allOperations_noChunk.get(value, -1) for value in processed_clean['seq_name']]
processed_clean['LoT Complexity']=matched_values

# ---------------------------------------
# ************* Geometry ***************
# ---------------------------------------
# Adding columns

# --- Point distance (dist_point): array of Euclidian distances between the different tokens that compose the sequence
# (unit of distance is a point of the figure)
holder=[]
for index, row in processed_clean.iterrows():
    holder.append(array_point_dist(row['seq']))
processed_clean['geom_dist_point']=holder

# ---------------------------------------
# **** Total duration of experiment *****
# ---------------------------------------


# To get the durations of the experiment, we get the values of last entry timepoints (in ms)
last_trial=[]

# Build a dictionary for durations
dict_duration={}

# -- We need an array that will hold the IDs of the participants to keep track of which timepoints we need to take
IDs=[processed_clean["participant_ID"][0]]
for i in range(len(processed_clean)-1):
    if processed_clean["participant_ID"][i] not in IDs:
        IDs.append(processed_clean["participant_ID"][i])
        last_trial.append(processed_clean["click_timings_after"][i-1][-1])
last_trial.append(processed_clean["click_timings_after"][len(processed_clean)-1][-1])
readable_last_trial=[i/60000 for i in last_trial]

# Fill in the duration dictionary by associating an ID with a duration of experiment
for k in range(len(readable_last_trial)):
    dict_duration[IDs[k]]=readable_last_trial[k]

# Create the new column
processed_clean['total duration experiment']=processed_clean['participant_ID'].apply(dict_duration.get)

# ---------------------------------------
# ******** Exclusion Criteria ***********
# ---------------------------------------
#
# ---- Token error on every trial
#
# Group by participant_ID and check if all TokenErr values are True
token_err_participants=processed_clean.groupby('participant_ID')['TokenErr'].all()

# Filter participant_IDs where all TokenErr values are True
participants_with_token_err = token_err_participants[token_err_participants].index

# Convert to list if needed
participants_with_token_err_list = participants_with_token_err.tolist()

# Drop the rows
processed_clean.drop(processed_clean[processed_clean['participant_ID'].isin(participants_with_token_err_list)].index, inplace=True)

#
# ---- Experiment lasted more than 25 minutes
#
filtered_data = processed_clean[processed_clean['total duration experiment'] <= 25]
drop_IDs_time=len(np.unique(processed_clean['participant_ID']))-len(np.unique(filtered_data['participant_ID']))
print(f'{drop_IDs_time} participants were dropped for spending more than 25 minutes on the experiment')
processed_clean=filtered_data
processed_clean.reset_index(inplace=True, drop=True)

# ---------------------------------------
# ********** Save new dataset ***********
# ---------------------------------------

# NOTE if you get "Parser Error", it appears with Python 3.11 and higher. Use python version 3.9 not higher 
processed_clean.to_csv(f"/Users/et/Documents/UNICOG/2-Experiments/memocrush/Data/processed/experiment-2/processed_{file_name}", index=False)

"""
# ---------------------------------------
# *********** Change logs ************
# ---------------------------------------

Date: 09 July 2024
Version: V1.4
Changes:
    - Adding the total duration of experiment to the processed dataset
    
Date: 23 May 2024
Version: V1.3
Changes:
    - using: complexities_allOperations_noChunk (removing chunk)
    
Date: 15 April 2024
Version: V1.1
Changes:
    - Adding the Language of Thought Complexity to each row
    - Adding math level and music level to each row

"""