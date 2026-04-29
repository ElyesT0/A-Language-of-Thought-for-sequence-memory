"""
Title: Pre_processing Memocrush original experiment
Author: Elyès Tabbane
Date: May 13, 2024
Code Version of Experiment: memocrush_V5.6
Version of the processing pipeline: V0.1
Python Version: 3.9.17 

Description:
This file pre-processes raw data from data/raw and saves the results pandas dataframe
in a csv file in data/processed. The goal is to get the first experiment data in the
same format as the second experiment data, and merge them for the geometry analyses.

Dependencies: see functions.py
"""
import sys
import os

# Adjust sys.path to include the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from modules.functions import *
from modules.params import *

file_name='10h05_31032023_data.csv'
csv_path=f'/Users/et/Documents/UNICOG/2-Experiments/memocrush/Data/raw/experiment-1/{file_name}'

col_names=['participant_ID', 'seq','sequences_structure',  'sequences_response','timestamps', 'interclick_time', 'performance', 'score',
       'confidence_level', 'start_time',
       'screen_width', 'screen_height',
       'type', 'nb_rep', 'nav_tech','survey1','survey2', 'problem']

raw=pd.read_csv(csv_path,names=col_names, header=None, index_col=False, on_bad_lines='warn')

saved_columns=['participant_ID', 'seq','sequences_structure',  'sequences_response','timestamps', 'interclick_time', 
               'performance', 'score',
       'confidence_level', 'start_time','type']

clean_raw=raw[saved_columns]
# Drop duplicates
clean_raw.drop_duplicates(inplace=True)
clean_raw.reset_index(drop=True,inplace=True)
clean_raw.drop(0,axis=0,inplace=True)
clean_raw.reset_index(drop=True,inplace=True)

# Droping trials from experimenters
experimenter_id=["PPQ4LZFYMU74","FUQJQ7XZ5AH7","M883JAG3284Y","UOORKZ311RVH"]
clean_raw.drop(clean_raw[clean_raw['participant_ID'].isin(experimenter_id)].index, inplace=True)
clean_raw.reset_index(drop=True,inplace=True)

# -----------------------------------------------------------
# ********** Adapt old data to fit new data format **********
# -----------------------------------------------------------
clean_raw=swap_columns(clean_raw,'seq','sequences_structure')
clean_raw=swap_columns(clean_raw,'sequences_response','performance')
clean_raw=swap_columns(clean_raw,'timestamps','confidence_level')
clean_raw=swap_columns(clean_raw,'interclick_time','start_time')
clean_raw=swap_columns(clean_raw,'timestamps','sequences_response')
clean_raw=swap_columns(clean_raw,'interclick_time','score')


clean_raw.rename(columns={'confidence_level': 'confidence', 'confidence': 'confidence_level'}, inplace=True)
clean_raw.rename(columns={'timestamps': 'click_timings_after', 'click_timings_after': 'timestamps'}, inplace=True)
clean_raw.rename(columns={'type': 'state', 'state': 'type'}, inplace=True)



# ---------------------------------------
# ********** Formating dataset **********
# ---------------------------------------

clean_raw=str2int_dataset(clean_raw, label_int_col=label_int_col_old_exp1,processed=True)



# ---------------------------------------
# ************* DL-Distance *************
# ---------------------------------------
processed_clean=clean_raw.copy()

# -- Add a column to my dataframe for distance DL. Same for dl-distance that compares the structure and not the true response
processed_clean["distance_dl"]=0


# -- Compute all the distance of Damerau-Levenshtein and add them to our dataframe
for i, row in processed_clean.iterrows():
    processed_clean.at[i, "distance_dl"] = dl_distance(row["seq"], row["sequences_response"])

    
# ---------------------------------------
# ************* Token Error *************
# ---------------------------------------
# explanation: a response is a token error if it misses or adds to the base set of positions used to build 
# the stimuli

for i in range(len(processed_clean)):
    test=is_token_err(processed_clean.at[i,"seq"], processed_clean.at[i,"sequences_response"])
    processed_clean.at[i,"TokenErr"]=test
    processed_clean.at[i,"TokenErr_forg"]=False
    processed_clean.at[i,"TokenErr_add"]=False
    if test:   
    # -- Adding (not in precedent experiment): token_forg and token_add
    # if True: token_forg => case of TokenErr where at least one of the tokens is in missing (has been forgotten)
    # if False: token_add => case of TokenErr where at least one of the tokens is missing
        test_forg=is_token_forg(processed_clean.at[i,"seq"], processed_clean.at[i,"sequences_response"])
        processed_clean.at[i,"TokenErr_forg"]=test_forg
        processed_clean.at[i,"TokenErr_add"]=not test_forg

# Create a column in processed_clean that allows comparing different answers with the same temporal structure
column_holder=[]
for i in range(len(processed_clean)):
    column_holder.append(compare_tokens(processed_clean.iloc[i]["seq"],processed_clean.iloc[i]["sequences_response"]))
processed_clean["comparable_temp"]=column_holder



# == Have all the structures expressions in the same format (starting at 0)
adjust_expression(processed_clean)

# -- Tagging sequences
processed_clean["seq_name"]=processed_clean["sequences_structure"].apply(lambda x: which_seq(x))

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
# *********** LoT Complexity ************
# ---------------------------------------

# -- We will assign a complexity of -1 for 'training' sequencess
matched_values = [complexities_allOperations_noChunk.get(value, -1) for value in processed_clean['seq_name']]
processed_clean['LoT Complexity']=matched_values
processed_clean=swap_columns(processed_clean,'LoT Complexity','geom_dist_point')


# ---------------------------------------
# ******** Duration Experiment ***********
# ---------------------------------------
# Drop training
processed_clean=processed_clean[processed_clean['state']=='main_experiment']
processed_clean.reset_index(drop=True,inplace=True)

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

print(f'{len(participants_with_token_err)} participants were dropped for making token errors on every trials.')
#
# ---- Experiment lasted more than 10 minutes
#
filtered_data = processed_clean[processed_clean['total duration experiment'] <= 10]
drop_IDs_time=len(np.unique(processed_clean['participant_ID']))-len(np.unique(filtered_data['participant_ID']))
print(f'{drop_IDs_time} participants were dropped for spending more than 10 minutes on the experiment')
processed_clean=filtered_data
processed_clean.reset_index(inplace=True, drop=True)


#
# ---- Drop first zero in interclick timings to be aligned with dataset of second experiment
#
for index,row in processed_clean.iterrows():
    processed_clean.at[index,'interclick_time']=row['interclick_time'][1:]

# ---------------------------------------
# ********** Save new dataset ***********
# ---------------------------------------

# NOTE if you get "Parser Error", it appears with Python 3.11 and higher. Use python version 3.9 not higher 
processed_clean.to_csv(f"/Users/et/Documents/UNICOG/2-Experiments/memocrush/Data/processed/experiment-1/experiment1_processed_{file_name}", index=False)
