from modules.params import *

def display_trial_counts(data):
    """
    Display the number of trials each participant has for each seq_name value,
    print the number of participants who have exactly 2 trials in every sequence,
    and print the number of participants who have exactly 2 trials in each sequence.
    
    Parameters:
    data (pd.DataFrame): The input data containing 'participant_ID' and 'seq_name' columns.
    
    Returns:
    pd.DataFrame: A DataFrame with participant_IDs as rows, seq_names as columns, and trial counts as values.
    """
    # Pivot the table to get the count of trials for each participant and seq_name
    trial_counts = data.groupby(['participant_ID', 'seq_name']).size().unstack(fill_value=0)
    
    # Calculate the number of participants who have exactly 2 trials in every sequence
    participants_with_2_trials_all_seq = (trial_counts == 2).all(axis=1).sum()
    print(f"Number of participants who have exactly 2 trials in every sequence: {participants_with_2_trials_all_seq}")
    
    # Calculate and print the number of participants who have exactly 2 trials in each sequence
    for seq_name in trial_counts.columns:
        participants_with_2_trials_per_seq = (trial_counts[seq_name] == 2).sum()
        print(f"Number of participants who have exactly 2 trials in sequence '{seq_name}': {participants_with_2_trials_per_seq}")
    

def sanity_checks(data):
    # number of trials per participant
    print('Number of main experiment trials expected: ',len(data['participant_ID'].unique())*50)


    print('Number of training trials expected: ',len(data['participant_ID'].unique())*5)


    # Participants have the same number of trials on each sequence ?
    print('\n***************************************************************************************************************************************************************************\n')
    print('CHECK THAT PARTICIPANTS HAVE THE SAME NUMBER OF TRIALS\n')
    print(data['participant_ID'].value_counts())

    print('\n***************************************************************************************************************************************************************************\n')
    # --- Count the number of rows per participant
    rows_per_participant = data['participant_ID'].value_counts()
    # --- Count the number of participants with the same number of rows
    participant_row_counts = rows_per_participant.value_counts()
    print('\nNumber of participants with the same number of trials: \n',participant_row_counts)

    print('\n***************************************************************************************************************************************************************************\n')
    display_trial_counts(data)