from modules.params import *

def prepare_dir(root_path):
    figure_path=os.path.join(root_path,'Figures')
    directories = [
        f'{figure_path}/models/regression',
        f'{figure_path}/interclick/individual/mean/singular_response',
        f'{figure_path}/length',
        f'{figure_path}/interclick/individual/median',
        f'{figure_path}/interclick/individual/mean',
        f'{figure_path}/interclick/individual/z-score',
        f'{figure_path}/heatmap/specific'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


def adjust_expression(data):
    # Iterate over the DataFrame rows using iterrows() for better readability and efficiency
    for index, row in data.iterrows():
        # Check if the first element of 'sequences_structure' starts with 1
        if row['sequences_structure'][0] == 1:
            # Subtract 1 from each element in 'sequences_structure'
            adjusted_sequence = [x - 1 for x in row['sequences_structure']]
            # Assign the adjusted sequence back to 'sequences_structure' column
            data.at[index, 'sequences_structure'] = adjusted_sequence

'''delete if the above works           
def adjust_expression(data):
    # This function serves the purpose of having the same mapping between previously tested sequences and
    # Presently tested sequences. It takes the all the sequences that starts with 1 and turns them into sequences
    # that start with zero.
    for i in range(len(data)):
        # -- Takes the sequences_structure array and adjust it if it starts with one.
        if data.at[i,'sequences_structure'][0]==1:
            holder=[]
            for k in range(len(data.at[i,'sequences_structure'])):
                holder.append(data.at[i,'sequences_structure'][k]-1)
            data.at[i,'sequences_structure']=holder    
'''    

def which_seq(seq):
    # Convert the sequence to a string
    seq_str = ''.join(map(str, seq))
    
    # Look up the sequence in the reverse mapping dictionary
    name = reverse_mapping.get(seq_str, "Training")
    
    return name

# ==> Damereau Levenshtein distance
def dl_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

def is_token_err(origin, recall):
    #Test if there is a token error: a token not presented have been reproduced or a token presented was forgotten
    return set(origin)!=set(recall)

def is_token_forg(origin,recall):
    # Adding (not in precedent experiment):
    # if True: token_forg => case of TokenErr where at least one of the tokens is in missing (has been forgotten)
    # if False: token_add => case of TokenErr where at least one of the tokens is missing
    return set(recall).issubset(set(origin))

def is_length_err(origin, recall):
    #Test if both original stimuli and response are of the same length
    return len(origin)!=len(recall)

def compare_tokens(origin, recall):
    """a function that will compare two sequences, and return an absolut mapping of the 
    reproduction (1 for token 1, 2 for token 2, 3 for token 3, 4 for token 4, 
    -1 for wrong token)
    
     Exemple :

    compare_tokens([1,2,3,1,2,3],[5,3,2,5,3,2]) => out: [-1, 3, 2, -1, 3, 2]
    compare_tokens([0,2,5,0,2,5],[0,2,5,0,2,5]) => out: [1, 2, 3, 1, 2, 3]



    Args:
        origin (arr): sequence shown to the participant
        recall (arr): sequence recalled by the participant

    Returns:
        _type_: _description_
    """
    holder=[]
    for i in range(len(recall)):
        if len(np.where(pd.unique(np.array(origin))==recall[i])[0])>0:
            holder.append(np.where(np.array(pd.unique(origin))==recall[i])[0][0])
        else:
            holder.append(-1)
    return holder

# NOTE this function was changed a bit from pilot n1 to accomodate for NaN values (especially in 'interclick_timings_before' for example). 
# When it encounters them, it turns them into empty arrays.
def str2int_dataset(df, str_columns=["participant_ID", "performance", "type"], processed=False, label_int_col=[]):
    """Turn a dataset retrieved from a csv file to a Pandas dataset that has arrays of numbers 
    instead of strings
    
    Args:
        df (Pandas DataFrame): this should be the result of logging a csv file to a pandas dataframe
        str_columns (list of strings): array that contains the labels of the columns that are to be kept into string format
        processed (bool): True if the data is already processed, False if it is raw data
        label_int_col (list of strings): columns that are already integers but stored as strings
        
    Returns:
        Pandas DataFrame: Transformed DataFrame
    """
    # -- Gather all labels into a list
    label_int = list(df.columns.values)

    # -- Pop out labels which are not to be turned into ints
    indices_to_remove = [index for index, value in enumerate(label_int) if value in str_columns]
    for j in sorted(indices_to_remove, reverse=True):
        label_int.pop(j)

    if processed:
        for label in label_int_col:
            try:
                df[label] = df[label].apply(lambda x: eval(x) if not pd.isna(x) else [])
                df[label] = df[label].apply(lambda x: list(x) if isinstance(x, tuple) else x)
            except Exception as e:
                #print(f"An error occurred while processing column {label}: {e}")
                continue

    else:
        # -- Go over the dataset and turn str into int
        rows_to_drop = []
        for label in label_int:
            for row in range(len(df)):
                try:
                    value = df.loc[row, label]
                    if pd.isna(value):
                        df.at[row, label] = []
                    else:
                        df.at[row, label] = [int(i) for i in value.split(",")]
                except (AttributeError, ValueError):
                    print(f"Error in row {row}: {label}={value}")
                    rows_to_drop.append(row)

        df.drop(rows_to_drop, inplace=True)
        print("---------------------\n")
        print("Participant number is: {}".format(len(np.unique(df["participant_ID"])) - 1))
        print(f"{len(rows_to_drop)} rows have been dropped")

    # -- Reset index
    df.reset_index(drop=True, inplace=True)
    return df

def swap_columns(df, col_label1, col_label2):
    """
    Swap two columns in a pandas DataFrame.
    
    Parameters:
        df (pandas DataFrame): The DataFrame containing the columns to be swapped.
        col_label1 (str): The label of the first column to be swapped.
        col_label2 (str): The label of the second column to be swapped.
        
    Returns:
        pandas DataFrame: The DataFrame with the columns swapped.
    """
    # Swap values between col_label1 and col_label2
    df[col_label1], df[col_label2] = df[col_label2].copy(), df[col_label1].copy()
    
    # Swap column labels
    df.rename(columns={col_label1: col_label2, col_label2: col_label1}, inplace=True)
    
    return df

    
# ---------------------------------------
# ************* Stats *******************
# ---------------------------------------

def info_survey():
    # Printss information about what each label of survey means
    print("[AGE]")
    print('\033[91m18_25\033[0m : 18-25 year old\n\033[91m25_40\033[0m : 25-40 year old\n\033[91m40_60\033[0m : 40-60 year old\n\033[91m60_\033[0m : >60 year old')
    print('----------------------------\n')
    print('[Highest Degree obtained]')
    print('\033[91mno_diplome\033[0m : Primary School\n\033[91mbrevet\033[0m : Middle School\n\033[91mBAC\033[0m : High School\n\033[91mBAC_3\033[0m : Bachelor\n\033[91mBAC_5\033[0m : Master\n\033[91mBAC_7\033[0m : PhD')
    print('----------------------------\n')
    print('[Describe your Musical experience]')
    print('\033[91mnoExp\033[0m : No experience\n\033[91mauto_1anMoins\033[0m : Self-learning for less than 1 year\n\033[91mauto_1anPlus\033[0m : Self-learning for more than 1 year\n\033[91mcvt_1anMoins\033[0m : Took classes for less than 1 year\n\033[91mcvt_1an3an\033[0m : Took classes for 1 to 3 years\n\033[91mcvt_3anPlus\033[0m : Took classes for more than 3 years\n\033[91mpro\033[0m : Professional musician')
    print('----------------------------\n')
    print('[Last Math lessons received]')
    print('\033[91mlycee\033[0m : Highschool\n\033[91mBAC_2/3\033[0m : Bachelor\n\033[91mBAC_5\033[0m : Master\n\033[91mBAC_5plus\033[0m : PhD')
    print('----------------------------\n')
    print('[How many times did you do this experiment ?]')
    print('\033[91m1\033[0m : 1 time\n\033[91m2\033[0m : 2 times\n\033[91m3\033[0m : 3 times\n\033[91m3plus\033[0m : More than 3')
    
def confidence_interval95(arr):
    
# 1. Calculate standard deviation of the array of values
    sd=np.std(arr)
    
# 2. Calculate standard error of the mean: SEM= sd/sqrt(n)
    sem=sd/np.sqrt(len(arr)/2) # pb: penses qu'il y a deux fois plus de participants
    
# 3. Using a t-table, find the t-score for the given degrees of freedom and the chosen confidence interval
    confidence_level = 0.95  # 95% confidence level
    df = len(arr)-1  # degrees of freedom
    t_score = stats.t.ppf((1 + confidence_level) / 2, df)

# 4. Calculate the Margin of Error: MOE=t-score*SEM
    moe=t_score*sem

# 5. Calculate the confidence interval: CI(95)= Mean ± MOE
    ci=(np.mean(arr)-moe,np.mean(arr)+moe)
    
    return ci

def top8(name_list, dataset):
    """ For each sequence we list the 8 most frequent responses. This function will then print, in a comparable format, the top response patterns. This function is ideal
    to have a quick look at what participants did in the experiment.

    Args:
        name_list (array of strings): typically this will be seq_name_list. An array containing the name of the sequences tested.
        dataset (pandas dataset): a dataset that has been cleaned and must contain on each row the names of the sequences as well as the answers in "comparable_temp"
    """
    for name in name_list:
        # Step 1: Filter the DataFrame
        filtered_df = dataset[dataset["seq_name"] == name]

        # Step 2: Select the "comparable_temp" column
        comparable_temp_column = filtered_df["comparable_temp"]

        # Step 3: Get frequency counts
        frequency_counts = comparable_temp_column.value_counts()

        # Step 4: Sort the frequency counts
        sorted_counts = frequency_counts.sort_values(ascending=False)

        # Step 5: Select the top 5 most frequent arrays
        top_8_frequent_arrays = sorted_counts.head(10)

        # Print the result
        print(f"correct array : {name}")
        print(dataset[dataset["seq_name"]==name]["sequences_structure"].to_numpy()[0])
        print("\nTop 8 by frequency")
        print(top_8_frequent_arrays)
        print("-----------------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------------\n")


def milliseconds2date(time):
    # Replace this variable with milliseconds obtained from Date.now() in JavaScript
    milliseconds = time

    # Convert milliseconds to seconds
    seconds = milliseconds / 1000

    # Convert to datetime object in UTC
    date_utc = datetime.utcfromtimestamp(seconds)

    # Set the timezone to Paris
    paris_timezone = pytz.timezone('Europe/Paris')
    date_paris = date_utc.replace(tzinfo=pytz.utc).astimezone(paris_timezone)

    # Format the time as 12-hour clock with AM/PM and date as DD/MM/YYYY
    formatted_date = date_paris.strftime('%I:%M%p %d/%m/%Y')

    return formatted_date

# ---------------------------------------
# *********** Error Analyses ************
# ---------------------------------------

def is_token_err(origin, recall):
    #Test if there is a token error: a token not presented have been reproduced or a token presented was forgotten
    return set(origin)!=set(recall)

def is_token_forg(origin,recall):
    # Adding (not in precedent experiment):
    # if True: token_forg => case of TokenErr where at least one of the tokens is in missing (has been forgotten)
    # if False: token_add => case of TokenErr where at least one of the tokens is missing
    return set(recall).issubset(set(origin))

def is_length_err(origin, recall):
    #Test if both original stimuli and response are of the same length
    return len(origin)!=len(recall)

def compare_tokens(origin, recall):
    """a function that will compare two sequences, and return an absolut mapping of the 
    reproduction (1 for token 1, 2 for token 2, 3 for token 3, 4 for token 4, 
    -1 for wrong token)
    
     Exemple :

    compare_tokens([1,2,3,1,2,3],[5,3,2,5,3,2]) => out: [-1, 3, 2, -1, 3, 2]
    compare_tokens([0,2,5,0,2,5],[0,2,5,0,2,5]) => out: [1, 2, 3, 1, 2, 3]



    Args:
        origin (arr): sequence shown to the participant
        recall (arr): sequence recalled by the participant

    Returns:
        _type_: _description_
    """
    holder=[]
    for i in range(len(recall)):
        if len(np.where(pd.unique(np.array(origin))==recall[i])[0])>0:
            holder.append(np.where(np.array(pd.unique(origin))==recall[i])[0][0])
        else:
            holder.append(-1)
    return holder

def fill_interclick(arr, target_length=15):
    new_arr=arr.copy()
    if len(arr)<target_length:
        for i in range(target_length-len(arr)):
            new_arr.append(0)
    return new_arr

def mean_median_interclick(name, data):
    new_arr=[]
    arr=np.asarray(data[data["seq_name"]==name]["interclick_time"])
    
    return new_arr



def num_alph(arr):
    #turn a sequence structure from a series of numbers to a series of letters
    new_arr=[]
    key=["A","B","C","D","E","F","G","H"]
    
    for item in arr:
        new_arr.append(key[item]) 
    return new_arr




#--------------------------------------------------
def plot_common_interclick(data,path,expression=True,x_axis_num=True,z_score=False):
    """Creates a plt.subplot, i.e., a big 5x5 (needs to be modified in the function when adding/removing sequences) plot of subplots.
    Each subplot contains the mean interclick timings for one sequence.
    IMPORTANTLY: only correct responses are considered.

    Args:
        data (pandas dataframe): preprocessed data and already put in a dataframe (typically data_main)
        path (str): path where plots are to be stored. This path needs to contains your_path/interclick/
        expression (bool): if True, plots will have the expression of the sequences as titles (e.g., AABBCC.AABBCC). If False, will have the name (e.g., repetition nested)
        x_axis_num (bool): if True, x-axis will be the index of the interclick (ex: ABC.ABC => 1/2/3/4/5). If False, it will be the expression of the elements (ex: ABC.ABC => AB/BC/CA/AB/BC)
        z_score (bool): if True, will plot the z-score of the interclick time instead of the absolute interclick time.s
    """
    ### Needed Variables
    # -- Constructing the x-ticks object 
    sequence_structure_str=[]
    sequence_structure_intClick=[]
    for index in range(len(seq_name_list)):
        sequence_structure_str.append(num_alph(data[data["seq_name"]==seq_name_list[index]]['sequences_structure'].iloc[0]))
        
    for k in range(len(sequence_structure_str)):
        holder=[]
        for index in range(11):
            holder.append('{a}-{b}'.format(a=sequence_structure_str[k][index],b=sequence_structure_str[k][index+1]))
        sequence_structure_intClick.append(holder)
        
    ### Holders arrays for mean of interclick timings and standard error of the mean
    all_mean_timings=[]
    sem_timings=[]
    all_z_scores=[]

    ### Collecting mean interclick-timings
    for index in range(len(seq_name_list)):
        # For each sequence:
        #
        # -- Get the interclick values of responses of the right length
        holder=data[(data['seq_name']==seq_name_list[index]) & (data['performance']=='success')]['interclick_time']
        
        if len(holder)==0:
            print(f'\033[1m{seq_name_list[index]}\033[0m: --- No correct responses were found --- ')
            continue
        else:
            print(f'\033[1m{seq_name_list[index]}\033[0m : [{len(holder)}] correct responses were considered. ({list_seq_expression[index]})')
        
        
        
        # -- Turn them in the right (numpy) format
        timings=[arr for arr in holder.to_numpy()]
        
        # -- Compute the mean
        mean_timings=np.mean(timings, axis=0)
        
        # -- Append results to the holder object
        all_mean_timings.append(mean_timings)
        
        # Generate the standard error of the mean for all the mean_timings
        sem_timings.append(np.std(timings,axis=0)/np.sqrt(len(timings)))
        
        # -- Compute z-scores
        z_scores = (mean_timings - np.mean(mean_timings)) / np.std(mean_timings)
        
        # -- Append results to the holder object
        all_z_scores.append(z_scores)

        
    ### Plotting 
    # Create the figure and axes
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

    plot_index=0 #this is the index for all_mean_timings which differs in the length and increments if will take from index of plots
    name_index=0 #tracks the name of the sequence to display
    for index, ax in enumerate(axes.flat):
        
        # If there's no correct response for this particular sequence, remove the axis
        if len(data[(data['seq_name']==seq_name_list[name_index]) & (data['performance']=='success')]['interclick_time'])==0:
            name_index+=1
            fig.delaxes(ax)
            continue
        

        ax.vlines(x=range(0, 11), ymin=np.min(all_mean_timings) - 50, ymax=np.max(all_mean_timings) + 100, colors='black', ls='--', lw=1)
        
        # -- Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
        if x_axis_num:
            ax.set_xticks(ticks=range(0,11), labels=range(1,12))
        else:
            ax.set_xticks(ticks=[i-0.5 for i in range(0,12)], labels=[i for i in alpha_seq_expression[name_index]])
            ax.set_xlim(xmin=-1, xmax=11)
            
        
        if expression:
            ax.set_title(f'{list_seq_expression[name_index]}')
        else:
            ax.set_title(f'{seq_name_list[name_index]}')
        ax.errorbar(range(11), all_mean_timings[plot_index], yerr=sem_timings[plot_index], fmt='o', capsize=5, capthick=2, color="black")
        ax.plot(range(11), all_mean_timings[plot_index])
        ax.set_ylim(ymin=300, ymax=np.max(all_mean_timings) + 100)  # Set y-axis limits
        # -- GO to next iteration
        plot_index+=1
        name_index+=1

    # Save and show the plot
    plt.savefig(f'{path}/mean_interclicks_subplots.png', bbox_inches='tight', dpi=800)
    plt.show()
    # Close the current figure window
    plt.close()
    
    if z_score:
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

        plot_index=0 #this is the index for all_mean_timings which differs in the length and increments if will take from index of plots
        name_index=0 #tracks the name of the sequence to display
        for index, ax in enumerate(axes.flat):
            
            # If there's no correct response for this particular sequence, remove the axis
            if len(data[(data['seq_name']==seq_name_list[name_index]) & (data['performance']=='success')]['interclick_time'])==0:
                name_index+=1
                fig.delaxes(ax)
                continue
            

            ax.vlines(x=range(0, 11), ymin=np.min(all_z_scores) - 5, ymax=np.max(all_z_scores) + 5, colors='black', ls='--', lw=1)
            
            # -- Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
            if x_axis_num:
                ax.set_xticks(ticks=range(0,11), labels=range(1,12))
            else:
                ax.set_xticks(ticks=[i-0.5 for i in range(0,12)], labels=[i for i in alpha_seq_expression[name_index]])
                ax.set_xlim(xmin=-1, xmax=11)
                
            
            if expression:
                ax.set_title(f'{list_seq_expression[name_index]}')
            else:
                ax.set_title(f'{seq_name_list[name_index]}')
            ax.plot(range(11), all_z_scores[plot_index])
            ax.axhline(y=0, color='orange')
            ax.set_ylim(ymin=-4, ymax=4) 
            ax.plot(range(11), all_z_scores[plot_index],'o',markersize=7,color='black')
            # -- GO to next iteration
            plot_index+=1
            name_index+=1

        # Save and show the plot
        plt.savefig(f'{path}/interclick/z-score_interclicks_subplots.png', bbox_inches='tight', dpi=800)
        plt.show()
        
        # Close the current figure window
        plt.close()
        
    
    
    
    
#--------------------------------------------------
def plot_individual_interclick(data,path,expression=True,x_axis_num=False,z_score=True):
    """Creates one plot per sequence.
    Each plot contains the mean interclick timings for one sequence.
    IMPORTANTLY: only correct responses are considered.

    Args:
        data (pandas dataframe): preprocessed data and already put in a dataframe (typically data_main)
        path (str): path where plots are to be stored. This path needs to contains your_path/interclick/individual/
        expression (bool): if True, plots will have the expression of the sequences as titles (e.g., AABBCC.AABBCC). If False, will have the name (e.g., repetition nested)
        x_axis_num (bool): if True, x-axis will be the index of the interclick (ex: ABC.ABC => 1/2/3/4/5). If False, it will be the expression of the elements (ex: ABC.ABC => AB/BC/CA/AB/BC)
        z_score (bool): if True, will plot the z-score of the interclick time instead of the absolute interclick time.s
    """
    ### Needed Variables
    # -- Constructing the x-ticks object 
    sequence_structure_str=[]
    sequence_structure_intClick=[]
    for index in range(len(seq_name_list)):
        sequence_structure_str.append(num_alph(data[data["seq_name"]==seq_name_list[index]]['sequences_structure'].iloc[0]))
        
    for k in range(len(sequence_structure_str)):
        holder=[]
        for index in range(11):
            holder.append('{a}-{b}'.format(a=sequence_structure_str[k][index],b=sequence_structure_str[k][index+1]))
        sequence_structure_intClick.append(holder)
        
    ### Holders arrays for mean of interclick timings and standard error of the mean
    all_mean_timings=[]
    all_z_scores=[]
    sem_timings=[]

    ### Collecting mean interclick-timings
    for index in range(len(seq_name_list)):
        # For each sequence:
        #
        # -- Get the interclick values of responses of the right length
        holder=data[(data['seq_name']==seq_name_list[index]) & (data['performance']=='success')]['interclick_time']

        # -- If there's no correct response for this particular sequence go to next iteration
        if len(holder)==0:
            print(f'\033[1m{seq_name_list[index]}\033[0m:  --- No correct responses were found --- ')
            continue
        else:
            print(f'\033[1m{seq_name_list[index]}\033[0m: [{len(holder)}] correct responses were considered ({list_seq_expression[index]}).')
        
        # -- Turn them in the right (numpy) format
        timings=[arr for arr in holder.to_numpy()]
        
        # -- Compute the mean
        mean_timings=np.mean(timings, axis=0)
        
        # -- Compute z-scores
        z_scores = (mean_timings - np.mean(mean_timings)) / np.std(mean_timings)
        
        # -- Append results to the holder object
        all_z_scores.append(z_scores)
        all_mean_timings.append(mean_timings)
        
        # Generate the standard error of the mean for all the mean_timings
        sem_timings.append(np.std(timings,axis=0)/np.sqrt(len(timings)))

    ### Plotting 

    plot_index=0 #this is the index for all_mean_timings
    for index in range(len(seq_name_list)):
    # *** Mean interclicks
    
        # -- If there's no correct response for this particular sequence go to next iteration
        if len(data[(data['seq_name']==seq_name_list[index]) & (data['performance']=='success')]['interclick_time'])==0:
            continue

        plt.vlines(x=range(0,11), ymin=np.min(all_mean_timings)-50, ymax=np.max(all_mean_timings)+100, colors='black', ls='--', lw=1)
        # -- Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
        if x_axis_num:
            plt.xticks(ticks=range(0,11), labels=range(1,12))
        else:
            plt.xticks(ticks=[i-0.5 for i in range(0,12)], labels=[i for i in alpha_seq_expression[index]])
            plt.xlim(xmin=-1, xmax=11)
            
        if expression:
            plt.title(f'{seq_name_list[index]}: {list_seq_expression[index]}',pad=padding_size,fontsize=title_size)
        else:
            plt.title(f'Mean Interclick times: {seq_name_list[index]}',pad=padding_size,fontsize=title_size)
        plt.errorbar(range(11), all_mean_timings[plot_index], yerr=sem_timings[plot_index], fmt='o', capsize=5, capthick=2, color="black")
        plt.plot(range(11),all_mean_timings[plot_index])
        #plt.ylim(ymin=300, ymax=np.max(all_mean_timings)+100)  # Set y-axis limits
        plt.ylim(ymin=350, ymax=800)  # Set y-axis limits
        plt.savefig(f'{path}/interclick/individual/mean/mean_interclicks_subplots_{seq_name_list[index]}.png', bbox_inches='tight', dpi=800)    
        plt.close()
        
        if z_score:
        # *** z-scores
            # -- If there's no correct response for this particular sequence go to next iteration
            if len(data[(data['seq_name']==seq_name_list[index]) & (data['performance']=='success')]['interclick_time'])==0:
                continue

            plt.vlines(x=range(0,11), ymin=np.min(all_z_scores)-5, ymax=np.max(all_z_scores)+5, colors='black', ls='--', lw=1)
            # -- Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
            if x_axis_num:
                plt.xticks(ticks=range(0,11), labels=range(1,12))
            else:
                plt.xticks(ticks=[i-0.5 for i in range(0,12)], labels=[i for i in alpha_seq_expression[index]])
                plt.xlim(xmin=-1, xmax=11)
                
            if expression:
                plt.title(f'Z-scores interclicks: {list_seq_expression[index]}',pad=padding_size, fontsize=title_size)
            else:
                plt.title(f'Z-scores interclicks: {seq_name_list[index]}',pad=padding_size,fontsize=title_size)
            plt.plot(range(11),all_z_scores[plot_index])
            plt.plot(range(11),all_z_scores[plot_index],'o', markersize=7,color='black')
            plt.axhline(y=0, color='orange')
            plt.ylim(ymin=-4, ymax=4) 
            plt.savefig(f'{path}/interclick/individual/z-score/z-score_subplots_{seq_name_list[index]}.png', bbox_inches='tight', dpi=800)
            
            plt.show()
            # Close the current figure window
            plt.close()
        plot_index+=1





#--------------------------------------------------      
def plot_error_rate(data,path):
    # == step1 == Create y axis: a list with rate of success per sequence
    success_rate=[]
    plt.rcParams['figure.facecolor'] = 'white'

    # New name_list with count of included sequences for each seq type in data_main (all main)

    count_seq_name_list=[]
    count_list_seq_expression=[]
    error_rates_all=[]
    for i in range(len(seq_name_list)):
        count_seq_name_list.append(seq_name_list[i]+" ({})".format(data[data["seq_name"]==seq_name_list[i]].count().iloc[0]))
        count_list_seq_expression.append(list_seq_expression[i]+" ({})".format(data[data["seq_name"]==seq_name_list[i]].count().iloc[0]))

    for i in range(len(seq_name_list)):
        nb_success=len(data[(data["seq_name"]==seq_name_list[i])&(data["performance"]=="success")])
        nb_total=len(data[data["seq_name"]==seq_name_list[i]])
        success_rate.append(100*nb_success/nb_total)
        error_rates_all.append(100-success_rate[i])

    colors = plot_colors

    plt.rcParams['figure.facecolor'] = 'white'
    fig,ax=plt.subplots(figsize=plot_figsize)
    ax.barh(np.arange(len(count_seq_name_list)),error_rates_all, align="center", color=colors)
    ax.set_yticks(np.arange(len(count_seq_name_list)))
    ax.set_yticklabels(seq_name_list)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    sec_axis=ax.secondary_yaxis("right")
    sec_axis.set_yticks(np.arange(len(count_seq_name_list)))
    sec_axis.set_yticklabels(list_seq_expression)
    sec_axis.tick_params(axis='y', labelsize=14)
    ax.set_xlim(0,100)
    ax.set_xlabel("Error Rate (%)", fontsize=14, labelpad=14)
    ax.set_title("Total Error Rate", fontsize=title_size, pad=padding_size)
    plt.savefig(f'{path}/errorRates_allSequences.jpg', 
                bbox_inches='tight', dpi=800)


#--------------------------------------------------
def plot_token_error(data,path):
    all_token_err=[]
    for name in seq_name_list:
        all_token_err.append(np.sum(data[data["seq_name"]==name]["TokenErr"]))

    fig, ax=plt.subplots(figsize=plot_figsize)
    ax.set_yticks(range(len(seq_name_list)))
    ax.set_yticklabels(seq_name_list)
    ax.set_xticks(range(100))
    ax.invert_yaxis()
    ax.barh(range(len(seq_name_list)),all_token_err)
    plt.title('Token Error (absolute number)', fontsize=title_size,pad=padding_size)
    plt.savefig(f'{path}/token_error_all.jpg', 
                bbox_inches='tight', dpi=800)



#--------------------------------------------------
def plot_median_dl(data,path):
    #Create an array that will contain the arrays of DL_values for each sequence
    seq_distance_DL=[]
    #Loop over the names of the sequences to fill the previous array
    for seq in seq_name_list:
        seq_distance_DL.append(np.array(data[data["seq_name"]=="{}".format(seq)]["distance_dl"]))
    #Create an array that holds the median values of distance DL for each sequence
    seq_distance_DL_median=[np.median(arr) for arr in seq_distance_DL]
    #Create an array that holds the mean values of distance DL for each sequence
    seq_distance_DL_mean=[np.mean(arr) for arr in seq_distance_DL]

    #Draw the figure
    fig,ax=plt.subplots(figsize=plot_figsize)
    colors = plot_colors
    ax.barh(np.arange(len(seq_name_list)),seq_distance_DL_median, align="center", color=plot_colors)
    ax.set_yticks(np.arange(len(seq_name_list)))
    ax.set_yticklabels(seq_name_list)
    ax.invert_yaxis()
    #ax.secondary_yaxis("right")
    plt.title("Median Damereau-Levenshtein Distance",size=title_size, pad=padding_size)
    #ax.set_xlabel("Median DL value", size=25)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    sec_axis=ax.secondary_yaxis("right")
    sec_axis.set_yticks(np.arange(len(seq_name_list)))
    sec_axis.set_yticklabels(list_seq_expression)
    sec_axis.tick_params(axis='y', labelsize=14)
    plt.savefig(f'{path}/median_dl_all.jpg', 
                bbox_inches='tight', dpi=800)



#--------------------------------------------------
def plot_mean_dl(data,path,save=True):
    # Participants IDs
    IDs=[data["participant_ID"][0]]
    for i in range(len(data)-1):
        if data["participant_ID"][i] not in IDs:
            IDs.append(data["participant_ID"][i])
    
    # Calculate the mean distance_DL for each sequence per participant
    temp_distDL_perParticipant = []

    for name in seq_name_list:
        new_arr = []
        for participant in IDs:
            subset = data[(data["participant_ID"] == participant) & (data["seq_name"] == name)]
            mean_distance_dl = np.nanmean(subset["distance_dl"])  # Use np.nanmean to handle NaN values
            new_arr.append(mean_distance_dl)
        temp_distDL_perParticipant.append(new_arr)

    # Convert the list of lists into a 2D NumPy array
    distDL_perParticipant = np.array(temp_distDL_perParticipant)

    # Calculate confidence intervals
    CI_meanDL = [confidence_interval95(dist) for dist in distDL_perParticipant]
    all_sem = [stats.sem(dist, nan_policy='omit') for dist in distDL_perParticipant]

    # Extract the lower and upper bounds of the confidence interval
    lower_bound = np.array([item[0] for item in CI_meanDL])
    upper_bound = np.array([item[1] for item in CI_meanDL])

    # Plotting
    plt.rcParams["figure.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=(10,8))
    # ax.errorbar(np.nanmean(distDL_perParticipant, axis=1), np.arange(len(seq_name_list)), fmt='o',
    #         xerr=all_sem, capsize=5)

    for i in range(len(seq_name_list)):
        ax.errorbar(np.nanmean(distDL_perParticipant[i]), i, fmt=figure_format_points[i],
                    xerr=all_sem[i], capsize=5, color=color_structure_control[i], markersize=8)

    ax.set_yticks(np.arange(len(seq_name_list)))
    ax.set_yticklabels(seq_name_list, fontsize=14)

    ax.invert_yaxis()
    ax.set_xlabel("Mean Damerau-Levenshtein Distance", fontsize=title_size, labelpad=padding_size)

    
    

    # sec_axis = ax.secondary_yaxis("right")
    # sec_axis.set_yticks(np.arange(len(seq_name_list)))
    # sec_axis.set_yticklabels(list_seq_expression, fontsize=12)

    # ax.tick_params(axis='x', labelsize=16)
    # ax.tick_params(axis='y', labelsize=16)
    # plt.title("Mean Damerau-Levenshtein Distance", fontsize=title_size, pad=padding_size)
    #ax.set_xlabel("Mean DL value", fontsize=14, labelpad=14)
    if save:
        plt.savefig(f'{path}/mean_dl_all.jpg', 
                    bbox_inches='tight', dpi=800)
    else:
        plt.show()


#--------------------------------------------------
def plot_mean_dl_structure(data,path):
    # Participants IDs
    IDs=[data["participant_ID"][0]]
    for i in range(len(data)-1):
        if data["participant_ID"][i] not in IDs:
            IDs.append(data["participant_ID"][i])
            
    # Calculate the mean distance_DL for each sequence per participant
    temp_distDL_perParticipant = []

    for name in seq_name_list:
        new_arr = []
        for participant in IDs:
            subset = data[(data["participant_ID"] == participant) & (data["seq_name"] == name)]
            mean_distance_dl = np.nanmean(subset["dl_structure"])  # Use np.nanmean to handle NaN values
            new_arr.append(mean_distance_dl)
        temp_distDL_perParticipant.append(new_arr)

    # Convert the list of lists into a 2D NumPy array
    distDL_perParticipant = np.array(temp_distDL_perParticipant)

    # Calculate confidence intervals
    CI_meanDL = [confidence_interval95(dist) for dist in distDL_perParticipant]
    all_sem = [stats.sem(dist, nan_policy='omit') for dist in distDL_perParticipant]

    # Extract the lower and upper bounds of the confidence interval
    lower_bound = np.array([item[0] for item in CI_meanDL])
    upper_bound = np.array([item[1] for item in CI_meanDL])

    # Plotting
    plt.rcParams["figure.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=plot_figsize)
    ax.barh(np.arange(len(seq_name_list)), np.nanmean(distDL_perParticipant, axis=1),
            xerr=all_sem, capsize=5, align="center", color=plot_colors)

    ax.set_yticks(np.arange(len(seq_name_list)))
    ax.set_yticklabels(seq_name_list, fontsize=14)
    ax.invert_yaxis()

    sec_axis = ax.secondary_yaxis("right")
    sec_axis.set_yticks(np.arange(len(seq_name_list)))
    sec_axis.set_yticklabels(list_seq_expression, fontsize=12)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.title("Mean Structure Distance DL - All Sequences", fontsize=title_size, pad=padding_size)
    #ax.set_xlabel("Mean DL value", fontsize=14, labelpad=14)

    plt.savefig(f'{path}/structure_mean_dl_all.jpg', 
                bbox_inches='tight', dpi=800)

#--------------------------------------------------
def plot_heatmap(data,path,show=False):
    #Variables
    #max_elements_sequence = np.max(data_main['sequences_response'].apply(len))
    max_elements_sequence=16
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Loop over seq_name_list
    for index, seq_name in enumerate(seq_name_list):
        # Build the heatmap object
        holder_heatmap = []
        all_comparable_temp = data[data['seq_name'] == seq_name]['comparable_temp'].to_numpy()
        # Cut excess elements from arrays with more than max_elements_sequence
        all_comparable_temp = [arr[:max_elements_sequence] for arr in all_comparable_temp]
        # These arrays must be filled with zeros to all have the same number of elements == max_elements_sequence
        all_comparable_temp = [np.pad(arr, (0, max_elements_sequence-len(arr)), mode='constant', constant_values=-2) for arr in all_comparable_temp]

        for token in np.unique(all_comparable_temp):
            holder = [sum(1 for response in all_comparable_temp if response[position] == token) for position in range(max_elements_sequence)]
            holder_heatmap.append(holder)
        holder_heatmap = holder_heatmap[::-1][:-1]
        
        # -- If token errors
        if -1 in np.unique(all_comparable_temp):
            heatmap_holder_shifted = np.concatenate((holder_heatmap[-1:], holder_heatmap[:-1]))
        # -- If no token errors
        else:
            heatmap_holder_shifted = np.concatenate((holder_heatmap, np.zeros((1, max_elements_sequence))))
            heatmap_holder_shifted = np.concatenate((heatmap_holder_shifted[-1:], heatmap_holder_shifted[:-1]))

        # Transform the heatmap object to be displayed in percent of initial responses
        column_totals = np.sum(holder_heatmap, axis=0)
        heatmap_holder_percent = np.round(heatmap_holder_shifted / column_totals[0] * 100).astype(int)

        # Draw the figure
        y_labels = list(alphabet[:(len(heatmap_holder_percent)-1)])
        y_labels.append('Token Error')
        y_labels.reverse()

        plt.rcParams['figure.facecolor'] = '#f1f3f5'
        plt.figure(figsize=(15, 4)) 
        sns.heatmap(heatmap_holder_percent, annot=True, yticklabels=y_labels, xticklabels=range(max_elements_sequence), fmt='g', linewidth=0.5, cmap="Purples")
        plt.title(f"Reproduction patterns {seq_name} (as%): {list_seq_expression[index]}", fontsize=title_size, pad=padding_size)
        plt.xlabel("Ordinal Rank")
        plt.yticks(rotation=0)

        plt.savefig(f'{path}/heatmap/heatmap_{seq_name}.jpg', bbox_inches='tight', dpi=800)
        if(show):
            plt.show()
        # Close the current figure window
        plt.close()

def plot_specific_heatmap(data,path,description,show=True,save=False):
    """Plot the response patterns as a heatmap for a particular set of data. 

    Args:
        data (pandas dataframe): sub-selection of the main data-set to observe a specific kind of answers patterns. example: data_main[(data_main['seq_name']=="control sub-programs 2")&(data_main['comparable_temp'].apply(lambda x: len(x)==10))]
        path (str): path where the plot will be saved
        description (str): Will be added at the end of the saved png name. Description of the criteria that selected the dataset. Also serves as a title for the figure. Example: controlSubPrograms2_lengthOf10
        show (bool, optional): If True, shows the plot, if False, saves only. Defaults to True.
        save (bool, optional): If True, saves the plot, if False, show only. Defaults to False.
    """
    #Variables
    #max_elements_sequence = np.max(data_main['sequences_response'].apply(len))
    max_elements_sequence=18
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Build the heatmap object
    holder_heatmap = []
    all_comparable_temp = data['comparable_temp'].to_numpy()
    # -- Cut excess elements from arrays with more than max_elements_sequence
    all_comparable_temp = [arr[:max_elements_sequence] for arr in all_comparable_temp]
    # -- These arrays must be filled with zeros to all have the same number of elements == max_elements_sequence
    all_comparable_temp = [np.pad(arr, (0, max_elements_sequence-len(arr)), mode='constant', constant_values=-2) for arr in all_comparable_temp]
    
    for token in np.unique(all_comparable_temp):
            holder = [sum(1 for response in all_comparable_temp if response[position] == token) for position in range(max_elements_sequence)]
            holder_heatmap.append(holder)
    holder_heatmap = holder_heatmap[::-1][:-1]
    
    # -- If token errors
    if -1 in np.unique(all_comparable_temp):
        heatmap_holder_shifted = np.concatenate((holder_heatmap[-1:], holder_heatmap[:-1]))
    # -- If no token errors
    else:
        heatmap_holder_shifted = np.concatenate((holder_heatmap, np.zeros((1, max_elements_sequence))))
        heatmap_holder_shifted = np.concatenate((heatmap_holder_shifted[-1:], heatmap_holder_shifted[:-1]))

    # Transform the heatmap object to be displayed in percent of initial responses
    column_totals = np.sum(holder_heatmap, axis=0)
    heatmap_holder_percent = np.round(heatmap_holder_shifted / column_totals[0] * 100).astype(int)

    # Draw the figure
    y_labels = list(alphabet[:(len(heatmap_holder_percent)-1)])
    y_labels.append('Token Error')
    y_labels.reverse()

    plt.rcParams['figure.facecolor'] = '#f1f3f5'
    plt.figure(figsize=(15, 4)) 
    sns.heatmap(heatmap_holder_percent, annot=True, yticklabels=y_labels, xticklabels=range(max_elements_sequence), fmt='g', linewidth=0.5, cmap="Purples")
    plt.title(f"Reproduction patterns {description} (as%). Population : {len(all_comparable_temp)}", fontsize=title_size, pad=padding_size)
    plt.xlabel("Ordinal Rank")
    plt.yticks(rotation=0)

    if(save):
        plt.savefig(f'{path}/heatmap/specific/heatmap_{description}.jpg', bbox_inches='tight', dpi=800)
    if(show):
        plt.show()
    # Close the current figure window
    plt.close()
    
#--------------------------------------------------

def deletion_error(arr,index):
    return np.delete(arr,index)

def plot_deletion_errors(data, path):
    """Plot the error of deletion heatmaps per indexes and per groups of items.
    
    In deletion per index, counts the percentage of responses with only the given index deleted over all the mistakes
    that were made (independently of the type of mistake).
    Example:  123.123.123.123 => [index=1] will count the number of 13.123.123.123
    
    In deletion per group, counts the percentage of responses with only the given group of indexes deleted over all the mistakes
    that were made (independently of the type of mistake).
    Example: 12.12.12.12.12.12 => [index: (2,3)] will count the number of 12.12.12.12.12.12 (yes it can be redundant)

    Args:
        data (_type_): _description_
        path (_type_): _description_
    """
    seq_name, seq_expressions=[i for i in real_mapping.keys()], [[int(char) for char in s] for s in real_mapping.values()]
    holder_all=[]
    error_number_all=[]
    for i in range(len(seq_name)):
        holder_sequence_deletion=[]
        for k in range(len(seq_expressions[i])):
            holder_sequence_deletion.append(len(data[(data['seq_name']==seq_name[i])&(data['comparable_temp'].apply(lambda x:np.array_equal(x,deletion_error(seq_expressions[i],k))))]))
            error_number_all.append(len(data[(data['seq_name']==seq_name[i])&(data['performance']=='fail')]))
        holder_all.append(holder_sequence_deletion)
    holder_all=np.array(holder_all)
    percentages = [(deletion/error)*100 for deletion,error in zip(holder_all,error_number_all)]
    # Draw the figure
    sns.heatmap(percentages,
                yticklabels=seq_name,
                fmt='g', 
                linewidth=0.5, 
                xticklabels=range(1,13),
                cmap="Purples")
    plt.xlabel('Deleted element index')
    plt.title('Error percentage. Per Index.')
    plt.savefig(f'{path}/deletion_errors_index.jpg', bbox_inches='tight', dpi=800)
    plt.show()
    
    all_groups=[]
    for num in range(6):
        p_eval=num
        all_groups.append([i for i in range(num*2,(num+1)*2)])
    for num in range(4):
        p_eval=num
        all_groups.append([i for i in range(num*3,(num+1)*3)])
    for num in range(3):
        p_eval=num
        all_groups.append([i for i in range(num*4,(num+1)*4)])

    holder_all=[]

    for i in range(len(seq_name)):
        holder_sequence_deletion=[]
        for arr in all_groups:
            holder_sequence_deletion.append(len(data[(data['seq_name']==seq_name[i])&(data['comparable_temp'].apply(lambda x:np.array_equal(x,deletion_error(seq_expressions[i],arr))))]))
        holder_all.append(holder_sequence_deletion)
    holder_all=np.array(holder_all)
    percentages = [(deletion/error)*100 for deletion,error in zip(holder_all,error_number_all)]
    
    # Draw the figure
    sns.heatmap(percentages,
                yticklabels=seq_name,
                fmt='g', 
                linewidth=0.5, 
                xticklabels=all_groups,
                cmap="Purples")
    plt.title('Error percentage. Deletion of whole groups')
    plt.xlabel('Deleted elements indexes')
    plt.savefig(f'{path}/deletion_errors_groups.jpg', bbox_inches='tight', dpi=800)

#--------------------------------------------------
def plot_regression(data, path, dl_distance=True, complexity_measure='LoT Complexity', labels=False):
    """ Plots the regression with seaborn. But also train an OLS model to output key components.

    Args:
        data (pandas dataFrame): dataframe containing 'LOT complexity', distance_dl. We recommand to use a no_training dataFrame (without the training sequences)
        path (str): root path for the saved plots.
        dl_distance (bool, optional): If True, will do the regression with y as the dl_distance. If False, uses the error rate instead. Defaults to True.
        complexity_measure (str, optional): Name of the column to consider for the complexity values of the sequences.
        labels (bool, optional): If True, will have the name of the sequences next to their mean. If False, it just draws the means. Default to False.
    """
    #NOTE this needs to be changed if we test other complexity sets
    complexities_ordered=[i for i in complexities_allOperations_noChunk.values()] 
    if(dl_distance):
        # -- Training a linear Regression OLS model
        # Independant Variable: LoT Complexity
        X=data[[f'{complexity_measure}']]

        # Dependant Variable: DL Distance
        y=data['distance_dl']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a Linear Regression model
        model = LinearRegression()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # -- Printing information about the correlation of our two variables
        pearson_corr, p_value = pearsonr(X.squeeze(), y)
        print("Pearson's r:", pearson_corr)
        print("p-value:", p_value)
        print('------------------')
        # Get the y-intercept (intercept) and slope coefficients (coefficients)
        intercept = model.intercept_
        coefficients = model.coef_

        print("Y-intercept (intercept):", intercept)
        print("Slope coefficient (coefficients):", coefficients)
        print('Accuracy',model.score(X_test,y_test))

        # -- Plotting the linear regression
        IDs=[data["participant_ID"][0]]
        for i in range(len(data)-1):
            if data["participant_ID"][i] not in IDs:
                IDs.append(data["participant_ID"][i])

        # Calculate the mean distance_DL for each sequence per participant
        temp_distDL_perParticipant = []

        for name in seq_name_list:
            new_arr = []
            for participant in IDs:
                subset = data[(data["participant_ID"] == participant) & (data["seq_name"] == name)]
                mean_distance_dl = np.nanmean(subset["distance_dl"])  # Use np.nanmean to handle NaN values
                new_arr.append(mean_distance_dl)
            temp_distDL_perParticipant.append(new_arr)

        # Convert the list of lists into a 2D NumPy array
        distDL_perParticipant = np.array(temp_distDL_perParticipant)

        # Calculate confidence intervals
        CI_meanDL = [confidence_interval95(dist) for dist in distDL_perParticipant]
        all_sem = [stats.sem(dist, nan_policy='omit') for dist in distDL_perParticipant]

        # # Extract the lower and upper bounds of the confidence interval
        # lower_bound = np.array([item[0] for item in CI_meanDL])
        # upper_bound = np.array([item[1] for item in CI_meanDL])

        #Get all means
        all_distDL_means=np.nanmean(distDL_perParticipant, axis=1)

        sns.set_style("white")
        # Plot the data and regression line with confidence intervals
        sns.regplot(data=data, x=f'{complexity_measure}', y='distance_dl',fit_reg=True, scatter=False, ci=95, color='black', line_kws={"color": "black"})

        # .squeeze() Removes any single-dimensional entries, essentially [..] 
        # [..] converting a DataFrame with a single column into a Series
        # Plot the mean distance_DL for each sequence per participant with error bars for confidence intervals
        plt.errorbar(x=complexities_ordered, y=all_distDL_means, yerr=all_sem, fmt='o', ecolor='black',elinewidth=1,color='firebrick',markeredgecolor='black')
        if(labels):
            spread_factor=0.3
            for i, (x, y) in enumerate(zip(complexities_ordered, all_distDL_means)):
                spread_y = random.uniform(0, spread_factor)
                plt.text(x - 1, y+spread_factor-spread_y, f'{seq_name_list[i]}', fontsize=8)  # Adjust the offset and font size as needed

            plt.xlabel('Language of Thought Complexity')
            plt.ylabel('DL Distance')
            plt.title('Linear Regression', fontsize=title_size, pad=padding_size)
            # Save and show the plot
            plt.savefig(f'{path}/models/regression/labels_dl_complexity_OLSregression.png', bbox_inches='tight', dpi=800)
            plt.show()
            # Close the current figure window
            plt.close() 

        plt.xlabel('Language of Thought Complexity')
        plt.ylabel('DL Distance')
        plt.title('Linear Regression', fontsize=title_size, pad=padding_size)
        # Save and show the plot
        plt.savefig(f'{path}/models/regression/dl_complexity_OLSregression.png', bbox_inches='tight', dpi=800)
        plt.show()
        # Close the current figure window
        plt.close() 
    else:
        # -- Running the same thing but with the error rate instead of the DL distance
        success_rate=[]
        count_seq_name_list=[]
        count_list_seq_expression=[]
        error_rates_all=[]
        for i in range(len(seq_name_list)):
            count_seq_name_list.append(seq_name_list[i]+" ({})".format(data[data["seq_name"]==seq_name_list[i]].count().iloc[0]))
            count_list_seq_expression.append(list_seq_expression[i]+" ({})".format(data[data["seq_name"]==seq_name_list[i]].count().iloc[0]))

        for i in range(len(seq_name_list)):
            nb_success=len(data[(data["seq_name"]==seq_name_list[i])&(data["performance"]=="success")])
            nb_total=len(data[data["seq_name"]==seq_name_list[i]])
            success_rate.append(100*nb_success/nb_total)
            error_rates_all.append(100-success_rate[i])
        # Independant Variable: LoT Complexity
        X=np.array([i for i in complexities_allOperations_version.values()]).reshape(-1, 1)

        # Dependant Variable: DL Distance
        y=error_rates_all

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Linear Regression model
        model = LinearRegression()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        pearson_corr, p_value = pearsonr(X.squeeze(), y)
        print("Pearson's r:", pearson_corr)
        print("p-value:", p_value)
        print('------------------')
        # Get the y-intercept (intercept) and slope coefficients (coefficients)
        intercept = model.intercept_
        coefficients = model.coef_

        print("Y-intercept (intercept):", intercept)
        print("Slope coefficient (coefficients):", coefficients)
        
        # -- Plotting the linear regression
        IDs=[data["participant_ID"][0]]
        for i in range(len(data)-1):
            if data["participant_ID"][i] not in IDs:
                IDs.append(data["participant_ID"][i])

        sns.set_style("white")
        # Plot the data and regression line with confidence intervals
        dict_error_rate=dict(zip(seq_name_list,error_rates_all))
        error_rates_regression_plot=[dict_error_rate[i] for i in data['seq_name']]
        sns.regplot(data=data, x='LoT Complexity', y=error_rates_regression_plot,fit_reg=True, scatter=False, ci=95, color='black', line_kws={"color": "black"})

        # -- Compute confidence interval of error rates
        nb_trials=len(data[data['seq_name']==seq_name_list[0]])
        success_counts=[]
        for name in seq_name_list:
            success_counts.append(len(data[(data['performance']=='success')&(data['seq_name']==name)]))

        # Compute confidence intervals for each participant's error rate
        # cis = []
        # for index in range(len(success_counts)):
        #     ci = proportion_confint(success_counts[index], nb_trials, alpha=0.05, method='normal')
        #     cis.append(ci)
        # Compute the lower and upper bounds of the error bars
        # lower_bounds = [ci[0] for ci in cis]
        # upper_bounds = [ci[1] for ci in cis]
        # yerr = [[error_rates_all[i] - lower_bounds[i] for i in range(len(error_rates_all))],
        #         [upper_bounds[i] + error_rates_all[i] for i in range(len(error_rates_all))]]
        #plt.errorbar(x=complexities_ordered, y=error_rates_all, yerr=yerr, fmt='o', ecolor='black',elinewidth=1,color='firebrick',markeredgecolor='black')
        
            
        # Plot the error rate for each sequence per participant with error bars for confidence intervals
        plt.errorbar(x=complexities_ordered, y=error_rates_all, fmt='o', ecolor='black',elinewidth=1,color='firebrick',markeredgecolor='black')
        if(labels):
            spread_factor=0.3
            for i, (x, y) in enumerate(zip(complexities_ordered, all_distDL_means)):
                spread_y = random.uniform(0, spread_factor)
                plt.text(x - 1, y+spread_factor-spread_y, f'{seq_name_list[i]}', fontsize=8)  # Adjust the offset and font size as needed

            plt.xlabel('Language of Thought Complexity')
            plt.ylabel('DL Distance')
            plt.title('Linear Regression', fontsize=title_size, pad=padding_size)
            # Save and show the plot
            plt.savefig(f'{path}/models/regression/labels_dl_complexity_OLSregression.png', bbox_inches='tight', dpi=800)
            plt.show()
            # Close the current figure window
            plt.close() 

        plt.xlabel('Language of Thought Complexity')
        plt.ylabel('Error Rate')
        plt.title('Linear Regression', fontsize=title_size, pad=padding_size)
        # Save and show the plot
        plt.savefig(f'{path}/models/regression/errorRate_complexity_OLSregression.png', bbox_inches='tight', dpi=800)
        plt.show()
        # Close the current figure window
        plt.close() 

#--------------------------------------------------

# Plotting
def plot_mean_error_rates(data,path):
    # Holder Objects
    all_error_rates_seq=[]
    mean_per_participant_error_rates=[]
    sem_per_participant_error_rates=[]

    # For each sequence
    for name in seq_name_list:
        error_rates_seq=[]
        # -- For each participant
        for IDs in data['participant_ID'].unique():
            # -- Total success for the sequence
            nb_success=len(data[(data['participant_ID']==IDs)&(data['seq_name']==name)&(data['performance']!='success')])
            # -- number of trials
            nb_trials=len(data[(data['participant_ID']==IDs)&(data['seq_name']==name)])
            # -- Divided by number of trials (2)
            error_rates_seq.append(100*nb_success/nb_trials)
        # -- Put all participants success rates together in one big array per sequence
        all_error_rates_seq.append(error_rates_seq) 

    # Mean value of each sequence error rates array and Standard error of the mean for for each sequence error rates array
    for i in range(len(all_error_rates_seq)):
        mean_per_participant_error_rates.append(np.mean(all_error_rates_seq[i]))
        sem_per_participant_error_rates.append(np.std(all_error_rates_seq[i])/np.sqrt(len(all_error_rates_seq[i])))


    # Plotting
    plt.rcParams["figure.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=plot_figsize)
    ax.barh(np.arange(len(seq_name_list)), mean_per_participant_error_rates,
            xerr=sem_per_participant_error_rates, capsize=5, align="center", color=plot_colors)

    ax.set_yticks(np.arange(len(seq_name_list)))
    ax.set_yticklabels(seq_name_list, fontsize=14)
    ax.invert_yaxis()

    sec_axis = ax.secondary_yaxis("right")
    sec_axis.set_yticks(np.arange(len(seq_name_list)))
    sec_axis.set_yticklabels(list_seq_expression, fontsize=12)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlim(0,100)
    plt.title("Mean Error Rates - All Sequences", fontsize=title_size, pad=padding_size)
    #ax.set_xlabel("Mean DL value", fontsize=14, labelpad=14)

    plt.savefig(f'{path}/mean_errorRates_allSequences.jpg', 
                bbox_inches='tight', dpi=800)
    plt.show()
    plt.close()

def plot_targeted_interclick(data,response_structure,path='path',save=False):
    """ Plot the response_structure interclick timings.
    Goal is to be able to have a look at a particular response in top8 and then look at the encoding through the interclicks. 


    Args:
        data (pandas dataframe): dataframe from which to draw the data.
        response_structure (array): Target response structure of the participant to consider. Recommend: copy-paste from top8 function
        path (str, optional): root path where plots are saved in the project. Defaults to 'path' (place holder)
        save (bool, optional): Save the figure into a folder singular_reponse. Defaults to False.
    """
    # 
    # Holders
    # -- Holds the interclick times for the responses that match the response_structure
    match_interclicks=[]
    # -- Holds the standard error of the mean
    sem_timings=[]
    
    # Search the dataset
    for index in range(len(data)):
        if data['comparable_temp'].iloc[index]==response_structure:
            # -- Get matching interclicks
            match_interclicks.append(data['interclick_time'].iloc[index])
            # -- Get Original sequence structure
            original=data['sequences_structure'].iloc[index]
    
    # Compute the mean
    mean_timings=np.mean(match_interclicks, axis=0)
    
    # Generate the standard error of the mean for all the mean_timings
    sem_timings.append(np.std(match_interclicks,axis=0)/np.sqrt(len(match_interclicks)))
    
    # Print number of responses considered
    print(f'{len(match_interclicks)} Responses were considered')
    
    # Plotting 
    plt.vlines(x=range(0,11), ymin=300, ymax=np.max(mean_timings)+100, colors='black', ls='--', lw=1)
    # -- Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
    plt.xticks(ticks=range(0,len(mean_timings)), labels=range(1,len(mean_timings)+1))
    plt.title(f'Mean Interclick times. Presented {original}, Response {response_structure}',pad=padding_size,fontsize=title_size)
    plt.errorbar(range(len(mean_timings)), mean_timings, yerr=sem_timings, fmt='o', capsize=5, capthick=2, color="black")
    plt.plot(range(0,len(mean_timings)),mean_timings)
    plt.ylim(ymin=300, ymax=np.max(mean_timings)+100)  # Set y-axis limits
    plt.xlim(xmin=-1, xmax=len(mean_timings))
    
    if save:
        plt.savefig(f'{path}/interclick/individual/mean/singular_response/mean_interclicks_{response_structure}.png', bbox_inches='tight', dpi=800)    
      
    plt.show()
    # Close the current figure window
    plt.close()
    
    
#--------------------------------------------------

def plot_length_distribution(data,path):
    """Plot and save the distribution of the length of answers per sequence type. It is interesting to look at this plot to observe the 

    Args:
        data (_type_): _description_
        path (_type_): _description_
    """
    max_y=140
    for i,name in enumerate(seq_name_list):
        # Holders
        holder_length=[]
        length_frequency=[]

        # Get the length of answers for one type of sequence
        builder_frequency=np.zeros(18)
        for index, row in data[data['seq_name']==name].iterrows():
            this_length=len(row['sequences_response'])
            holder_length.append(this_length)
            if this_length<18:
                builder_frequency[this_length]+=1
        length_frequency.append(builder_frequency)
        # Append 1 to the index of zeros for the length of the sequences
        fig, ax = plt.subplots(figsize=(5,5))
        # Draw the bar plot for that sequence
        ax.bar(np.arange(0,18,1),builder_frequency, align='center')
        ax.set_xticks(np.arange(0,19,1))
        #ax.set_xlabel('Length of answer', fontsize=10)
        #ax.set_ylabel('Number of answers', fontsize=10)
        ax.set_ylim(0,max_y)
        ax.set_title(f'{name}\n({alpha_seq_expression[i]})',pad=padding_size, fontsize=title_size+5)
        plt.savefig(f'{path}/length/length_distribution_{name}.png', bbox_inches='tight', dpi=800)
        plt.close()
        
def plot_all_length(data,path):
    ### Plotting 
    # Create the figure and axes
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))
    max_y=140
    plot_index=0 #this is the index for all_mean_timings which differs in the length and increments if will take from index of plots
    name_index=0 #tracks the name of the sequence to display

    for index, ax in enumerate(axes.flat):
        # Holders
        holder_length=[]
        length_frequency=[]

        # Get the length of answers for one type of sequence
        builder_frequency=np.zeros(18)
        for index, row in data[data['seq_name']==seq_name_list[plot_index]].iterrows():
            this_length=len(row['sequences_response'])
            holder_length.append(this_length)
            if this_length<18:
                builder_frequency[this_length]+=1
        length_frequency.append(builder_frequency)
        


        ax.set_title(f'{seq_name_list[name_index]} : {list_seq_expression[name_index]}')
        ax.bar(np.arange(0,18,1),builder_frequency, align='center')
        ax.set_xticks(np.arange(0,19,1),labels=np.arange(0,19,1))
        ax.set_ylim(0,max_y)
        # -- GO to next iteration
        plot_index+=1
        name_index+=1

    # Save and show the plot
    plt.savefig(f'{path}/all_length_distribution_subplots.png', bbox_inches='tight', dpi=800)
    plt.show()
    # Close the current figure window
    plt.close()

#--------------------------------------------------

def plot_median_individual_interclick(data,path,expression=True,x_axis_num=False,z_score=True):
    """Creates one plot per sequence.
    Each plot contains the median interclick timings for one sequence.
    IMPORTANTLY: only correct responses are considered.

    Args:
        data (pandas dataframe): preprocessed data and already put in a dataframe (typically data_main)
        path (str): path where plots are to be stored. This path needs to contains your_path/interclick/individual/
        expression (bool): if True, plots will have the expression of the sequences as titles (e.g., AABBCC.AABBCC). If False, will have the name (e.g., repetition nested)
        x_axis_num (bool): if True, x-axis will be the index of the interclick (ex: ABC.ABC => 1/2/3/4/5). If False, it will be the expression of the elements (ex: ABC.ABC => AB/BC/CA/AB/BC)
        z_score (bool): if True, will plot the z-score of the interclick time instead of the absolute interclick time.s
    """
    ### Needed Variables
    # -- Constructing the x-ticks object 
    sequence_structure_str=[]
    sequence_structure_intClick=[]
    for index in range(len(seq_name_list)):
        sequence_structure_str.append(num_alph(data[data["seq_name"]==seq_name_list[index]]['sequences_structure'].iloc[0]))
        
    for k in range(len(sequence_structure_str)):
        holder=[]
        for index in range(11):
            holder.append('{a}-{b}'.format(a=sequence_structure_str[k][index],b=sequence_structure_str[k][index+1]))
        sequence_structure_intClick.append(holder)
        
    ### Holders arrays for median of interclick timings and standard error of the median
    all_median_timings=[]
    all_z_scores=[]
    sem_timings=[]

    ### Collecting median interclick-timings
    for index in range(len(seq_name_list)):
        # For each sequence:
        #
        # -- Get the interclick values of responses of the right length
        holder=data[(data['seq_name']==seq_name_list[index]) & (data['performance']=='success')]['interclick_time']

        # -- If there's no correct response for this particular sequence go to next iteration
        if len(holder)==0:
            print(f'\033[1m{seq_name_list[index]}\033[0m:  --- No correct responses were found --- ')
            continue
        else:
            print(f'\033[1m{seq_name_list[index]}\033[0m: [{len(holder)}] correct responses were considered ({list_seq_expression[index]}).')
        
        # -- Turn them in the right (numpy) format
        timings=[arr for arr in holder.to_numpy()]
        
        # -- Compute the median
        median_timings=np.median(timings, axis=0)
        
        # -- Compute z-scores
        z_scores = (median_timings - np.median(median_timings)) / np.std(median_timings)
        
        # -- Append results to the holder object
        all_z_scores.append(z_scores)
        all_median_timings.append(median_timings)
        
        # Generate the standard error of the median for all the median_timings
        sem_timings.append(np.std(timings,axis=0)/np.sqrt(len(timings)))

    ### Plotting 

    plot_index=0 #this is the index for all_mean_timings
    for index in range(len(seq_name_list)):
    # *** Mean interclicks
    
        # -- If there's no correct response for this particular sequence go to next iteration
        if len(data[(data['seq_name']==seq_name_list[index]) & (data['performance']=='success')]['interclick_time'])==0:
            continue

        plt.vlines(x=range(0,11), ymin=np.min(all_median_timings)-50, ymax=np.max(all_median_timings)+100, colors='black', ls='--', lw=1)
        # -- Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
        if x_axis_num:
            plt.xticks(ticks=range(0,11), labels=range(1,12))
        else:
            plt.xticks(ticks=[i-0.5 for i in range(0,12)], labels=[i for i in alpha_seq_expression[index]])
            plt.xlim(xmin=-1, xmax=11)
            
        if expression:
            plt.title(f'{seq_name_list[index]}: {list_seq_expression[index]}',pad=padding_size,fontsize=title_size)
        else:
            plt.title(f'Mean Interclick times: {seq_name_list[index]}',pad=padding_size,fontsize=title_size)
        plt.errorbar(range(11), all_median_timings[plot_index], yerr=sem_timings[plot_index], fmt='o', capsize=5, capthick=2, color="black")
        plt.plot(range(11),all_median_timings[plot_index])
        #plt.ylim(ymin=300, ymax=np.max(all_median_timings)+100)  # Set y-axis limits
        plt.ylim(ymin=350, ymax=800)  # Set y-axis limits
        plt.savefig(f'{path}/interclick/individual/median/median_interclicks_subplots_{seq_name_list[index]}.png', bbox_inches='tight', dpi=800)    
        plt.close()
     
# ---------------------------------------
# *********** Geometry ************
# ---------------------------------------   

def point_dist(x,y):
    """Euclidian distance between two points (unit of distance is a point on the figure)
    CAREFUL: Works only for hexagonal figures. Otherwise, change 6 to the number of points of the figure

    Args:
        x (int): coordinate of point 1
        y (int): coordinate of point 2

    Returns:
        int: return the euclidian distance between x and y
    """
    '''
    a=min(x,y)
    b=max(x,y)
    if (b-a)>3:
        return min(b-a,6-b+a)
    else:
        return a-b
    '''
    x=x+1
    y=y+1
    a=max(x,y)
    b=min(x,y)
    if abs(x-y)>3:
        return a-6+b
    else:
        return y-x
    


#Il faut que l'array commence à zéro pour éliminer l'effet de rotation
def array_point_dist(arr):
    #FIXME DOESN'T WORK AS INTENDED
    """Return an array of Euclidian distance between points of a sequence two by two (unit of distance is a point on the figure).
    It considers the set of tokens as they first appeared.

    Args:
        arr (array): sequence

    Returns:
        array: array of distances between the different tokens that compose the sequence
    """
    mapping=2*[i for i in range(6)]
    new_arr=[]
    for num in arr:
        new_arr.append(mapping[num-arr[0]+6])
    transformed_arr = [1 if x == 5 else (2 if x == 4 else x) for x in new_arr]
            
    dists=[]
    set_arr=pd.unique(transformed_arr)
    for i in range(len(set_arr)-1):
        dists.append(point_dist(set_arr[i],set_arr[i+1]))
    return dists

import numpy as np

# ---------------------------------------
# *********** Investigation ************
# ---------------------------------------   

def check_if_contained(larger_seq, chunk):
    """
    Checks if a sequence chunk is contained within a larger sequence using sliding window comparison.

    Parameters:
    -----------
    larger_seq : list or array-like
        The larger sequence in which to check for the presence of the chunk.
    
    chunk : list or array-like
        The subsequence to check for within the larger sequence.

    Returns:
    --------
    bool
        Returns True if the chunk is found as a contiguous subsequence within the larger sequence, False otherwise.

    Example:
    --------
    >>> larger_seq = [0, 1, 2, 3, 4]
    >>> chunk = [1, 2]
    >>> check_if_contained(larger_seq, chunk)
    True

    >>> larger_seq = [0, 1, 2, 3, 4]
    >>> chunk = [2, 4]
    >>> check_if_contained(larger_seq, chunk)
    False

    Notes:
    ------
    - The function uses NumPy's stride tricks to generate sliding windows of the larger sequence.
    - The chunk must appear in the same order and be contiguous within the larger sequence.
    """
    
    # Convert lists to numpy arrays
    larger_seq = np.array(larger_seq)
    chunk = np.array(chunk)
    
    # Define the window length
    window_length = len(chunk)
    
    # Create the sliding window view
    windows = np.lib.stride_tricks.sliding_window_view(larger_seq, window_length)
    
    # Check if any window matches the chunk
    return np.any(np.all(windows == chunk, axis=1))

def check_if_contained_percentage(df,chunk):
    """
    Checks the percentage of sequences in `data_main` where `chunk` is contained within the 'comparable_temp' column,
    for each sequence name in `seq_name_list`. Prints or displays results in Markdown format based on availability.

    Parameters:
    -----------
    chunk : list or array-like
        The subsequence to check for within each sequence in `data_main['comparable_temp']`.
    
    df: Pandas data Frame
        The main data with all the responses of the participants.

    Returns:
    --------
    None

    Example:
    --------
    >>> check_if_contained_percentage([1, 2])
    [1, 2] is contained in **50.0%** of [seq1] responses.    structure1
    [1, 2] is contained in **0.0%** of [seq2] responses.    structure2
    [1, 2] is contained in **0.0%** of [seq3] responses.    structure3
    """
    
    for name in seq_name_list:
        # Calculate the percentage of `comparable_temp` containing the chunk
        subset = df[df['seq_name'] == name]
        match_count = subset['comparable_temp'].apply(lambda x: check_if_contained(x, chunk)).sum()
        total_count = len(subset)
        percentage = (match_count / total_count) * 100
        percentage = round(percentage, 2)
        
        # Prepare the result string
        result_str = f'**{chunk}** is contained in **{percentage}%** of [{name}] responses.\t **{subset["sequences_structure"].iloc[0]}**'
        
        # Display in Markdown if enabled
        try: 
            display(Markdown(result_str))
        except NameError as e:
            print(f"Markdown display failed: {e}")
            print(result_str)  # Print result as a fallback

def count_subsequences(sequence, subsequence):
    """
    Count the number of times a subsequence appears in a sequence.

    Args:
    - sequence (list): The sequence to search within.
    - subsequence (list): The subsequence to count occurrences of.

    Returns:
    - int: Number of times the subsequence appears in the sequence.
    """
    count = 0
    # Turn lists into numpy arrays
    subsequence=np.array(subsequence)
    sequence=np.array(sequence)
    
    len_subsequence = len(subsequence)
    len_sequence = len(sequence)
    index = 0
    
    while index <= len_sequence - len_subsequence:
        # Check if the slices are equal element-wise
        if np.array_equal(sequence[index:index+len_subsequence], subsequence):
            count += 1
            index += len_subsequence  # Move index past the subsequence
        else:
            index += 1  # Move index by 1 to check the next position
    
    return count
        
def check_transition_probs(df,seq_name, size):
    # Get sequence of the queried name
    subset = df[df['seq_name'] == seq_name]

    # Get sequence structure
    seq_expression=subset['sequences_structure'].iloc[0]

    # Get all possible transitions of queried size
    possible_transitions = np.lib.stride_tricks.sliding_window_view(seq_expression, size)

    # Convert to list of tuples to handle as hashable
    transition_tuples = [tuple(transition) for transition in possible_transitions]

    # Get unique transitions as arrays
    unique_transitions = np.unique(transition_tuples, axis=0)

    # Print the percentage of these transitions found in all the answers
    all_chunk_counts=[]
    # Counts the number of transition occuring in the real structure
    original_chunk_counts=[]
    for chunk in unique_transitions:
        match_count = subset['comparable_temp'].apply(lambda x: count_subsequences(x, chunk)).sum()
        all_chunk_counts.append(match_count)
        original_chunk_counts.append(count_subsequences(seq_expression,chunk))
    
    total_count=sum(all_chunk_counts)
    original_total_count=sum(original_chunk_counts)

    try: 
        display(Markdown(f"# {seq_name} -- **{seq_expression}** \n"))
    except NameError as e:
        print(f"Markdown display failed: {e}")
        print(f"###{seq_name}### -- {seq_expression}")  # Print result as a fallback
            
    for nb, chunk,original_nb in zip(all_chunk_counts, unique_transitions,original_chunk_counts):
        percentage = round(nb / total_count * 100, 2)
        original_percentage=round(original_nb/original_total_count*100,2)
        # Display in Markdown if enabled
        percent_color="green" if (percentage - original_percentage) >= 0 else "red"
        result_str=(
            f"Transition **{chunk}** accounts for <span style='color:blue;'>"
            f"*{percentage}%*</span> of all transitions in response sequences "
            f"while representing <span style='color:purple;'>*{original_percentage}%*</span> of original sequence transitions. "
            f"Difference: <span style='color:{percent_color};'>"
            f"{round(percentage - original_percentage,2):+}%</span>."
        )
        try: 
            display(Markdown(result_str))
        except NameError as e:
            print(f"Markdown display failed: {e}")
            print(result_str)  # Print result as a fallback
    

# ---------------------------------------
# *****Dataset testing functions ********
# ---------------------------------------   
def check_nb_trials_per_seq(df):
    print('Number of trials per sequence type in the given dataframe \n')
    print('-------------------------------------------------------')
    for name in seq_name_list:
        print(f'{name} : {len(df[df["seq_name"] == name])}')

def check_nb_trials_per_participant(df):
    print('Number of trials per participant ID in the given dataframe \n')  
    all_length_2=[]
    all_participants_2=[i for i in df['participant_ID'].unique()]
    for id in all_participants_2:
        all_length_2.append(len(df[df['participant_ID']==id]))
    all_length_2=np.array(all_length_2)    
    print(np.unique(all_length_2))
        
'''
# ---------------------------------------
# ********* Versions changelogs *********
# ---------------------------------------
Current: Version 2.1

*** 13.07.2024: Version 2.1
- Changing the which_seq function so that it compare the sequence with the dictionnary reverse_mapping (in params.py) instead of going over every case.


*** 17.06.2024: Version 2.0
- Changing all the mean / median plots to have a similar format to Al Roumi et al. (Neuron, 2021)


*** 14.06.2024: Version 1.6
- Adding a new section: Investigation. This section is helpful for the investigation_memocrush.ipynb
- Section contains new functions such as check_if_contained

*** 10.06.2024: Version 1.5
- Adding the function plot deletion errors

*** 13.05.2024: Version 1.4
- Adding functions useful for the geometry effect testing: point_dist(), array_point_dist()
- Adding the swap_columns function to format original experiment dataframe into the same format as the new experiment dataframe

*** 26.04.2024: Version 1.3
- Adding the plot_all_length() function
- Adding the plot_specific_heatmap() function


*** 17.04.2024: Version 1.2.1
- Adding the plot_regression() function.
- Adding the plot_mean_error_rates function.
- Adding the plot_targeted_interclick function.
- Adding the plot_length_distribution function.
- Adding the plot_median_individual_interclick function.



*** 15.04.2024: Version 1.2
- Changed the interclick plotting functions so that it displays the index and labels of elements of the sequence in between ticks (because it represents intervals) rather than on ticks.
- Changed the interclick plotting functions so that they also compute z-scores

*** 02.04.2024: Version 1.1
- changes to the plot_individual_interclick() and plot_common_interclick() functions. Removed the expression of sequences in the x-axis as a standard behavior.
Replaced it with the numerals as index from 1 to 11.
- Removed the inclusion of all sequences in the right length for the plot_individual_interclick() and plot_common_interclick() functions. Replaced by including 
only correct reproductions of the sequence. + added a print() of how many responses have been considered for each plot.

'''