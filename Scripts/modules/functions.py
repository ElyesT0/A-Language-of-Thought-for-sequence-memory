from modules.params import *

def _adaptive_dpi(target_px=3000, min_dpi=150, max_dpi=600):
    """Compute DPI so that the longest figure dimension hits target_px pixels.
    Keeps file sizes proportional to figure content rather than flat 800 DPI everywhere."""
    w, h = plt.gcf().get_size_inches()
    return min(max_dpi, max(min_dpi, int(target_px / max(w, h))))

def prepare_dir(plot_path):
    figure_path=plot_path
    directories = [
        f'{figure_path}/error_rate_subset',
        f'{figure_path}/models/regression',
        f'{figure_path}/interclick/individual/mean/singular_response',
        f'{figure_path}/length',
        f'{figure_path}/interclick/individual/median',
        f'{figure_path}/interclick/individual/mean',
        f'{figure_path}/interclick/individual/mean_custom_y',
        f'{figure_path}/interclick/individual/z-score',
        f'{figure_path}/interclick/differentials',
        f'{figure_path}/heatmap/specific',
        f'{figure_path}/heatmap-structure',
        f'{figure_path}/models/comparison_explanation',
        f'{figure_path}/TP_heatmap',
        f'{figure_path}/learning_level_analysis',
        f'{figure_path}/first_items_accuracy',
        f'{figure_path}/first_items_accuracy/first_items_accuracy_structure',
        f'{figure_path}/first_items_accuracy/first_items_accuracy_regular',
        
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

        # Step 2: Select the "response_structure" column
        response_structure_column = filtered_df["response_structure"]

        # Step 3: Get frequency counts
        frequency_counts = response_structure_column.value_counts()

        # Step 4: Sort the frequency counts
        sorted_counts = frequency_counts.sort_values(ascending=False)

        # Step 5: Select the top 10 most frequent arrays
        top_8_frequent_arrays = sorted_counts.head(10)

        # Step 6: Compute the proportion
        total_count = response_structure_column.count()
        proportions = (top_8_frequent_arrays / total_count) * 100

        # Combine the frequency and proportions into a DataFrame for better formatting
        top_8_df = pd.DataFrame({'Frequency': top_8_frequent_arrays, 'Proportion (%)': proportions})

        # Print the result
        print(f"Correct array: {name}")
        print(dataset[dataset["seq_name"] == name]["sequences_structure"].to_numpy()[0])
        print("\nTop 8 by frequency")
        print(top_8_df)
        print("-----------------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------------\n")

def proportion_structureCorrect_fail(data,name_list=seq_name_list):
    """Prints out the proportion, for each sequence type, of responses with a perfect structure but that were incorrect. 
    For example: original is ABC.ABC.ABC.ABC, response is BCA.BCA.BCA.BCA.

    Args:
        data (_type_): _description_
    """
    proportion_holder=[]

    for name in name_list:
        
        # Step 1: Filter the DataFrame to get only the right sequence
        subset_name=data[data["seq_name"] == name]

        # Step 2: Filter the DataFrame to get only mistakes but with the right structure
        subset_name_fail=subset_name[(subset_name['performance']=='fail')&(subset_name['perfect_structure'])]

        # Step 3: Select the "response_structure" column
        total_nb_answers=len(subset_name["response_structure"])

        # Step 4: append to proportion holder
        proportion_holder.append(100*len(subset_name_fail)/total_nb_answers)

    # Step 5: Return a dictionary for better readability
    correctStructure_wrongAnswer=dict(zip(name_list,proportion_holder))

    # Step 6: Print results
    print(f'Proportion of perfect structure for a wrong answer (ex: original is ABC.ABC.ABC, response is BAC.BAC.BAC.')
    for (key, item) in correctStructure_wrongAnswer.items():
        if 'control' not in key:
            print('\n')
        print(f'{key} : {round(item,3)}%')
    
    
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

def array_structure(arr):
    seen = set()
    ordered_list = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            ordered_list.append(item)
    arr_dict=dict(zip(ordered_list,range(len(ordered_list))))
    return [arr_dict.get(item, -1) for item in arr]

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
def plot_common_interclick(data, path, save=False, x_axis_num=False, ylim_min=0.3, ylim_max=1.4,sequence_list=seq_name_list,nb_rows=3,nb_cols=3,figsize=(15,15),yticks_interval=4, print_values=False, colors=False, vlines=True, plot_title=''):
    """
    Generate and save plots visualizing the interclick timings and accuracy of sequences.

    This function processes the input data to compute interclick timings and accuracy metrics
    for a list of sequences. The results are plotted with subplots representing each sequence.
    It includes secondary y-axes for accuracy histograms and saves the final plot as a pdf file.

    Args:
        data (DataFrame): Input data containing sequence information, interclick timings, and responses.
        path (str): Directory path to save the generated plot.
        x_axis_num (bool, optional): Whether to display numerical indices on the x-axis. Defaults to False.
        ylim_min (float, optional): Minimum limit for the y-axis (interclick duration). Defaults to 0.3.
        ylim_max (float, optional): Maximum limit for the y-axis (interclick duration). Defaults to 1.4.
        sequence_list (list, optional): List of sequence names to process. Defaults to `seq_name_list`.
        nb_rows (int, optional): Number of rows in the subplot grid. Defaults to 3.
        nb_cols (int, optional): Number of columns in the subplot grid. Defaults to 3.
        figsize (tuple, optional): Size of the figure (width, height). Defaults to (15, 15).
        yticks_interval (int, optional): Interval for setting y-ticks on the secondary y-axis. Defaults to 4.

    Returns:
        None. The function generates a plot and saves it as a file.

    Notes:
        - Interclick timings are converted to seconds and plotted with error bars representing
          the standard error of the mean.
        - Accuracy histograms with error bars are displayed on secondary y-axes.
        - The plots include distinct colors based on sequence complexity (e.g., 2-element, 3-element sequences).
        - Labels and legends are adjusted dynamically for aesthetics.
    """
    
    
    # Define objects used to plot
    all_cumulative_interclicks=[]
    all_sem_interclick=[]
    all_accuracies=[]
    all_sem_accuracies=[]
    
    for name_index in range(len(sequence_list)):
        # Holds all the responses' interclicks that are correct until item of index "step"
        holder_subsequence_interclicks=[]
        
        # Holds the last mean interclick timings of the holder_subsequence_interclicks. Will be used for plotting.
        holder_mean_interclicks=[]
        
        # Holds the Standard error of the mean for each mean timing.
        holder_sem_interclicks=[]
        
        # Select relevant data
        # -- 1. Subset based on sequence name
        subset_name=data[data['seq_name']==sequence_list[name_index]].copy()
        

         # Calculate and plot accuracy histogram
        accuracy = compute_accuracy(subset_name)
        mean_accuracy = np.mean(accuracy, axis=0)
        sem_accuracy = np.std(accuracy, axis=0) / np.sqrt(len(accuracy))

        
        # Loop for steps start here
        for step in range(1,12):
        # -- 2. Subset based on correct answers up to index number 'step'
            subset_correct = subset_name[
            (subset_name['sequences_structure'].apply(len) > step) & 
            (subset_name['comparable_temp'].apply(len) > step) & 
            (subset_name['sequences_structure'].apply(lambda x: x[:step]) == subset_name['comparable_temp'].apply(lambda x: x[:step]))
        ]          
            if len(subset_correct)==0:
                holder_subsequence_interclicks=0
                holder_mean_interclicks.append(np.nan)
                holder_sem_interclicks.append(np.nan)
                
                
            else:
                # Fill the holders
                holder_subsequence_interclicks = subset_correct['interclick_time'].apply(lambda x: np.array(x[:step]) / 1000).to_numpy()  # Convert to seconds
                last_interclicks_column=[arr[-1] for arr in holder_subsequence_interclicks]
                mean_temp=np.mean(last_interclicks_column)
                holder_mean_interclicks.append(mean_temp)
                holder_sem_interclicks.append(np.std(last_interclicks_column)/np.sqrt(len(last_interclicks_column)))
                
                    
        # Append results to plotting objects
        all_cumulative_interclicks.append(holder_mean_interclicks)
        all_sem_interclick.append(holder_sem_interclicks)
        all_accuracies.append(mean_accuracy)
        all_sem_accuracies.append(sem_accuracy)

    # Turn everything into numpy arrays
    all_cumulative_interclicks=np.array(all_cumulative_interclicks)
    all_sem_interclick=np.array(all_sem_interclick)
    all_accuracies=np.array(all_accuracies)
    all_sem_accuracies=np.array(all_sem_accuracies)
    
    # If printing values
    if print_values:
        for i in range(len(sequence_list)):
                print(f'---------------\n\033[1mSequence {sequence_list[i]} mean cumulated accuracy\033[0m')
                for position in range(len(all_accuracies[i])):
                    print(f'[position: {position+1}]: {all_accuracies[i][position]}')
                    print(f'SEM: {all_sem_accuracies[i][position]}\n')

    # Plotting interclicks
    fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=figsize)
    plt.subplots_adjust(wspace=0.1)  # Adjust horizontal space between subplots
    
    plot_index = 0
    name_index = 0
    
    
    for index, ax in enumerate(axes.flat):
        #
        # ---------- Plot interclicks -------------
        #
        # Plot vertical lines
        if vlines:
            ax.vlines(x=range(11), ymin=ylim_min, ymax=ylim_max, colors='black', ls='--', lw=1)
            
        # Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
        if x_axis_num:
            ax.set_xticks(ticks=[i - 0.5 for i in range(12)], labels=range(1, 13))
            ax.set_xlim(xmin=-1, xmax=11)
        else:
            mapped_seq_expressions = [real_mapping.get(item, item) for item in sequence_list] if sequence_list else []
            ax.set_xticks(ticks=[i - 0.5 for i in range(12)], labels=[i for i in mapped_seq_expressions[name_index]])
            ax.set_xlim(xmin=-1, xmax=11)
        
        # Blue if 2-elements, Red if 3-elements, Green if 4-elements and more
        if not colors:
            color = (
            "#0466C8" if len(set(real_mapping[sequence_list[index]])) == 2 
            else "#9B2226" if len(set(real_mapping[sequence_list[index]])) == 3 
            else "#386641"
        )
        else:
            color= colors[index]
        ax.set_title(f'{sequence_list[name_index]}', fontweight='bold',color=color)
        ax.errorbar(range(11), all_cumulative_interclicks[plot_index], yerr=all_sem_interclick[plot_index], fmt='o', capsize=5, capthick=2, color="black")
        ax.plot(range(11), all_cumulative_interclicks[plot_index],color=color)
        ax.set_ylim(ymin=ylim_min, ymax=ylim_max)  # Set y-axis limits
        
        # Remove y-tick labels except for the first subplot in each row
        if index %5!=0:
            ax.set_yticks([])
        if index == 0:
            ax.set_ylabel("Interclick \nduration (s)",rotation=0,fontstyle='italic')
            ax.yaxis.set_label_coords(-0.3, 0.92)  # Position the label at the top
        # if index % 5 != 0:
        #     ax.set_yticks([])
        # if index % 5 == 0:
        #     ax.set_ylabel("Interclick \nduration (s)",rotation=0)
        #     ax.yaxis.set_label_coords(-0.2, 0.95)  # Position the label at the top


        #
        # ---------- Plot Accuracies -------------
        #
        # Add secondary y-axis for histogram
        ax2 = ax.twinx()
        
       
        ax2.bar(
            [i - 0.5 for i in range(0, 12)],  # X positions
            all_accuracies[plot_index],  # Heights
            yerr=all_sem_accuracies[plot_index],  # Error bars
            capsize=5,  # Size of the caps on the error bars
            alpha=0.3,  # Transparency of the bars themselves
            color='gray',  # Color of the bars
            error_kw=dict(elinewidth=2, alpha=0.5)  # Customizing error bars: line width and transparency
        )
        ax2.set_ylim(0, 1)
        
        # Set y-tick labels only for the far-right subplot of each row
        if index %5!=yticks_interval:
            ax2.set_yticks([])
        if index == yticks_interval:
            ax2.set_ylabel("Accuracy on \nn first items",rotation=0,fontstyle='italic')
            ax2.yaxis.set_label_coords(1.42, 1.02)  # Position the label at the top
        # if index % 5 != 4:
        #     ax2.set_yticks([])
        # else:
        #     ax2.set_ylabel("Accuracy on \nn first items",rotation=0)
        #     ax2.yaxis.set_label_coords(1.25, 1.02)  # Position the label at the top


    
        plot_index += 1
        name_index += 1
    if save:
        plt.savefig(f'{path}/{plot_title}_mean_interclicks_with_accuracy.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
    plt.show()

#--------------------------------------------------

def plot_single_interclick_only_full_correct(data, path, save=False, file_format='pdf', x_axis_num=False, ylim_min=0.3, ylim_max=1.4,
                                              sequence_name=seq_name_list[0], figsize=(15,15), vlines=True,
                                              plot_title='', limit_interclick=1, show_x_ticks=True, show_y_ticks=True,
                                              colors=False, text_size=sub_title_size):
    """
    Generate and save a plot visualizing interclick timings and accuracy for a single sequence.

    This function processes input data to compute interclick timings and accuracy metrics for a 
    single sequence specified by `sequence_name`. It plots interclick durations with error bars 
    (with timings converted to seconds) and overlays an accuracy histogram on a secondary y-axis. 
    Trials where the response was not fully correct are excluded. If the number of successful trials 
    is fewer than `limit_interclick`, the interclick data are not plotted.

    Args:
        data (DataFrame): Input data containing sequence information, interclick timings, and responses.
        path (str): Directory path to save the generated plot.
        save (bool, optional): If True, saves the generated plot as a pdf file. Defaults to False.
        x_axis_num (bool, optional): Whether to display numerical indices on the x-axis. Defaults to False.
        ylim_min (float, optional): Minimum y-axis limit for the interclick duration plot. Defaults to 0.3.
        ylim_max (float, optional): Maximum y-axis limit for the interclick duration plot. Defaults to 1.4.
        sequence_name (str): The name or identifier of the sequence to process.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (15, 15).
        vlines (bool, optional): Whether to display vertical dashed lines for alignment. Defaults to True.
        plot_title (str, optional): Title for the generated plot (also used in the filename if saving). Defaults to an empty string.
        limit_interclick (int, optional): Minimum number of successful trials required for plotting interclick timings. Defaults to 1.
        show_x_ticks (bool, optional): If True, displays x-ticks in the plot. Defaults to True.
        show_y_ticks (bool, optional): If True, displays y-ticks in the plot. Defaults to True.
        colors (str or bool, optional): Custom color for the plot. If False, a default color is determined 
            dynamically based on sequence complexity. Defaults to False.

    Returns:
        None: The function displays the plot and optionally saves it as a pdf file.

    Notes:
        - Interclick timings are converted to seconds and plotted with error bars representing the standard error of the mean (SEM).
        - An accuracy histogram with SEM error bars is displayed on a secondary y-axis.
        - If the number of successful trials is fewer than `limit_interclick`, the interclick data (error bars and line) are not plotted.
        - X-ticks and y-ticks can be optionally hidden using `show_x_ticks` and `show_y_ticks`.
    """
    # Filter data for the given sequence and compute accuracy metrics
    subset = data[data['seq_name'] == sequence_name].copy()
    accuracy = compute_accuracy(subset)
    mean_accuracy = np.mean(accuracy, axis=0)
    sem_accuracy = np.std(accuracy, axis=0) / np.sqrt(len(accuracy))
    
    # Compute interclick timings for successful trials and convert to seconds
    correct_times = subset[subset['performance'] == "success"]['interclick_time'].to_numpy()
    sample_size = correct_times.shape[0]
    timings = np.array([arr for arr in correct_times])
    mean_timings = np.mean(timings, axis=0) / 1000
    sem_timings = (np.std(timings, axis=0) / np.sqrt(len(timings))) / 1000

    # Create figure and set x-axis
    plt.figure(figsize=figsize)
    if show_x_ticks:
        if x_axis_num:
            plt.xticks(ticks=[i - 0.5 for i in range(12)], labels=range(1, 13))
        else:
            mapped_labels = [real_mapping.get(item, item) for item in sequence_name]
            plt.xticks(ticks=[i - 0.5 for i in range(12)], labels=mapped_labels)
        plt.xlim(-1, 11)
    else:
        plt.xticks([])

    plt.text(
        0.87, 0.85,  # X and Y positions in axes coordinates (1,1 would be the top-right corner)
        f'n={sample_size}',  # Text to display
        transform=plt.gcf().transFigure,  # Use figure-relative coordinates
        fontsize=text_size,  
        verticalalignment='top',
        horizontalalignment='right',
    )

    plt.title(f'{sequence_name}', fontweight='bold', color='black', pad=padding_size, fontsize=title_size)


    # Plot interclick data if sufficient trials exist
    if sample_size >= limit_interclick:
        if vlines:
            plt.vlines(x=range(11), ymin=ylim_min, ymax=ylim_max, colors='black', ls='--', lw=1)
        plt.errorbar(range(11), mean_timings, yerr=sem_timings, fmt='o', capsize=5, capthick=2, color="black")
        plt.plot(range(11), mean_timings, color="black")
        plt.ylim(ylim_min, ylim_max)
    else:
        print("Not enough successful trials to plot interclick data.")

    if show_y_ticks:
        plt.ylabel("Interclick \nduration (s)", rotation=0, fontstyle='italic', labelpad=20, fontsize=15)
        plt.gca().yaxis.set_label_coords(-0.15, 0.95)  # Adjust position (X, Y)
    else:
        plt.yticks([])

    # Plot accuracy histogram on a secondary y-axis
    ax2 = plt.gca().twinx()
    ax2.bar([i - 0.5 for i in range(12)], mean_accuracy, yerr=sem_accuracy,
            capsize=5, alpha=0.3, color='gray', error_kw=dict(elinewidth=2, alpha=0.5))
    ax2.set_ylim(0, 1)
    if mean_timings.size < 10 and vlines:
        ax2.vlines(x=range(11), ymin=0, ymax=1, colors='black', ls='--', lw=1)
    if show_y_ticks:
        ax2.set_ylabel("Accuracy on \nn first items", rotation=0, fontstyle='italic', labelpad=15,fontsize=15)
        ax2.yaxis.set_label_coords(1.15, 1)  # Adjust position (X, Y)

    else:
        ax2.set_yticks([])

    if save:
        plt.savefig(f'{path}/{plot_title}_mean_FULL_CORRECT_interclicks_with_accuracy.{file_format}',
                    bbox_inches='tight', dpi=_adaptive_dpi())
    plt.show()


#--------------------------------------------------
def plot_common_interclick_only_full_correct(data, path, save= False, x_axis_num=False, ylim_min=0.3, ylim_max=1.4,sequence_list=seq_name_list,nb_rows=3,nb_cols=3,figsize=(15,15),yticks_interval=4, print_values=False, colors=False, vlines=True, plot_title='', limit_interclick=1, show_x_ticks=True, show_y_ticks= True, sub_size=sub_title_size,text_size=13):
    """
    Generate and save plots visualizing interclick timings and accuracy for sequences.

    This function processes input data to compute interclick timings and accuracy metrics 
    for a list of sequences. It generates subplots for each sequence, displaying interclick 
    durations with error bars and accuracy histograms on secondary y-axes. The function 
    only includes trials where the response was fully correct and applies a minimum threshold 
    on the number of successful trials before plotting interclick data.

    Args:
        data (DataFrame): Input data containing sequence information, interclick timings, and responses.
        path (str): Directory path to save the generated plot.
        save (bool, optional): If True, saves the generated plot to the specified path. Defaults to False.
        x_axis_num (bool, optional): Whether to display numerical indices on the x-axis. Defaults to False.
        ylim_min (float, optional): Minimum y-axis limit for interclick duration plots. Defaults to 0.3.
        ylim_max (float, optional): Maximum y-axis limit for interclick duration plots. Defaults to 1.4.
        sequence_list (list, optional): List of sequence names to process. Defaults to `seq_name_list`.
        nb_rows (int, optional): Number of rows in the subplot grid. Defaults to 3.
        nb_cols (int, optional): Number of columns in the subplot grid. Defaults to 3.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (15, 15).
        yticks_interval (int, optional): Interval for setting y-ticks on the secondary y-axis. Defaults to 4.
        print_values (bool, optional): If True, prints computed accuracy values for each sequence. Defaults to False.
        colors (list or bool, optional): Custom colors for sequences; if False, default colors are used. Defaults to False.
        vlines (bool, optional): Whether to display vertical dashed lines for alignment in plots. Defaults to True.
        plot_title (str, optional): Title for the generated plot. Defaults to an empty string.
        limit_interclick (int, optional): Minimum number of successful trials required for plotting interclick timings. Defaults to 1.
        show_x_ticks (bool, optional): If True, displays x-ticks in plots. Defaults to True.
        show_y_ticks (bool, optional): If True, displays y-ticks in plots. Defaults to True.

    Returns:
        None: The function generates a plot and optionally saves it as a pdf file.

    Notes:
        - Interclick timings are converted to seconds and plotted with error bars representing 
          the standard error of the mean (SEM).
        - Accuracy histograms with SEM error bars are displayed on secondary y-axes.
        - Sequences with fewer than `limit_interclick` successful trials do not have their 
          interclick data plotted.
        - Colors are assigned dynamically based on sequence complexity unless specified.
        - X-ticks and y-ticks can be optionally hidden using `show_x_ticks` and `show_y_ticks`.
    """
    
    
    # Define objects used to plot
    all_correct_interclicks=[]
    all_sem_interclick=[]
    all_accuracies=[]
    all_sem_accuracies=[]
    all_sample_size=[]
    
    for name_index in range(len(sequence_list)):
        # Holds all the responses' interclicks that are correct until item of index "step"
        holder_subsequence_interclicks=[]
        
        # Holds the last mean interclick timings of the holder_subsequence_interclicks. Will be used for plotting.
        holder_mean_interclicks=[]
        
        # Holds the Standard error of the mean for each mean timing.
        holder_sem_interclicks=[]
        
        # Select relevant data
        # -- 1. Subset based on sequence name
        subset_name=data[data['seq_name']==sequence_list[name_index]].copy()
        

         # Calculate and plot accuracy histogram
        accuracy = compute_accuracy(subset_name)
        mean_accuracy = np.mean(accuracy, axis=0)
        sem_accuracy = np.std(accuracy, axis=0) / np.sqrt(len(accuracy))

        # Get interclick timings related data
        holder_correct_interclicks=subset_name[subset_name['performance']=="success"]['interclick_time'].to_numpy()  # Convert to seconds
        all_sample_size.append(np.shape(holder_correct_interclicks)[0])

        if len(holder_correct_interclicks)<limit_interclick:
            holder_correct_interclicks=[]

        # Compute mean and SEM for interclicks
        timings=[arr for arr in holder_correct_interclicks] # Changing to the right format (numpy)
        holder_mean_interclicks=np.mean(timings,axis=0)
        holder_sem_interclicks=np.std(timings,axis=0)/np.sqrt(len(timings))

                    
        # Append results to plotting objects
        all_correct_interclicks.append(holder_mean_interclicks/1000)
        all_sem_interclick.append(holder_sem_interclicks/1000)
        all_accuracies.append(mean_accuracy)
        all_sem_accuracies.append(sem_accuracy)

    # Turn everything into numpy arrays
    # all_sem_interclick=np.array(all_sem_interclick)
    all_accuracies=np.array(all_accuracies)
    all_sem_accuracies=np.array(all_sem_accuracies)
    
    # If printing values
    if print_values:
        for i in range(len(sequence_list)):
                print(f'---------------\n\033[1mSequence {sequence_list[i]} mean cumulated accuracy\033[0m')
                for position in range(len(all_accuracies[i])):
                    print(f'[position: {position+1}]: {all_accuracies[i][position]}')
                    print(f'SEM: {all_sem_accuracies[i][position]}\n')

    # Plotting interclicks
    fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=figsize)
    plt.subplots_adjust(wspace=0.1)  # Adjust horizontal space between subplots
    
    plot_index = 0
    name_index = 0
    
    
    for index, ax in enumerate(axes.flat):
        #
        # ---------- Plot interclicks -------------
        #
        # Plot vertical lines
    
            
        # Define the labels used in the x-axis. Either letters constitutive of the sequence structure or simple indexes.
        if show_x_ticks:
            if x_axis_num:
                ax.set_xticks(ticks=[i - 0.5 for i in range(12)], labels=range(1, 13))
                ax.set_xlim(xmin=-1, xmax=11)
            else:
                mapped_seq_expressions = [real_mapping.get(item, item) for item in sequence_list] if sequence_list else []
                ax.set_xticks(ticks=[i - 0.5 for i in range(12)], labels=[i for i in mapped_seq_expressions[name_index]])
                ax.set_xlim(xmin=-1, xmax=11)
        else:
            ax.set_xticks([])  # Hide x-ticks
            ax.set_xlim(xmin=-1, xmax=11)


        # Blue if 2-elements, Red if 3-elements, Green if 4-elements and more
        if not colors:
            color = (
            "#0466C8" if len(set(real_mapping[sequence_list[index]])) == 2 
            else "#9B2226" if len(set(real_mapping[sequence_list[index]])) == 3 
            else "#386641"
        )
        else:
            color= colors[index]

        sequence_name_tags = [
        "Global Repetition" if name == "control NoLocal nested" else 
        "Local Repetition" if name == "control NoGlobal nested" else name
        for name in sequence_list
    ]

        ax.set_title(f'{sequence_name_tags[name_index]}', fontweight='bold', color=color,fontsize=sub_size, pad=padding_size)

        # Sample size should appear in the top right corner
        ax.text(
        0.98, 0.98,  # X and Y positions in axes coordinates (1,1 would be the top-right corner)
        f'n={all_sample_size[name_index]}',  # Text to display
        transform=ax.transAxes,  # Use axes coordinates (0,0 is bottom-left, 1,1 is top-right)
        fontsize=text_size,  
        verticalalignment='top',
        horizontalalignment='right',
    )
        
        # -- If there's no correct response for this particular sequence go to next iteration
        if all_correct_interclicks[plot_index].size > 10:
            if vlines:
                ax.vlines(x=range(11), ymin=ylim_min, ymax=ylim_max, colors='black', ls='--', lw=1)
            interclicks = all_correct_interclicks[plot_index]
            sem_interclicks = all_sem_interclick[plot_index]

            # Plot error bars and data
            ax.errorbar(range(11), interclicks, yerr=sem_interclicks, fmt='o', capsize=5, capthick=2, color="black")
            ax.plot(range(11), interclicks, color="black")
            ax.set_ylim(ymin=ylim_min, ymax=ylim_max)  # Set y-axis limits

        # Remove y-tick labels except for the first subplot in each row
        if show_y_ticks:
            if index % nb_cols != 0:
                ax.set_yticks([])
            if index == 0:
                ax.set_ylabel("Interclick \nduration (s)", rotation=0, fontstyle='italic')
                ax.yaxis.set_label_coords(-0.3, 0.92)  # Position the label at the top
        else:
            ax.set_yticks([])

        #
        # ---------- Plot Accuracies -------------
        #
        # Add secondary y-axis for histogram
        
            ax2 = ax.twinx()

        if vlines:
            ax2.bar(
                [i - 0.5 for i in range(0, 12)],  # X positions
                all_accuracies[plot_index],  # Heights
                yerr=all_sem_accuracies[plot_index],  # Error bars
                capsize=5,  # Size of the caps on the error bars
                alpha=0.3,  # Transparency of the bars themselves
                color='gray',  # Color of the bars
                error_kw=dict(elinewidth=2, alpha=0.5)  # Customizing error bars: line width and transparency
            )
            ax2.set_ylim(0, 1)
        else:
            ax2.bar(
                [i - 0.5 for i in range(0, 12)],  # X positions
                all_accuracies[plot_index],  # Heights
                yerr=all_sem_accuracies[plot_index],  # Error bars
                capsize=5,  # Size of the caps on the error bars
                alpha=0.3,  # Transparency of the bars themselves
                color='gray',  # Color of the bars
                width=0.9,
                error_kw=dict(elinewidth=2, alpha=0.5)  # Customizing error bars: line width and transparency
            )
            ax2.set_ylim(0, 1)
        if all_correct_interclicks[plot_index].size < 10 & vlines:
            
            ax2.vlines(x=range(11), ymin=0, ymax=1, colors='black', ls='--', lw=1)

        if show_y_ticks:
            # Set y-tick labels only for the far-right subplot of each row
            if index % nb_cols != yticks_interval:
                ax2.set_yticks([])
            if index == yticks_interval:
                ax2.set_ylabel("Accuracy on \nn first items", rotation=0, fontstyle='italic')
                ax2.yaxis.set_label_coords(1.42, 1.02)  # Position the label at the top
        else:
            ax2.set_yticks([])

        plot_index += 1
        name_index += 1
    if save:
        plt.savefig(f'{path}/{plot_title}_mean_FULL_CORRECT_interclicks_with_accuracy.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
    plt.show()



#--------------------------------------------------
def compare_seq_interclick_indexes_groups(data, seq_index, group1, group2):
    """
    Compares the distributions of the mean interclick times for cumulatively correct substrings
    at specified groups of indexes for a given sequence.

    Parameters:
        data (pd.DataFrame): The dataset containing the interclick times and sequence information.
        seq_index (int): The index of the sequence to analyze in seq_name_list.
        group1 (list): A list of ordinal positions (indexes) for group 1.
        group2 (list): A list of ordinal positions (indexes) for group 2.
        
    Returns:
        dict: A dictionary containing the test statistics and p-value for the comparison.
    """

    # Filter to fully correct sequences (performance=="success")
    data = data[data['performance'] == "success"]

    seq_name = seq_name_list[seq_index]
    
    # Subset data for the given sequence name
    subset_data = data[data['seq_name'] == seq_name].copy()

    def get_group_interclick_times(subset, indexes):
        """
        Extracts and averages interclick times at a list of indexes for trials where the 
        preceding clicks are cumulatively correct.
        Returns a Series with the mean interclick time per participant.
        """
        # This list will store per-index Series of mean interclick times per participant.
        group_series = []
        for index in indexes:
            # Extract interclick time at the specified index (if available)
            interclick_times = subset['interclick_time'].apply(lambda x: x[index] if len(x) > index else np.nan).dropna()
            # Compute the mean interclick time per participant for this index
            group_series.append(interclick_times.groupby(subset['participant_ID']).mean())
        
        # Combine the per-index Series into one DataFrame and compute the average across indexes.
        if group_series:
            combined = np.column_stack([s for s in group_series])
            # Create an index based on the intersection of participants who have data for each index.
            common_index = group_series[0].index
            for s in group_series[1:]:
                common_index = common_index.intersection(s.index)
            # Restrict each Series to common participants
            combined = np.column_stack([s.loc[common_index] for s in group_series])
            # Compute the average interclick time across the indexes for each participant
            return pd.Series(np.mean(combined, axis=1), index=common_index)
        else:
            return pd.Series(dtype=float)

    # Get mean interclick times for group1 and group2
    mean_interclicks_group1 = get_group_interclick_times(subset_data, group1)
    mean_interclicks_group2 = get_group_interclick_times(subset_data, group2)
    
    # Ensure that we only compare participants who have data in both groups
    common_participants = mean_interclicks_group1.index.intersection(mean_interclicks_group2.index)
    mean_interclicks_group1 = mean_interclicks_group1.loc[common_participants]
    mean_interclicks_group2 = mean_interclicks_group2.loc[common_participants]
    
    # Calculate descriptive statistics
    mean_1 = np.mean(mean_interclicks_group1)
    std_1 = np.std(mean_interclicks_group1)
    sem_1 = stats.sem(mean_interclicks_group1)
    
    mean_2 = np.mean(mean_interclicks_group2)
    std_2 = np.std(mean_interclicks_group2)
    sem_2 = stats.sem(mean_interclicks_group2)
    
    print(f"Comparing interclick times for sequence '{seq_name}'")
    print(f"Group 1 (indexes: {group1}): Mean = {mean_1:.3f}, Std Dev = {std_1:.3f}, SEM = {sem_1:.3f}")
    print(f"Group 2 (indexes: {group2}): Mean = {mean_2:.3f}, Std Dev = {std_2:.3f}, SEM = {sem_2:.3f}")
    print('>>>>> Comparing <<<<<<<<<')
    group1_str=sorted([i+1 for i in group1])
    str_partition_seq=str(list_seq_expression[seq_index])
    for num in reversed(group1_str):
        str_partition_seq=str_partition_seq[:num] + "|" + str_partition_seq[num:]
    print('group 1 : ',str_partition_seq)
    group2_str=sorted([i+1 for i in group2])
    str_partition_seq=str(list_seq_expression[seq_index])
    for num in reversed(group2_str):
        str_partition_seq=str_partition_seq[:num] + "|" + str_partition_seq[num:]
    print('group 2 : ',str_partition_seq)

    
    # Test for normality
    normality_group1 = stats.shapiro(mean_interclicks_group1)
    normality_group2 = stats.shapiro(mean_interclicks_group2)
    
    if normality_group1.pvalue > 0.05 and normality_group2.pvalue > 0.05:
        # Both distributions appear normal: use a paired t-test
        test_stat, p_value = stats.ttest_rel(mean_interclicks_group1, mean_interclicks_group2)
        test_type = 'Paired t-test'
    else:
        # Otherwise, use the Wilcoxon signed-rank test
        test_stat, p_value = stats.wilcoxon(mean_interclicks_group1, mean_interclicks_group2)
        test_type = 'Wilcoxon signed-rank test'
    
    print(f"Test type: {test_type}")
    print(f"Test statistic: {test_stat:.3f}")
    print(f"P-value: {p_value:.4f}")

    # Compute and print effect size
    if test_type == 'Paired t-test':
        diff = mean_interclicks_group1 - mean_interclicks_group2
        cohen_d = diff.mean() / diff.std(ddof=1)
        print(f"Effect size (Cohen's d): {cohen_d:.3f}")
    elif test_type == 'Wilcoxon signed-rank test':
        diffs = mean_interclicks_group1 - mean_interclicks_group2
        non_zero_diffs = diffs[diffs != 0]
        n = len(non_zero_diffs)
        if n > 0:
            mean_W = n * (n + 1) / 4
            std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (test_stat - mean_W) / std_W
            r = abs(z) / np.sqrt(n)
            print(f"Effect size (r = Z / √N): {r:.3f}")
        else:
            print("Effect size (r): NA (no non-zero differences)")
    else:
        print("Effect size: Not computed (unsupported test type)")

    # Interpretation
    if p_value < 0.05:
        print("Conclusion: The difference in interclick times is statistically significant.\n")
    else:
        print("Conclusion: The difference in interclick times is not statistically significant.\n")

    
    #return {'test_type': test_type, 'test_stat': test_stat, 'p_value': p_value}

# Example of usage
# compare_seq_interclick_indexes_groups(data_main, seq_index=0, group1=[3,5,7,9], group2=[2,4,6,8])

        
    
    
    
    
#--------------------------------------------------
def plot_individual_interclick(data,path,expression=True,x_axis_num=False,z_score=False,y_boundaries=0):
    """Creates one plot per sequence.
    Each plot contains the mean interclick timings for one sequence.
    IMPORTANTLY: only correct responses are considered.

    Args:
        data (pandas dataframe): preprocessed data and already put in a dataframe (typically data_main)
        path (str): path where plots are to be stored. This path needs to contains your_path/interclick/individual/
        seq_name_list (list): list of sequence names
        list_seq_expression (list): list of sequence expressions
        num_alph (function): function to convert numbers to alphabets
        alpha_seq_expression (list): list of alpha sequence expressions
        expression (bool): if True, plots will have the expression of the sequences as titles (e.g., AABBCC.AABBCC). If False, will have the name (e.g., repetition nested)
        x_axis_num (bool): if True, x-axis will be the index of the interclick (ex: ABC.ABC => 1/2/3/4/5). If False, it will be the expression of the elements (ex: ABC.ABC => AB/BC/CA/AB/BC)
        z_score (bool): if True, will plot the z-score of the interclick time instead of the absolute interclick time.
        padding_size (int): padding size for the title
        title_size (int): font size for the title
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
        if y_boundaries:
            plt.ylim(ymin=y_boundaries[0], ymax=y_boundaries[1])  # Set y-axis limits
            plt.savefig(f'{path}/interclick/individual/mean_custom_y/{index}_mean_interclicks_subplots_{seq_name_list[index]}.pdf', bbox_inches='tight', dpi=_adaptive_dpi())    
            
        else:
            plt.ylim(ymin=350, ymax=800)  # Set y-axis limits
            plt.savefig(f'{path}/interclick/individual/mean/{index}_mean_interclicks_subplots_{seq_name_list[index]}.pdf', bbox_inches='tight', dpi=_adaptive_dpi())    
            
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
            plt.savefig(f'{path}/interclick/individual/z-score/z-score_subplots_{seq_name_list[index]}.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
            
            plt.show()
            # Close the current figure window
            plt.close()
        plot_index+=1

#--------------------------------------------------   

# I want to look at the plot_targeted_interclick() of Mirror-NoRep (index: 23). Specifically for responses that 
## have the first 7 elements correct.

def count_correct_before_first_incorrect(row):
    """
    Counts the number of correct elements before the first incorrect element in a sequence.

    Parameters:
    - row: A row from a DataFrame containing the sequence and the participant's response.
    - structure (bool): If True, use 'response_structure'; if False, use 'sequences_response'.

    Returns:
    - int: The count of correct elements before the first incorrect element.
    """
    response = row['sequences_response']
    sequence = row['seq']
    
    count = 0
    for i in range(min(len(sequence), len(response))):
        if sequence[i] == response[i]:
            count += 1
        else:
            break
    return count

def apply_correct_count(data):
    """
    Applies the count_correct_before_first_incorrect function to each row of the DataFrame.

    Parameters:
    - data: DataFrame containing the experimental data.
    - structure (bool): If True, use 'response_structure'; if False, use 'sequences_response'.

    Returns:
    - Series: A Pandas Series with the count of correct elements before the first incorrect element for each row.
    """
    return data.apply(count_correct_before_first_incorrect, axis=1)


def plot_targeted_interclick_firstItems(data, name_index, correct_count, path='path', y_boundaries=0, save=False):
    """
    Plot the interclick timings for a specific sequence and a subset of data where
    the first 'n' elements of the sequence were correctly answered.

    This function filters the dataset to focus on responses corresponding to a particular
    sequence (specified by `seq_name`) and where the participant correctly answered the 
    first `correct_count` elements. It then calculates and plots the mean interclick 
    timings for the first `(n-1)` interclicks with error bars representing the standard 
    error of the mean (SEM).

    Args:
        data (pandas.DataFrame): DataFrame containing the experimental data, including 
                                 columns for sequence responses, interclick times, and 
                                 sequence structures.
        name_index (int): The index to fetch name of the sequence in seq_name_list to filter the data by.
        correct_count (int): The number of correct first elements in the sequence to 
                             consider for filtering the data.
        path (str, optional): The directory path where the plot will be saved if `save=True`.
                              Defaults to 'path'.
        y_boundaries (tuple, optional): A tuple specifying the y-axis boundaries 
                                        as (min, max). If not provided, the y-axis 
                                        will be automatically scaled. Defaults to 0.
        save (bool, optional): If True, the plot will be saved to the specified `path`.
                               Defaults to False.

    Returns:
        None. The function generates a plot showing the mean interclick times for the 
        filtered responses and displays it. If `save=True`, the plot is saved as a 
        pdf file in the specified directory.
    
    Example:
        plot_targeted_interclick(data, seq_name='Mirror-NoRep', correct_count=7, 
                                 path='results', y_boundaries=(300, 1000), save=True)
    
        This example generates and saves a plot of the interclick timings for the 
        sequence 'Mirror-NoRep', considering only responses where the first 7 elements 
        were correctly answered. The plot is saved in the 'results' directory with 
        y-axis limits set between 300 and 1000 milliseconds.
    """
    data=data.copy()
    data['size_correct_chunk']=apply_correct_count(data)
    seq_name=seq_name_list[name_index]
    # Filter the data by sequence name and correct count
    subset_data = data[(data['seq_name'] == seq_name) & (data['size_correct_chunk'] == correct_count)]
    
    if subset_data.empty:
        print(f"No data found for sequence '{seq_name}' with {correct_count} correct first answers.")
        return
    
    # Holders
    match_interclicks = []
    sem_timings = []
    
    # Search the dataset
    for index, row in subset_data.iterrows():
        # Get the first (n-1) interclicks
        interclicks = row['interclick_time'][:correct_count-1]
        if len(interclicks) == correct_count - 1:
            match_interclicks.append(interclicks)
        # Get Original sequence structure
        original = row['sequences_structure']
    
    # Compute the mean
    mean_timings = np.mean(match_interclicks, axis=0)
    
    # Generate the standard error of the mean for all the mean_timings
    sem_timings.append(np.std(match_interclicks, axis=0) / np.sqrt(len(match_interclicks)))
    
    # Print number of responses considered
    print(f'{len(match_interclicks)} Responses were considered')
    
    # Plotting 
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes object
    if y_boundaries:
        plt.vlines(x=range(0, len(mean_timings)), ymin=y_boundaries[0], ymax=y_boundaries[1], colors='black', ls='--', lw=1)
    else:
        plt.vlines(x=range(0, len(mean_timings)), ymin=300, ymax=np.max(mean_timings) + 100, colors='black', ls='--', lw=1)

    ax.set_xticks(ticks=[i-0.5 for i in range(0,12)], labels=[i for i in alpha_seq_expression[name_index]])
    ax.set_xlim(xmin=-1, xmax=11)
    
    plt.title(f'Mean Interclick times. Presented {original}, Response with {correct_count} correct first elements', pad=20, fontsize=14)
    plt.xlabel(f'{len(match_interclicks)} Responses were considered')
    plt.errorbar(range(len(mean_timings)), mean_timings, yerr=sem_timings, fmt='o', capsize=5, capthick=2, color="black")
    plt.plot(range(0, len(mean_timings)), mean_timings)
    
    if y_boundaries:
        plt.ylim(ymin=y_boundaries[0], ymax=y_boundaries[1])
    else:
        plt.ylim(ymin=300, ymax=np.max(mean_timings) + 100)  # Set y-axis limits
    
    plt.xlim(xmin=-1, xmax=len(mean_timings))
    
    if save:
        plt.savefig(f'{path}/interclick/individual/mean/singular_response/{name_index}_mean_interclicks_{seq_name}_{correct_count}_correct.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
    
    plt.show()
    plt.close()


#--------------------------------------------------   

def compute_error_rate(data,sequences=seq_name_list,length_considered=0): 
    """Compute the error rates of provided sequences

    Args:
        data (pandas Dataframe): Regular dataset that contains columns ['seq_name', 'sequences_response', 'seq'].
        sequences (list of str): The list of the names of the sequences on which to compute the error rate. Defaults to seq_name_list.
        length_considered (int, optional): List of length associated to the sequences. For every sequence the associated length defines what is a success. Such
            that: answer[:length] == original_sequence[:length]. Defaults to 0. If no value is provided, error rate is computed on the whole sequence.
        
    Comment:
     I want to compare the error rate of Rep-3 (first 3 chunks) and Rep-Nested. 
        I don't understand this: 
        - if chunks are non-breaking, then how is Rep-Nested that easy ?
        - if chunks are breaking, then how is Rep-3 that easy ?
        - if both are easy (I want to see if the difficulty comes from counting 4 chunks), then it is possible that the brain can alternate between
        chunking and not chunking the elements. OR it could mean that Stan is right, and we indeed navigate freely inside a set of slots of memory.

    """
    if length_considered==0:
        length_considered=[12 for i in sequences]

    success_rate=[]
    error_rates_all=[]

    for index_name in range(len(sequences)):
        subset_name=data[data['seq_name']==sequences[index_name]]
        nb_success=0
        nb_total=len(subset_name)
        for index,row in subset_name.iterrows():
            nb_success+=int(row["seq"][:length_considered[index_name]]==row['sequences_response'][:length_considered[index_name]])
            
        success_rate=100*nb_success/nb_total
        error_rates_all.append(100-success_rate)

    print('Dictionary links sequence_name to length such that, for answers to given sequence, the success is defined as \n=> answer[:length] == original[:length]')
            
    print(f'\nError Rate for {dict(zip(sequences,length_considered))}\n')
    for index_name in range(len(sequences)):
        print(f"{sequences[index_name]}: {round(error_rates_all[index_name],2)}")
   
def plot_error_rate(data,path,seq_list = seq_name_list, seq_expression = list_seq_expression):
    # == step1 == Create y axis: a list with rate of success per sequence
    success_rate=[]
    plt.rcParams['figure.facecolor'] = 'white'

    # New name_list with count of included sequences for each seq type in data_main (all main)

    count_seq_name_list=[]
    count_list_seq_expression=[]
    error_rates_all=[]
    for i in range(len(seq_list)):
        count_seq_name_list.append(seq_list[i]+" ({})".format(data[data["seq_name"]==seq_list[i]].count().iloc[0]))
        count_list_seq_expression.append(seq_expression[i]+" ({})".format(data[data["seq_name"]==seq_list[i]].count().iloc[0]))

    for i in range(len(seq_list)):
        nb_success=len(data[(data["seq_name"]==seq_list[i])&(data["performance"]=="success")])
        nb_total=len(data[data["seq_name"]==seq_list[i]])
        success_rate.append(100*nb_success/nb_total)
        error_rates_all.append(100-success_rate[i])

    colors = plot_colors

    plt.rcParams['figure.facecolor'] = 'white'
    fig,ax=plt.subplots(figsize=plot_figsize)
    
    
    ax.barh(np.arange(len(count_seq_name_list)),error_rates_all, align="center", color=colors,height=bar_thickness,linewidth=bar_frame_width)
    ax.set_yticks(np.arange(len(count_seq_name_list)))
    ax.set_yticklabels(seq_list)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    sec_axis=ax.secondary_yaxis("right")
    sec_axis.set_yticks(np.arange(len(count_seq_name_list)))
    sec_axis.set_yticklabels(seq_expression)
    sec_axis.tick_params(axis='y', labelsize=14)
    ax.set_xlim(0,100)
    ax.set_xlabel("Error Rate (%)", fontsize=14, labelpad=14)
    ax.set_title("Total Error Rate", fontsize=title_size, pad=padding_size)
    plt.savefig(f'{path}/errorRates_allSequences.jpg', 
                bbox_inches='tight', dpi=_adaptive_dpi())


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
                bbox_inches='tight', dpi=_adaptive_dpi())



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
    ax.barh(np.arange(len(seq_name_list)),seq_distance_DL_median, align="center", color=plot_colors,height=bar_thickness,linewidth=bar_frame_width)
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
                bbox_inches='tight', dpi=_adaptive_dpi())



#--------------------------------------------------
def plot_mean_dl(
                    data,
                    path,
                    plot_name='all',
                    print_values=True,
                    seq_expression=False,
                    sequences=seq_name_list,
                    save=True,
                    unfill_controls=True,
                    colors_figure=plot_colors,
                    change_label=False,
                    x_interval=False,
                    x_ticks = True,
                    gap_index = False,
                    gap_length = False,
                    figsize_x = 8,
                    figsize_y = 4.5,
                    pairs=pairs_for_stat_test_exp1,
                    group_structured=group_structured_exp1,
                    group_control=group_control_exp1):
    from scipy import stats as scipy_stats

    # Participants IDs
    IDs=[data.iloc[0]["participant_ID"]]
    for i in range(len(data)-1):
        if data.iloc[i]["participant_ID"] not in IDs:
            IDs.append(data.iloc[i]["participant_ID"])

    # Calculate the mean distance_DL for each sequence per participant
    temp_distDL_perParticipant = []
    # Dict: seq_name -> {participant_ID -> mean_dl}, for matched statistical tests (NaN participants excluded)
    participant_dl = {}

    # Gather sequence_expression
    sequence_expressions=[dict_expressions[key] for key in sequences]

    for name in sequences:
        new_arr = []
        participant_dl[name] = {}
        for participant in IDs:
            subset = data[(data["participant_ID"] == participant) & (data["seq_name"] == name)]
            mean_distance_dl = np.nanmean(subset["distance_dl"])  # Use np.nanmean to handle NaN values
            new_arr.append(mean_distance_dl)
            if not np.isnan(mean_distance_dl):
                participant_dl[name][participant] = mean_distance_dl

        temp_distDL_perParticipant.append(new_arr)

    # Convert the list of lists into a 2D NumPy array
    distDL_perParticipant = np.array(temp_distDL_perParticipant)

    # Calculate confidence intervals
    CI_meanDL = [confidence_interval95(dist) for dist in distDL_perParticipant]
    all_sem = [stats.sem(dist, nan_policy='omit') for dist in distDL_perParticipant]

    mean_distDL_perParticipant=[]
    summary_lines = []

    for i in range(len(all_sem)):
        # We use np.nanmean because participants from experiment 1 don't have values for sequences tested in experiment 2
        mean_distDL_perParticipant.append(round(np.nanmean(distDL_perParticipant[i]),2))
        if print_values:
            print(f'[{sequences[i]}] mean DL distance: {round(np.nanmean(distDL_perParticipant[i]),2)}')
            print(f'[{sequences[i]}] SEM: {round(all_sem[i],2)}\n')

    # ---- Build summary text ----
    W = 68
    def _section(title):
        return [' ' + '=' * (W - 2), f'  {title}', ' ' + '=' * (W - 2)]
    def _hline(char='-'):
        return ' ' + char * (W - 2)
    def _effect_size(x, y, p_val):
        n_nonzero = int(np.sum(np.array(x) - np.array(y) != 0))
        if n_nonzero == 0:
            return float('nan'), 'n/a', 0
        z = scipy_stats.norm.isf(p_val / 2)
        r = z / np.sqrt(n_nonzero)
        label = 'large' if r >= 0.5 else 'medium' if r >= 0.3 else 'small'
        return r, label, n_nonzero

    summary_lines += _section('MEAN DAMERAU-LEVENSHTEIN DISTANCE — SUMMARY')
    summary_lines.append('')
    summary_lines.append('  Per-Sequence Statistics')
    summary_lines.append(_hline())
    summary_lines.append(f"  {'Sequence':<34} {'N':>4}   {'Mean DL':>10}   {'SEM':>10}")
    summary_lines.append(_hline())
    for i, name in enumerate(sequences):
        n = len(participant_dl[name])
        summary_lines.append(f"  {name:<34} {n:>4}   {mean_distDL_perParticipant[i]:>10.4f}   {all_sem[i]:>10.4f}")
    summary_lines.append('')

    # Pairwise Wilcoxon
    summary_lines += _section('PAIRWISE WILCOXON SIGNED-RANK TESTS')
    summary_lines.append('  (mean DL distance per participant per sequence)')
    summary_lines.append(f"  Effect size r = Z / sqrt(N), N = non-zero difference pairs (Cohen, 1988)")
    summary_lines.append('')
    summary_lines.append(f"  {'Pair':<52} {'n':>4}   {'W':>8}   {'p':>8}   {'r':>6}  {'':>6}")
    summary_lines.append(_hline())
    for seq_a, seq_b in pairs:
        pair_label = f'{seq_a}  vs  {seq_b}'
        if seq_a not in participant_dl or seq_b not in participant_dl:
            summary_lines.append(f"  {pair_label:<52}  —  one or both sequences missing")
            continue
        common = sorted(set(participant_dl[seq_a].keys()) & set(participant_dl[seq_b].keys()))
        if len(common) < 2:
            summary_lines.append(f"  {pair_label:<52}  —  not enough matched participants (n={len(common)})")
            continue
        x = [participant_dl[seq_a][p] for p in common]
        y = [participant_dl[seq_b][p] for p in common]
        stat, p_val = scipy_stats.wilcoxon(x, y)
        r, r_label, n_nonzero = _effect_size(x, y, p_val)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        summary_lines.append(f"  {pair_label:<52} {len(common):>4}   {stat:>8.2f}   {p_val:>8.4f}   {r:>6.4f}  {r_label} {sig}")
    summary_lines.append('')

    # Group comparison
    summary_lines += _section('GROUP COMPARISON: STRUCTURED vs CONTROL')
    summary_lines.append(f"  Structured : {group_structured}")
    summary_lines.append(f"  Control    : {group_control}")
    summary_lines.append('')
    g_struct, g_ctrl, matched = [], [], []
    for p in IDs:
        s = [participant_dl[seq][p] for seq in group_structured if seq in participant_dl and p in participant_dl[seq]]
        c = [participant_dl[seq][p] for seq in group_control   if seq in participant_dl and p in participant_dl[seq]]
        if s and c:
            g_struct.append(np.mean(s))
            g_ctrl.append(np.mean(c))
            matched.append(p)
    if len(matched) >= 2:
        stat, p_val = scipy_stats.wilcoxon(g_struct, g_ctrl)
        r, r_label, n_nonzero = _effect_size(g_struct, g_ctrl, p_val)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        sem_s = np.std(g_struct) / np.sqrt(len(g_struct))
        sem_c = np.std(g_ctrl)   / np.sqrt(len(g_ctrl))
        summary_lines.append(_hline())
        summary_lines.append(f"  {'Group':<20} {'n':>4}   {'Mean DL':>10}   {'SEM':>8}   {'W':>8}   {'p':>8}   {'r':>6}  {'':>6}")
        summary_lines.append(_hline())
        summary_lines.append(f"  {'Structured':<20} {len(matched):>4}   {np.mean(g_struct):>10.4f}   {sem_s:>8.4f}   {stat:>8.2f}   {p_val:>8.4f}   {r:>6.4f}  {r_label} {sig}")
        summary_lines.append(f"  {'Control':<20} {len(matched):>4}   {np.mean(g_ctrl):>10.4f}   {sem_c:>8.4f}")
        summary_lines.append(_hline())
    else:
        summary_lines.append(f"  Not enough matched participants (n={len(matched)})")
    summary_lines.append('')
    summary_lines.append('  Significance: * p<0.05   ** p<0.01   *** p<0.001   ns = not significant')
    summary_lines.append('  Thresholds (Cohen, 1988): small >= 0.1, medium >= 0.3, large >= 0.5')
        

    # Extract the lower and upper bounds of the confidence interval
    lower_bound = np.array([item[0] for item in CI_meanDL])
    upper_bound = np.array([item[1] for item in CI_meanDL])

 
    # Plotting
    plt.rcParams["figure.facecolor"] = "white"


    # fig, ax = plt.subplots(figsize=(10,8))
    #plot_figsize_original = (10, len(sequences))
    #plot_figsize_current = (plot_figsize_coef * plot_figsize_original[0], (plot_figsize_coef-0.3) * plot_figsize_original[1])

   
    fig, ax = plt.subplots(figsize=(figsize_x,figsize_y))

    # Alternate colors for y-tick labels based on 'control' keyword
    yticklabels = []
    fill_conditions=[]
    for label in sequences:
        #color = 'grey' if 'control' in label.lower() else 'black'
        # yticklabels.append((label, color))
        weight='bold' if 'control' not in label.lower() else 'skip'
        yticklabels.append((label, weight))
        
    label_map = {
        "control NoLocal nested": "Global Repetition",
        "control NoGlobal nested": "Local Repetition"
    }

    # Replace only if the sequence name is in the dictionary
    display_labels = [label_map.get(seq, seq) for seq in sequences]

    if unfill_controls:
        for label in display_labels:
            fill_conditions.append(not 'control' in label.lower())
    else:
        fill_conditions = [True] * len(display_labels)

    # if change_label:
    #     fill_conditions=[True, True, True, False]

    y_ticks = np.arange(len(display_labels))
    if gap_index>0 and gap_index<len(y_ticks) and gap_length > 0:
        y_ticks[gap_index:] += gap_length

    for i, (filled, color) in enumerate(zip(fill_conditions, colors_figure)):

        ax.barh(y_ticks[i], mean_distDL_perParticipant[i],
                xerr=all_sem[i], capsize=5, align="center",
                edgecolor=color, facecolor=color if filled else 'none',height=bar_thickness,linewidth=bar_frame_width)
    
    if x_interval:
        ax.set_xlim(x_interval[0],x_interval[1])




    ax.set_yticks(y_ticks)

    # if change_label:
    #     new_labels=['Repetition-Nested','Global Repetition', 'Local Repetition', 'control Repetition-3']
    #     ax.set_yticklabels(new_labels,fontsize=14)
    # else:
    #     ax.set_yticklabels(sequences, fontsize=14)
    # Mapping dictionary
    
    # ax.set_yticks(y_ticks)
    ax.set_yticklabels(display_labels, fontsize=14)
    

    for tick, (label, weight) in zip(ax.get_yticklabels(), yticklabels):
        # tick.set_color(color)
        tick.set_text(label)
        tick.set_fontsize(14)
        if weight=='bold':
            tick.set_fontweight('bold')

    ax.invert_yaxis()

    if seq_expression:
        sec_axis = ax.secondary_yaxis("right")
        sec_axis.set_yticks(y_ticks)
        sec_axis.set_yticklabels(sequence_expressions, fontsize=12)
        
        # Set colors for secondary y-tick labels
        sec_yticklabels = sec_axis.get_yticklabels()
        for tick, (label, weight) in zip(sec_yticklabels, yticklabels):
            if weight=='bold':
                tick.set_fontweight('bold')
    else:
        # Hide the right spine when no secondary axis is used
        ax.spines["right"].set_visible(False) 
        ax.spines["top"].set_visible(False)
    
        

    if x_ticks:
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlabel("Mean Damerau-Levenshtein Distance", fontsize=title_size, labelpad=padding_size)
    else:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)

    if save:
        plt.savefig(f'{path}/mean_dl_{plot_name}.jpg',
                    bbox_inches='tight', dpi=_adaptive_dpi())
        print(f'successfully saved to :{path}/mean_dl_{plot_name}.jpg')
        txt_path = f'{path}/summary_dl_{plot_name}.txt'
        with open(txt_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f'successfully saved summary to: {txt_path}')
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
            xerr=all_sem, capsize=5, align="center", color=plot_colors,height=bar_thickness,linewidth=bar_frame_width)

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
                bbox_inches='tight', dpi=_adaptive_dpi())

#--------------------------------------------------
def plot_heatmap(data,path,seq_name_list=seq_name_list,structure=False,show=False,save_format="png", font_size = 18, color_bar = True, title = True, x_ticks_appear = True):
    sequence_name_tags = [
        "Global Repetition" if name == "control NoLocal nested" else 
        "Local Repetition" if name == "control NoGlobal nested" else name
        for name in seq_name_list
    ]
    #Variables
    #max_elements_sequence = np.max(data_main['sequences_response'].apply(len))
    max_elements_sequence=16
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Loop over seq_name_list
    for index, seq_name in enumerate(seq_name_list):
        # Build the heatmap object
        holder_heatmap = []
        if structure:
            # Get the number of tokens
            nb_tokens=len(set(data[data['seq_name'] == seq_name]['sequences_structure'].iloc[0]))
            all_comparable_temp = data[data['seq_name'] == seq_name]['response_structure'].to_numpy()
            
        else:
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
        if structure:
            heatmap_holder_percent=heatmap_holder_percent[-nb_tokens:]
            y_labels = list(alphabet[:(len(heatmap_holder_percent))])
        else:
            y_labels = list(alphabet[:(len(heatmap_holder_percent)-1)])
            y_labels.append('Error')
            
            
            
        y_labels.reverse()

        plt.rcParams['figure.facecolor'] = '#f1f3f5'
        
        if x_ticks_appear:
                xticks = range(max_elements_sequence)   # keep ticks
        else:
            xticks = False                          # no ticks
            
        if structure:
            # Calculate the aspect ratio to ensure cells are square
            aspect = max_elements_sequence / len(y_labels)

            # Set the figsize dynamically to maintain square cells
            plt.figure(figsize=(15, 15 / aspect))
        
            sns.heatmap(
                heatmap_holder_percent,
                annot=True,
                yticklabels=y_labels,
                xticklabels=xticks,
                fmt='g',
                linewidth=0.5,
                cmap="Purples",
                vmin=0,
                vmax=100,
                annot_kws={"size": font_size},
                cbar=color_bar
            )
            if title:
                plt.title(f"{sequence_name_tags[index].capitalize()}", pad=padding_size)
            
          
        else:  
            plt.figure(figsize=(15, 4))
            
            sns.heatmap(
                heatmap_holder_percent,
                annot=True,
                yticklabels=y_labels,
                xticklabels=xticks,
                fmt='g',
                linewidth=0.5,
                cmap="Purples",
                vmin=0,
                vmax=100,
                annot_kws={"size": font_size},
                cbar=color_bar
            )

            if title:
                plt.title(f"Reproduction patterns {sequence_name_tags[index]} (as%): {dict_expressions[seq_name]}", fontsize=title_size, pad=padding_size)
        if x_ticks_appear :
            plt.xlabel("Ordinal Rank")
        plt.yticks(rotation=0)

        if structure:
            plt.savefig(f'{path}/heatmap-structure/{index}_heatmap-structure_{sequence_name_tags[index]}.{save_format}', bbox_inches='tight', dpi=_adaptive_dpi())

        else:
            plt.savefig(f'{path}/heatmap/{index}_heatmap_{sequence_name_tags[index]}.{save_format}', bbox_inches='tight', dpi=_adaptive_dpi())
        if(show):
            plt.show()
        # Close the current figure window
        plt.close()

def plot_specific_heatmap(data,path,description,show=True,save=False):
    """Plot the response patterns as a heatmap for a particular set of data. 

    Args:
        data (pandas dataframe): sub-selection of the main data-set to observe a specific kind of answers patterns. example: data_main[(data_main['seq_name']=="control sub-programs 2")&(data_main['comparable_temp'].apply(lambda x: len(x)==10))]
        path (str): path where the plot will be saved
        description (str): Will be added at the end of the saved pdf name. Description of the criteria that selected the dataset. Also serves as a title for the figure. Example: controlSubPrograms2_lengthOf10
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
    y_labels.append('Error')
    y_labels.reverse()

    plt.rcParams['figure.facecolor'] = '#f1f3f5'
    plt.figure(figsize=(15, 4)) 
    sns.heatmap(heatmap_holder_percent, annot=True, yticklabels=y_labels, xticklabels=range(max_elements_sequence), fmt='g', linewidth=0.5, cmap="Purples")
    plt.title(f"Reproduction patterns {description} (as%). Population : {len(all_comparable_temp)}", fontsize=title_size, pad=padding_size)
    plt.xlabel("Ordinal Rank")
    plt.yticks(rotation=0)

    if(save):
        plt.savefig(f'{path}/heatmap/specific/heatmap_{description}.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
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
    plt.savefig(f'{path}/deletion_errors_index.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
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
    plt.savefig(f'{path}/deletion_errors_groups.jpg', bbox_inches='tight', dpi=_adaptive_dpi())

#--------------------------------------------------
def plot_regression(data,
 path,
 dl_distance=True,
 complexity_measure='LoT Complexity',
dict_complexity=complexities_initial_version,
 labels=False,
plot_colors=plot_colors,
 title='',
 y_label_pad =y_label_pad,
 x_label_pad =20,
 y_limit_plot= (0,7)):
    """ Plots the regression with seaborn. But also train an OLS model to output key components.

    Args:
        data (pandas dataFrame): dataframe containing 'LOT complexity', distance_dl. We recommand to use a no_training dataFrame (without the training sequences)
        path (str): root path for the saved plots.
        dl_distance (bool, optional): If True, will do the regression with y as the dl_distance. If False, uses the error rate instead. Defaults to True.
        complexity_measure (str, optional): Name of the column to consider for the complexity values of the sequences.
        labels (bool, optional): If True, will have the name of the sequences next to their mean. If False, it just draws the means. Default to False.
    """
    #NOTE this needs to be changed if we test other complexity sets
    unique_seq_names=data['seq_name'].unique()
    sequences_names=[name for name in seq_name_list if name in unique_seq_names]
    
    t_stat, p_value_t= t_test_on_OLS(aggregate_participants_OLS(data),display_text=False)
    formatted_name = "—".join((complexity_measure.replace(" ", "_"),title))

    
    
    complexities_ordered = [dict_complexity[name] for name in sequences_names if name in dict_complexity]
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
        # print("p-value:", p_value)
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

        for name in sequences_names:
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
        sns.regplot(data=data, x=f'{complexity_measure}', y='distance_dl', fit_reg=True, scatter=False,scatter_kws={'color': plot_colors}, ci=95, color='black', line_kws={"color": "black"})


        # .squeeze() Removes any single-dimensional entries, essentially [..] 
        # [..] converting a DataFrame with a single column into a Series
        # Plot the mean distance_DL for each sequence per participant with error bars for confidence intervals
        
        #plt.errorbar(x=complexities_ordered, y=all_distDL_means, yerr=all_sem, fmt='o', ecolor='black',elinewidth=1,color='firebrick',markeredgecolor='black')
        for i, (x, y, color) in enumerate(zip(complexities_ordered, all_distDL_means, plot_colors)):
            # We want the marker to be a square for controls, and a circle for structured sequences
            marker_style='s' if 'control' in sequences_names[i] else 'o'
            plt.errorbar(x, y, yerr=all_sem[i], fmt=marker_style, ecolor='black', elinewidth=1, color=color, markeredgecolor='black')

        # Create a custom legend
        for color, seq_name in zip(plot_colors, sequences_names[:len(plot_colors)]):
            plt.scatter([], [], color=color)  # Always plot the point
            if 'control' not in seq_name.lower():
                plt.scatter([], [], color=color, label=seq_name)  # Add to legend only if not 'control'

        plt.legend(prop={'size': legend_size}, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        x_min = data[f'{complexity_measure}'].min()
        x_max = data[f'{complexity_measure}'].max()
        # Expand limits by 5% of the range
        lower = x_min - (x_max - x_min) * 0.05
        upper = x_max + (x_max - x_min) * 0.05

        y_lower, y_upper = y_limit_plot

        # Apply truncation and rounding
        plt.xlim(math.floor(lower), math.ceil(upper))
        plt.ylim(y_lower, y_upper)
        # plt.xlim(x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.05) 
        
        # Add the legend
        
        if(labels):
            spread_factor=0.3
            for i, (x, y) in enumerate(zip(complexities_ordered, all_distDL_means)):
                spread_y = random.uniform(0, spread_factor)
                plt.text(x - 1, y+spread_factor-spread_y, f'{sequences_names[i]}', fontsize=8)  # Adjust the offset and font size as needed

            if complexity_measure=="LoT Complexity":
                plt.xlabel('Language of Thought Complexity', labelpad = x_label_pad)

            else:
                plt.xlabel(complexity_measure, labelpad = x_label_pad)

            plt.ylabel('DL Distance',rotation=0, labelpad=y_label_pad)
            plt.title(f'Linear Regression — {formatted_name}', fontsize=title_size, pad=padding_size)
            # Save and show the plot
            plt.savefig(f'{path}/models/regression/labels_dl_{formatted_name}_OLSregression.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
            plt.show()
            # Close the current figure window
            plt.close() 

        if complexity_measure=="LoT Complexity":
            plt.xlabel('Language of Thought Complexity')
        else:
            plt.xlabel(complexity_measure)

        plt.ylabel('DL Distance',rotation=0, labelpad=y_label_pad)
        plt.title(f'Linear Regression — {formatted_name}', fontsize=title_size, pad=padding_size)
        # Add Pearson's R value below the legend
        plt.text(1.05, 0.20, f"Pearson's r: {round(pearson_corr, 3)}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        # plt.text(1.05, 0.20, f"P-value: {'<0.001' if p_value < 0.001 else round(p_value, 3)}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(1.05, 0.10, "-----", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(1.05, 0.00, f"t-stat: {round(t_stat, 3)}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(1.05, -0.05, f"P-value: {'<0.001' if p_value_t < 0.001 else round(p_value_t, 3)}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        
        # Save and show the plot
        plt.savefig(f'{path}/models/regression/dl_{formatted_name}_OLSregression.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
        plt.show()
        # Close the current figure window
        plt.close() 
    else:
        # -- Running the same thing but with the error rate instead of the DL distance
        success_rate=[]
        count_sequences_names=[]
        count_list_seq_expression=[]
        error_rates_all=[]
        for i in range(len(sequences_names)):
            count_sequences_names.append(sequences_names[i]+" ({})".format(data[data["seq_name"]==sequences_names[i]].count().iloc[0]))
            count_list_seq_expression.append(list_seq_expression[i]+" ({})".format(data[data["seq_name"]==sequences_names[i]].count().iloc[0]))

        for i in range(len(sequences_names)):
            nb_success=len(data[(data["seq_name"]==sequences_names[i])&(data["performance"]=="success")])
            nb_total=len(data[data["seq_name"]==sequences_names[i]])
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
        dict_error_rate=dict(zip(sequences_names,error_rates_all))
        error_rates_regression_plot=[dict_error_rate[i] for i in data['seq_name']]
        sns.regplot(data=data, x='LoT Complexity', y=error_rates_regression_plot,fit_reg=True,scatter_kws={'color': plot_colors}, scatter=False, ci=95, color='black', line_kws={"color": "black"})

        
        # -- Compute confidence interval of error rates
        nb_trials=len(data[data['seq_name']==sequences_names[0]])
        success_counts=[]
        for name in sequences_names:
            success_counts.append(len(data[(data['performance']=='success')&(data['seq_name']==name)]))

            
        # Plot the error rate for each sequence per participant with error bars for confidence intervals
        #plt.errorbar(x=complexities_ordered, y=error_rates_all, fmt='o', ecolor='black',elinewidth=1,color='firebrick',markeredgecolor='black')
        
        for i, (x, y, color) in enumerate(zip(complexities_ordered, all_distDL_means, plot_colors)):
            # We want the marker to be a square for controls, and a circle for structured sequences
            marker_style='s' if 'control' in sequences_names[i] else 'o'
            plt.errorbar(x, y, yerr=all_sem[i], fmt=marker_style, ecolor='black', elinewidth=1, color=color, markeredgecolor='black')

        # Create a custom legend
        for color, seq_name in zip(plot_colors, sequences_names[:len(plot_colors)]):
            plt.scatter([], [], color=color)  # Always plot the point
            if 'control' not in seq_name.lower():
                plt.scatter([], [], color=color, label=seq_name)  # Add to legend only if not 'control'

        # Add the legend
        plt.legend(prop={'size': legend_size}, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        


        if(labels):
            spread_factor=0.3
            for i, (x, y) in enumerate(zip(complexities_ordered, all_distDL_means)):
                spread_y = random.uniform(0, spread_factor)
                plt.text(x - 1, y+spread_factor-spread_y, f'{sequences_names[i]}', fontsize=8)  # Adjust the offset and font size as needed

            plt.xlabel('Language of Thought Complexity', labelpad = x_label_pad)
            plt.ylabel('DL Distance',rotation=0, labelpad=y_label_pad)
            plt.title(f'Linear Regression — {formatted_name}', fontsize=title_size, pad=padding_size)
            
            # Save and show the plot
            plt.savefig(f'{path}/models/regression/labels_dl_{formatted_name}_OLSregression.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
            plt.show()
            # Close the current figure window
            plt.close() 

        plt.xlabel('Language of Thought Complexity', labelpad = x_label_pad)
        plt.ylabel('Error Rate',rotation=0, labelpad=y_label_pad)
        plt.title(f'Linear Regression — {formatted_name}', fontsize=title_size, pad=padding_size)
        # Add Pearson's R value below the legend
        plt.text(1.05, 0.5, f"Pearson's r: {pearson_corr:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(1.05, 0.3, f"p_value: {p_value:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        # Save and show the plot
        formatted_name = complexity_measure.replace(" ", "_")
        formatted_name = "—".join(formatted_name,title)
        saved_path = f'{path}/models/regression/errorRate_{formatted_name}_OLSregression.pdf'
        plt.savefig(saved_path, bbox_inches='tight', dpi=_adaptive_dpi())
        plt.show()
        # Close the current figure window
        print(f'Saved to : {saved_path}')
        plt.close() 

#--------------------------------------------------

# Plotting
def plot_mean_error_rates(data, path, print_values=True, save=True, seq_expression=False, sequences=seq_name_list, colors_figure=plot_colors, unfill_controls=True, order_diff=True, pairs=pairs_for_stat_test_exp1, group_structured=group_structured_exp1, group_control=group_control_exp1):
    from scipy import stats as scipy_stats
    # If order_diff is True, then on the plot, the histograms will be ordered by increasing level of difficulty
    # Preserve original sequence order for statistical tests (sequences may be reordered later for plotting)
    original_sequences = list(sequences)

    # Holder Objects
    all_error_rates_seq = []
    mean_per_participant_error_rates = []
    sem_per_participant_error_rates = []
    # Dict: seq_name -> {participant_ID -> error_rate}, for matched statistical tests
    participant_er = {}

    sequence_expressions = [dict_expressions[key] for key in sequences]

    summary_lines = []

    # For each sequence
    for name in original_sequences:
        error_rates_seq = []
        participant_er[name] = {}
        # -- For each participant
        for IDs in data['participant_ID'].unique():
            # -- Number of trials
            nb_trials = len(data[(data['participant_ID'] == IDs) & (data['seq_name'] == name)])

            # -- Test if there is at least one trial for this sequence
            if nb_trials != 0:
                # -- Total number of errors for the sequence
                nb_error = len(data[(data['participant_ID'] == IDs) & (data['seq_name'] == name) & (data['performance'] != 'success')])
                er = 100 * nb_error / nb_trials
                error_rates_seq.append(er)
                participant_er[name][IDs] = er

        all_error_rates_seq.append(error_rates_seq)

        if print_values:
            summary_lines.append(f'[{name}]: {len(error_rates_seq)} participants considered for error rate computation')

    summary_lines.append('---------------------------------------------------------')

    # Compute mean and SEM for each sequence
    for i in range(len(all_error_rates_seq)):
        mean_error_holder = np.mean(all_error_rates_seq[i])
        sem_holder = np.std(all_error_rates_seq[i]) / np.sqrt(len(all_error_rates_seq[i]))
        mean_per_participant_error_rates.append(mean_error_holder)
        sem_per_participant_error_rates.append(sem_holder)

        if print_values:
            summary_lines.append(f'[{original_sequences[i]}] Mean error rate: {np.round(mean_error_holder, 4)}')
            summary_lines.append(f'[{original_sequences[i]}] SEM: {np.round(sem_holder, 4)}')

    # Flatten all_error_rates_seq to combine error rates across all sequences
    all_error_rates_combined = [rate for seq_rates in all_error_rates_seq for rate in seq_rates]

    # Calculate overall mean and SEM
    overall_error_rate_all = np.mean(all_error_rates_combined)
    sem_overall_error_rate_all = np.std(all_error_rates_combined) / np.sqrt(len(all_error_rates_combined))
    summary_lines.append(f'\nOverall error rate (average over all sequences): {np.round(overall_error_rate_all, 4)}, SEM: {np.round(sem_overall_error_rate_all, 4)}')

    def _wilcoxon_effect_size(x, y, p_val):
        """Effect size r = Z / sqrt(N) for Wilcoxon signed-rank test.
        Z is derived from the two-tailed p-value via the normal approximation.
        N is the number of non-zero difference pairs (Cohen, 1988)."""
        n_nonzero = int(np.sum(np.array(x) - np.array(y) != 0))
        if n_nonzero == 0:
            return float('nan'), 'n/a'
        z = scipy_stats.norm.isf(p_val / 2)   # positive Z from two-tailed p
        r = z / np.sqrt(n_nonzero)
        label = 'large' if r >= 0.5 else 'medium' if r >= 0.3 else 'small'
        return r, label

    # ---- Pairwise Wilcoxon signed-rank tests ----
    summary_lines.append('\n' + '=' * 60)
    summary_lines.append('Pairwise Wilcoxon signed-rank tests (mean error rate per participant)')
    summary_lines.append('=' * 60)
    for seq_a, seq_b in pairs:
        if seq_a not in participant_er or seq_b not in participant_er:
            summary_lines.append(f'  [{seq_a}] vs [{seq_b}]: one or both sequences not found in data')
            continue
        common = sorted(set(participant_er[seq_a].keys()) & set(participant_er[seq_b].keys()))
        if len(common) < 2:
            summary_lines.append(f'  [{seq_a}] vs [{seq_b}]: not enough matched participants (n={len(common)})')
            continue
        x = [participant_er[seq_a][p] for p in common]
        y = [participant_er[seq_b][p] for p in common]
        stat, p_val = scipy_stats.wilcoxon(x, y)
        r, r_label = _wilcoxon_effect_size(x, y, p_val)
        n_nonzero = int(np.sum(np.array(x) - np.array(y) != 0))
        summary_lines.append(f'  [{seq_a}] vs [{seq_b}]')
        summary_lines.append(f'    n={len(common)} (non-zero pairs={n_nonzero}), W={stat:.4f}, p={p_val:.4f}, r={r:.4f} ({r_label})')

    # ---- Group comparison: structured vs control ----
    summary_lines.append('\n' + '=' * 60)
    summary_lines.append('Group Wilcoxon signed-rank test: structured vs control')
    summary_lines.append(f'  Structured: {group_structured}')
    summary_lines.append(f'  Control:    {group_control}')
    summary_lines.append('=' * 60)

    all_participants = data['participant_ID'].unique()
    group_structured_means = []
    group_control_means = []
    matched_participants = []
    for p in all_participants:
        s_rates = [participant_er[seq][p] for seq in group_structured if seq in participant_er and p in participant_er[seq]]
        c_rates = [participant_er[seq][p] for seq in group_control if seq in participant_er and p in participant_er[seq]]
        if s_rates and c_rates:
            group_structured_means.append(np.mean(s_rates))
            group_control_means.append(np.mean(c_rates))
            matched_participants.append(p)

    if len(matched_participants) >= 2:
        stat, p_val = scipy_stats.wilcoxon(group_structured_means, group_control_means)
        r, r_label = _wilcoxon_effect_size(group_structured_means, group_control_means, p_val)
        n_nonzero = int(np.sum(np.array(group_structured_means) - np.array(group_control_means) != 0))
        sem_structured = np.std(group_structured_means) / np.sqrt(len(group_structured_means))
        sem_control    = np.std(group_control_means)    / np.sqrt(len(group_control_means))
        summary_lines.append(f'  n={len(matched_participants)} (non-zero pairs={n_nonzero}), W={stat:.4f}, p={p_val:.4f}, r={r:.4f} ({r_label})')
        summary_lines.append(f'  Mean structured: {np.round(np.mean(group_structured_means), 4)}  (SEM: {np.round(sem_structured, 4)})')
        summary_lines.append(f'  Mean control:    {np.round(np.mean(group_control_means), 4)}  (SEM: {np.round(sem_control, 4)})')
    else:
        summary_lines.append(f'  Not enough matched participants for group comparison (n={len(matched_participants)})')
    summary_lines.append('\nEffect size r = Z / sqrt(N), where Z is from normal approximation of p-value and N = non-zero difference pairs.')
    summary_lines.append('Thresholds (Cohen, 1988): small >= 0.1, medium >= 0.3, large >= 0.5')
    
    plt.rcParams["figure.facecolor"] = "white"

    # Harmonized plot size with reference code
    plot_figsize_original = (10, len(sequences))
    plot_figsize_current = (plot_figsize_coef * plot_figsize_original[0], (plot_figsize_coef - 0.3) * plot_figsize_original[1])

    
    fig, ax = plt.subplots(figsize=plot_figsize_current)

    # Prepare yticklabels and fill conditions based on 'control' keyword
    yticklabels = []
    fill_conditions = []
    for label in sequences:
        weight = 'bold' if 'control' not in label.lower() else 'skip'
        yticklabels.append((label, weight))

    if unfill_controls:
        for label in sequences:
            fill_conditions.append(not 'control' in label.lower())
    else:
        fill_conditions = [True] * len(sequences)

    if order_diff:
        sorted_indexes=np.argsort(mean_per_participant_error_rates)
        # reorder sequences
        sequences_tmp=np.array(sequences)
        sequences=sequences_tmp[sorted_indexes]
        fill_conditions=np.array(fill_conditions)
        colors_figure=colors_figure
        
        # Bar plot with harmonized parameters
        for i, (filled, color) in enumerate(zip(fill_conditions[sorted_indexes], colors_figure)):
            ax.barh(i, mean_per_participant_error_rates[sorted_indexes[i]],
                    xerr=sem_per_participant_error_rates[sorted_indexes[i]], capsize=5, align="center",
                    edgecolor=color, facecolor=color if filled else 'none',
                    height=bar_thickness, linewidth=bar_frame_width)
        

    else:
        # Bar plot with harmonized parameters
        for i, (filled, color) in enumerate(zip(fill_conditions, colors_figure)):
            ax.barh(i, mean_per_participant_error_rates[i],
                    xerr=sem_per_participant_error_rates[i], capsize=5, align="center",
                    edgecolor=color, facecolor=color if filled else 'none',
                    height=bar_thickness, linewidth=bar_frame_width)
            
    # Set y-ticks and labels
    ax.set_yticks(np.arange(len(sequences)))
    ax.set_yticklabels(sequences, fontsize=14)

    if not order_diff:
        for tick, (label, weight) in zip(ax.get_yticklabels(), yticklabels):
            tick.set_text(label)
            tick.set_fontsize(14)
            if weight == 'bold':
                tick.set_fontweight('bold')

    ax.invert_yaxis()

    # Add secondary axis if required
    if seq_expression:
        sec_axis = ax.secondary_yaxis("right")
        sec_axis.set_yticks(np.arange(len(sequences)))
        sec_axis.set_yticklabels(sequence_expressions, fontsize=12)

        # Set bold font for secondary axis labels if needed
        if not order_diff:
            sec_yticklabels = sec_axis.get_yticklabels()
            for tick, (label, weight) in zip(sec_yticklabels, yticklabels):
                if weight == 'bold':
                    tick.set_fontweight('bold')

    # Harmonized x-axis and labels
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlim(35, 100)
    ax.set_xlabel("Mean Error Rate (%)", fontsize=title_size, labelpad=padding_size)

    # Set title for the plot
    plt.title("Mean Error Rates - All Sequences", fontsize=title_size, pad=padding_size)

    # Save or show plot
    if save:
        if len(original_sequences) == len(seq_name_list):
            plot_dir = path
            plt.savefig(f'{plot_dir}/mean_errorRates_allSequences.jpg', bbox_inches='tight', dpi=280)
        else:
            plot_dir = f'{path}/error_rate_subset'
            plt.savefig(f'{plot_dir}/mean_errorRates_subset.jpg', bbox_inches='tight', dpi=280)
        with open(f'{plot_dir}/summary_error_rate.txt', 'w') as f:
            f.write('\n'.join(summary_lines))
        print('✅ Successfully saved summary text in : ', f'{plot_dir}/summary_error_rate.txt')
    else:
        plt.show()
    plt.close()


def plot_mean_total_duration_trial(data, path, print_values=True, save=True, seq_expression=False, sequences=seq_name_list, colors_figure=plot_colors, unfill_controls=True, order_diff=True, pairs=pairs_for_stat_test_exp1, group_structured=group_structured_exp1, group_control=group_control_exp1):
    from scipy import stats as scipy_stats
    from pathlib import Path

    # Create total_RT column (sum of interclick times per trial)
    data = data.copy()
    data['total_RT'] = data['interclick_time'].apply(lambda x: sum(x))

    # Output directory
    out_dir = Path(path) / 'total_duration'
    out_dir.mkdir(parents=True, exist_ok=True)

    original_sequences = list(sequences)
    sequence_expressions = [dict_expressions[key] for key in sequences]

    # --- Data collection ---
    all_durations_seq = []
    mean_per_participant_durations = []
    sem_per_participant_durations = []
    participant_dur = {}   # seq_name -> {participant_ID -> mean_total_RT}

    for name in original_sequences:
        durations_seq = []
        participant_dur[name] = {}
        for IDs in data['participant_ID'].unique():
            subset = data[(data['participant_ID'] == IDs) & (data['seq_name'] == name)]
            if len(subset) != 0:
                mean_rt = subset['total_RT'].mean()
                durations_seq.append(mean_rt)
                participant_dur[name][IDs] = mean_rt
        all_durations_seq.append(durations_seq)

    for i, durations_seq in enumerate(all_durations_seq):
        mean_dur = np.mean(durations_seq)
        sem_dur  = np.std(durations_seq) / np.sqrt(len(durations_seq))
        mean_per_participant_durations.append(mean_dur)
        sem_per_participant_durations.append(sem_dur)

    all_durations_combined = [d for seq_d in all_durations_seq for d in seq_d]
    overall_mean = np.mean(all_durations_combined)
    overall_sem  = np.std(all_durations_combined) / np.sqrt(len(all_durations_combined))

    # --- Build pretty text summary ---
    W = 68   # total line width
    def section(title):
        return [' ' + '=' * (W - 2), f'  {title}', ' ' + '=' * (W - 2)]

    def hline(char='-'):
        return ' ' + char * (W - 2)

    lines = []
    lines += section('MEAN TOTAL TRIAL DURATION — SUMMARY')
    lines.append('')

    # Per-sequence table
    if print_values:
        lines.append('  Per-Sequence Statistics')
        lines.append(hline())
        col = f"  {'Sequence':<34} {'N':>4}   {'Mean (ms)':>10}   {'SEM (ms)':>10}"
        lines.append(col)
        lines.append(hline())
        for i, name in enumerate(original_sequences):
            n = len(all_durations_seq[i])
            lines.append(f"  {name:<34} {n:>4}   {mean_per_participant_durations[i]:>10.2f}   {sem_per_participant_durations[i]:>10.2f}")
        lines.append(hline())
        lines.append(f"  {'Overall (all sequences)':<34} {'':>4}   {overall_mean:>10.2f}   {overall_sem:>10.2f}")
        lines.append('')

    def _effect_size(x, y, p_val):
        n_nonzero = int(np.sum(np.array(x) - np.array(y) != 0))
        if n_nonzero == 0:
            return float('nan'), 'n/a', 0
        z = scipy_stats.norm.isf(p_val / 2)
        r = z / np.sqrt(n_nonzero)
        label = 'large' if r >= 0.5 else 'medium' if r >= 0.3 else 'small'
        return r, label, n_nonzero

    # Pairwise Wilcoxon
    lines += section('PAIRWISE WILCOXON SIGNED-RANK TESTS')
    lines.append('  (mean total duration per participant per sequence)')
    lines.append('  Effect size r = Z / sqrt(N), N = non-zero difference pairs (Cohen, 1988)')
    lines.append('')
    lines.append(f"  {'Pair':<52} {'n':>4}   {'W':>8}   {'p':>8}   {'r':>6}  {'':>6}")
    lines.append(hline())
    for seq_a, seq_b in pairs:
        pair_label = f'{seq_a}  vs  {seq_b}'
        if seq_a not in participant_dur or seq_b not in participant_dur:
            lines.append(f"  {pair_label:<52}  —  one or both sequences missing")
            continue
        common = sorted(set(participant_dur[seq_a].keys()) & set(participant_dur[seq_b].keys()))
        if len(common) < 2:
            lines.append(f"  {pair_label:<52}  —  not enough matched participants (n={len(common)})")
            continue
        x = [participant_dur[seq_a][p] for p in common]
        y = [participant_dur[seq_b][p] for p in common]
        stat, p_val = scipy_stats.wilcoxon(x, y)
        r, r_label, n_nonzero = _effect_size(x, y, p_val)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        lines.append(f"  {pair_label:<52} {len(common):>4}   {stat:>8.2f}   {p_val:>8.4f}   {r:>6.4f}  {r_label} {sig}")
    lines.append('')

    # Group comparison
    lines += section('GROUP COMPARISON: STRUCTURED vs CONTROL')
    lines.append(f"  Structured : {group_structured}")
    lines.append(f"  Control    : {group_control}")
    lines.append('')
    all_participants = data['participant_ID'].unique()
    g_struct, g_ctrl, matched = [], [], []
    for p in all_participants:
        s = [participant_dur[seq][p] for seq in group_structured if seq in participant_dur and p in participant_dur[seq]]
        c = [participant_dur[seq][p] for seq in group_control   if seq in participant_dur and p in participant_dur[seq]]
        if s and c:
            g_struct.append(np.mean(s))
            g_ctrl.append(np.mean(c))
            matched.append(p)
    if len(matched) >= 2:
        stat, p_val = scipy_stats.wilcoxon(g_struct, g_ctrl)
        r, r_label, n_nonzero = _effect_size(g_struct, g_ctrl, p_val)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        sem_s = np.std(g_struct) / np.sqrt(len(g_struct))
        sem_c = np.std(g_ctrl)   / np.sqrt(len(g_ctrl))
        lines.append(hline())
        lines.append(f"  {'Group':<20} {'n':>4}   {'Mean (ms)':>10}   {'SEM':>8}   {'W':>8}   {'p':>8}   {'r':>6}  {'':>6}")
        lines.append(hline())
        lines.append(f"  {'Structured':<20} {len(matched):>4}   {np.mean(g_struct):>10.2f}   {sem_s:>8.2f}   {stat:>8.2f}   {p_val:>8.4f}   {r:>6.4f}  {r_label} {sig}")
        lines.append(f"  {'Control':<20} {len(matched):>4}   {np.mean(g_ctrl):>10.2f}   {sem_c:>8.2f}")
        lines.append(hline())
    else:
        lines.append(f"  Not enough matched participants (n={len(matched)})")
    lines.append('')
    lines.append('  Significance: * p<0.05   ** p<0.01   *** p<0.001   ns = not significant')
    lines.append('  Thresholds (Cohen, 1988): small >= 0.1, medium >= 0.3, large >= 0.5')

    # --- Plot ---
    plt.rcParams["figure.facecolor"] = "white"
    plot_figsize_original = (10, len(sequences))
    plot_figsize_current  = (plot_figsize_coef * plot_figsize_original[0], (plot_figsize_coef - 0.3) * plot_figsize_original[1])
    fig, ax = plt.subplots(figsize=plot_figsize_current)

    yticklabels    = [(label, 'bold' if 'control' not in label.lower() else 'skip') for label in sequences]
    fill_conditions = [not 'control' in label.lower() for label in sequences] if unfill_controls else [True] * len(sequences)

    if order_diff:
        sorted_indexes  = np.argsort(mean_per_participant_durations)
        sequences_tmp   = np.array(sequences)
        sequences       = sequences_tmp[sorted_indexes]
        fill_conditions = np.array(fill_conditions)
        for i, (filled, color) in enumerate(zip(fill_conditions[sorted_indexes], colors_figure)):
            ax.barh(i, mean_per_participant_durations[sorted_indexes[i]],
                    xerr=sem_per_participant_durations[sorted_indexes[i]], capsize=5, align='center',
                    edgecolor=color, facecolor=color if filled else 'none',
                    height=bar_thickness, linewidth=bar_frame_width)
    else:
        for i, (filled, color) in enumerate(zip(fill_conditions, colors_figure)):
            ax.barh(i, mean_per_participant_durations[i],
                    xerr=sem_per_participant_durations[i], capsize=5, align='center',
                    edgecolor=color, facecolor=color if filled else 'none',
                    height=bar_thickness, linewidth=bar_frame_width)

    ax.set_yticks(np.arange(len(sequences)))
    ax.set_yticklabels(sequences, fontsize=14)

    if not order_diff:
        for tick, (label, weight) in zip(ax.get_yticklabels(), yticklabels):
            tick.set_text(label)
            tick.set_fontsize(14)
            if weight == 'bold':
                tick.set_fontweight('bold')

    ax.invert_yaxis()

    if seq_expression:
        sec_axis = ax.secondary_yaxis('right')
        sec_axis.set_yticks(np.arange(len(sequences)))
        sec_axis.set_yticklabels(sequence_expressions, fontsize=12)
        if not order_diff:
            for tick, (label, weight) in zip(sec_axis.get_yticklabels(), yticklabels):
                if weight == 'bold':
                    tick.set_fontweight('bold')

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlim(5500, 7000)
    ax.set_xlabel('Mean Total Trial Duration (ms)', fontsize=title_size, labelpad=padding_size)
    plt.title('Mean Total Trial Duration — All Sequences', fontsize=title_size, pad=padding_size)

    if save:
        plt.savefig(out_dir / 'mean_totalDuration_allSequences.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
        txt_path = out_dir / 'summary_total_duration.txt'
        txt_path.write_text('\n'.join(lines))

    plt.close()


def plot_mean_error_rates_structure(data,path,print_values=True,save=True,seq_expression=False,sequences=seq_name_list,colors_figure=plot_colors,unfill_controls=True):
    # Holder Objects
    all_error_rates_seq=[]
    mean_per_participant_error_rates=[]
    sem_per_participant_error_rates=[]


    sequence_expressions=[dict_expressions[key] for key in sequences]
    # For each sequence
    for name in sequences:
        error_rates_seq=[]
        # -- For each participant
        for IDs in data['participant_ID'].unique():
            subset_participant=data[(data['participant_ID']==IDs)&(data['seq_name']==name)]
            # -- number of trials
            nb_trials=len(subset_participant)

            # -- Test if there is at least one trial for this sequence (participants who did exp1 don't have trials on sequences of exp2)
            if nb_trials!=0:
                # -- Total success for the sequence
                nb_error=len(subset_participant[''])
                
            
                # -- Divided by number of trials (2)
                error_rates_seq.append(100*nb_error/nb_trials)
            
                
        # -- Put all participants success rates together in one big array per sequence
        all_error_rates_seq.append(error_rates_seq)
        
        if print_values:
            print(f'[{name}]: {len(error_rates_seq)} were considered for error_rate computation')

    print('---------------------------------------------------------\n')
    # Mean value of each sequence error rates array and Standard error of the mean for for each sequence error rates array
    for i in range(len(all_error_rates_seq)):
        mean_error_holder=np.mean(all_error_rates_seq[i])
        sem_holder=np.std(all_error_rates_seq[i])/np.sqrt(len(all_error_rates_seq[i]))
        mean_per_participant_error_rates.append(mean_error_holder)
        sem_per_participant_error_rates.append(sem_holder)
        if print_values:
            print(f'[{sequences[i]}] error rate: {mean_error_holder}')
            print(f'[{sequences[i]}] SEM: {sem_holder}\n')
    plt.rcParams["figure.facecolor"] = "white"

    plot_figsize_original = (10, len(sequences))
    plot_figsize_current = (plot_figsize_coef * plot_figsize_original[0], plot_figsize_coef * plot_figsize_original[1])

    fig, ax = plt.subplots(figsize=plot_figsize_current)

    # Alternate colors for y-tick labels based on 'control' keyword
    yticklabels = []
    fill_conditions=[]
    for label in sequences:
        #color = 'grey' if 'control' in label.lower() else 'black'
        # yticklabels.append((label, color))
        weight='bold' if 'control' not in label.lower() else 'skip'
        yticklabels.append((label, weight))
    
    if unfill_controls:
        for label in sequences:
            fill_conditions.append(not 'control' in label.lower())
    else:
        fill_conditions = [True] * len(sequences)

    for i, (filled, color) in enumerate(zip(fill_conditions, colors_figure)):
        ax.barh(i, mean_per_participant_error_rates[i],
                xerr=sem_per_participant_error_rates[i], capsize=5, align="center",
                edgecolor=color, facecolor=color if filled else 'none',height=bar_thickness,linewidth=bar_frame_width)

    # FIXME erase this later
    # ax.barh(np.arange(len(sequences)), mean_per_participant_error_rates,
    #         xerr=sem_per_participant_error_rates, capsize=5, align="center", color=colors_figure)

    ax.set_yticks(np.arange(len(sequences)))
    ax.set_yticklabels(sequences, fontsize=14)


    for tick, (label, weight) in zip(ax.get_yticklabels(), yticklabels):
        tick.set_text(label)
        tick.set_fontsize(14)
        if weight=='bold':
            tick.set_fontweight('bold')

    ax.invert_yaxis()

    if seq_expression:
        sec_axis = ax.secondary_yaxis("right")
        sec_axis.set_yticks(np.arange(len(sequences)))
        sec_axis.set_yticklabels(sequence_expressions, fontsize=12)
        
        # Set colors for secondary y-tick labels
        sec_yticklabels = sec_axis.get_yticklabels()
        for tick, (label, weight) in zip(sec_yticklabels, yticklabels):
            if weight=='bold':
                tick.set_fontweight('bold')
            

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlim(35,100)
    plt.title("Mean Error Rates - All Sequences STRUCTURE", fontsize=title_size, pad=padding_size)
    #ax.set_xlabel("Mean DL value", fontsize=14, labelpad=14)

    if save:
        if sequences==seq_name_list:
            plt.savefig(f'{path}/mean_errorRates_allSequences_structure.jpg', 
                        bbox_inches='tight', dpi=_adaptive_dpi())
        else:
            plt.savefig(f'{path}/error_rate_subset/mean_errorRates_subset_structure.jpg', 
                        bbox_inches='tight', dpi=_adaptive_dpi())
    plt.show()
    plt.close()     
    

def plot_targeted_interclick(data,response_structure,path='path',y_boundaries=0,save=False):
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
    if y_boundaries:
        plt.ylim(ymin=y_boundaries[0], ymax=y_boundaries[1])
    else:
        plt.ylim(ymin=300, ymax=np.max(mean_timings)+100)  # Set y-axis limits
    plt.xlim(xmin=-1, xmax=len(mean_timings))
    
    if save:
        plt.savefig(f'{path}/interclick/individual/mean/singular_response/mean_interclicks_{response_structure}.pdf', bbox_inches='tight', dpi=_adaptive_dpi())    
      
    plt.show()
    # Close the current figure window
    plt.close()
    
    
#--------------------------------------------------

def _classify_peaks(counts, dip_p, sig_level=0.05):
    """Geometry-first hierarchical classification of a count distribution.

    Peak detection runs once on the histogram with a neutral strategy (height >= 25% of
    maximum, minimum distance of 4 bins so adjacent high bars form one peak).  The
    resulting geometry (n_peaks, FWHM) is then the primary signal; the Dip Test is only
    used as a tiebreaker when the geometry is already ambiguous.

    Classification hierarchy (in priority order):
      1. n_peaks >= 2               → 'Multimodal'
      2. fwhm <= 2                  → 'Sharp/Leptokurtic'  (geometry beats statistics)
      3. dip_p < sig_level AND n_peaks > 1 → 'Multimodal'  (dip confirms peaks)
      4. otherwise                  → 'Broad/Gaussian'

    The consistency gate in rule 2 prevents the Dip Test from labelling a single,
    razor-thin peak as Multimodal — a known artefact when N is large (~200+).

    Args:
        counts    (array-like): raw count array (e.g. builder_frequency, length 18).
        dip_p     (float)     : p-value from Hartigan's Dip Test (or nan if unavailable).
        sig_level (float)     : significance threshold for the dip test (default 0.05).

    Returns:
        tuple: (n_peaks, classification, fwhm)
    """
    from scipy.signal import find_peaks
    counts = np.asarray(counts, dtype=float)
    h_max = counts.max()

    if h_max == 0:
        return 0, 'Empty', float('nan')

    # FWHM: span of indices where counts >= half the maximum
    above = np.where(counts >= h_max / 2.0)[0]
    fwhm = float(above[-1] - above[0] + 1) if len(above) >= 2 else (1.0 if len(above) == 1 else float('nan'))

    # Consistent peak detection: height threshold + distance to merge adjacent clusters
    peaks, _ = find_peaks(counts, height=0.25 * h_max, distance=4)
    n_peaks = len(peaks)

    # Geometry-first classification
    if n_peaks >= 2:
        classification = 'Multimodal'
    elif fwhm <= 2:
        classification = 'Sharp/Leptokurtic'
    elif (not np.isnan(dip_p)) and dip_p < sig_level and n_peaks > 1:
        classification = 'Multimodal'
    else:
        classification = 'Broad/Gaussian'

    return n_peaks, classification, fwhm

#--------------------------------------------------

def plot_length_distribution(data,path,show_plot=False,max_y=140,sequence_list=seq_name_list,exclude_long=False,comparison_pairs=pairs_for_stat_test_exp1,participant_col="participant_ID", max_length=17, file_prefix = ""):
    """Plot and save the distribution of the length of answers per sequence type. It is interesting to look at this plot to observe the

    Args:
        data (_type_): _description_
        path (_type_): _description_
    """
    from scipy import stats
    try:
        import diptest
        has_diptest = True
    except ImportError:
        has_diptest = False

    # ── Participant column detection ───────────────────────────────────────────
    if participant_col is None:
        candidates = [c for c in data.columns
                      if any(k in c.lower() for k in ('participant','subject','subj','_id','pp'))]
        if len(candidates) == 1:
            participant_col = candidates[0]
        elif len(candidates) == 0:
            raise ValueError(
                "Could not find a participant ID column automatically. "
                "Please pass participant_col='<column_name>' explicitly.")
        else:
            raise ValueError(
                f"Ambiguous participant ID column — candidates: {candidates}. "
                "Please pass participant_col='<column_name>' explicitly.")

    # ── Step 1: deduplicate to max 2 responses per (participant, seq_name) ────
    n_before = len(data)
    data = (data
            .sort_values([participant_col, 'seq_name'])
            .groupby([participant_col, 'seq_name'])
            .head(2)
            .reset_index(drop=True))
    n_after  = len(data)
    print(f"Deduplication: kept {n_after} rows, dropped {n_before - n_after} rows "
          f"(max 2 responses per participant per condition)")

    n_participants = data[participant_col].nunique()
    n_conditions   = data['seq_name'].nunique()
    mean_trials    = round(data.groupby([participant_col,'seq_name']).size().mean(), 2)
    print(f"Analysis pipeline validated: N_participants={n_participants}, "
          f"N_conditions={n_conditions}, mean trials per cell={mean_trials}")

    # ── Step 1.5: Filter by sequence length ──────────────────────────────────────
    # Calculate lengths once and filter the dataframe
    data['seq_len_temp'] = data['sequences_response'].map(len)
    n_before_len = len(data)

    # Keep only sequences shorter than max_length
    data = data[data['seq_len_temp'] < max_length].copy()
    data = data.drop(columns=['seq_len_temp'])

    n_after_len = len(data)
    print(f"Length Filter: Dropped {n_before_len - n_after_len} sequences with length >= {max_length}")

    # ── Step 2: per-participant means (Track A inferential unit) ──────────────
    per_participant = (
        data.groupby([participant_col, 'seq_name'])
        .apply(lambda g: pd.Series({
            'mean_length': np.mean([len(r) for r in g['sequences_response']])
        }))
        .reset_index()
    )

    sequence_list_renamed = [
                "Rep-Local" if x == "control NoGlobal nested" else 
                "Rep-Global" if x == "control NoLocal nested" else x 
                for x in sequence_list
            ]
    stats_rows    = []
    arrs_by_name  = {}   # trial-level length arrays  (Track B)
    pp_arrs_by_name = {} # per-participant mean arrays (Track A)
    freqs_by_name = {}   # builder_frequency arrays   (Track B)
    for i,name in enumerate(sequence_list):
        display_name = sequence_list_renamed[i]
        # Holders
        holder_length=[]
        length_frequency=[]

        # Track B — trial-level data ───────────────────────────────────────────
        builder_frequency=np.zeros(18)
        for index, row in data[data['seq_name']==name].iterrows():
            this_length=len(row['sequences_response'])
            if exclude_long and this_length >= max_length:
                continue
            holder_length.append(this_length)
            if this_length<max_length:
                builder_frequency[this_length]+=1
        length_frequency.append(builder_frequency)

        arr     = np.array(holder_length)
        n_trial = len(arr)

        arrs_by_name[name]  = arr
        freqs_by_name[name] = builder_frequency.copy()

        # Track B descriptive stats ────────────────────────────────────────────
        variance = np.var(arr) if n_trial > 0 else float('nan')
        seq_sd   = float(np.std(arr, ddof=1)) if n_trial >= 2 else float('nan')
        seq_mean = float(np.mean(arr)) if n_trial > 0 else float('nan')
        seq_mode = int(np.argmax(builder_frequency)) if n_trial > 0 else float('nan')

        if n_trial >= 3:
            sw_stat, sw_p = stats.shapiro(arr)
        else:
            sw_stat, sw_p = float('nan'), float('nan')

        if n_trial >= 4:
            skew = stats.skew(arr)
            kurt = stats.kurtosis(arr)
            bc   = (skew**2 + 1) / (kurt + 3 * (n_trial-1)**2 / ((n_trial-2) * (n_trial-3)))
        else:
            kurt, bc = float('nan'), float('nan')

        if has_diptest and n_trial >= 4:
            dip_stat, dip_p = diptest.diptest(arr)
        else:
            dip_stat, dip_p = float('nan'), float('nan')

        n_peaks, shape, fwhm = _classify_peaks(builder_frequency, dip_p)

        # Track A — per-participant means ──────────────────────────────────────
        pp_sub = per_participant[per_participant['seq_name'] == name]['mean_length'].values
        pp_arrs_by_name[name] = pp_sub
        n_pp   = len(pp_sub)

        stats_rows.append((display_name, name, n_trial, n_pp, seq_mean, seq_sd, seq_mode,
                           variance, kurt, sw_stat, sw_p, bc,
                           dip_stat, dip_p, n_peaks, fwhm, shape))

        # Append 1 to the index of zeros for the length of the sequences
        fig, ax = plt.subplots(figsize=(5,5))
        # Draw the bar plot for that sequence
        ax.bar(np.arange(0,18,1),builder_frequency, align='center')
        ax.set_xticks(np.arange(0,19,1))
        #ax.set_xlabel('Length of answer', fontsize=10)
        #ax.set_ylabel('Number of answers', fontsize=10)
        ax.set_ylim(0,max_y)
        ax.set_xlim(5,17)
        ax.set_title(f'{display_name}\n({alpha_seq_expression[i]})',pad=padding_size, fontsize=title_size+5)
        plt.savefig(f'{path}/length/{file_prefix}_{i}_length_distribution_{name}.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
        if show_plot:
            plt.show()
        plt.close()

    dip_note    = '' if has_diptest else '\n[!] diptest package not found — Dip Test columns are NaN. Install with: pip install diptest\n'
    sections_ok = 0

    with open(f'{path}/length/{file_prefix}length_variance.txt', 'w') as f:

        def _f(v, w, d=4): return f"{v:>{w}.{d}f}" if not np.isnan(v) else f"{'nan':>{w}}"

        # ── Pipeline note ─────────────────────────────────────────────────────
        f.write('Analysis pipeline: responses averaged per participant for inference '
                f'(N={n_participants}); full trial data used for shape descriptives (N≈{n_after//n_conditions if n_conditions else "?"}). '
                'Max 2 responses per participant enforced before all analyses.\n')
        f.write('=' * 130 + '\n')
        if dip_note:
            f.write(dip_note)

        # ══════════════════════════════════════════════════════════════════════
        # TRACK B — DESCRIPTIVE / SHAPE STATISTICS (trial-level data)
        # ══════════════════════════════════════════════════════════════════════
        f.write('\n')
        f.write('TRACK B — DESCRIPTIVE / SHAPE STATISTICS (trial-level data, N≈{} per condition,\n'.format(
            n_after // n_conditions if n_conditions else '?'))
        f.write('          2 trials per participant; mild non-independence does not affect descriptive estimates)\n')
        f.write('=' * 130 + '\n')

        col_header = (f"{'Sequence':<20} {'N_trial':>8} {'N_pp':>6} {'Mean':>7} {'SD':>7} {'Mode':>5} "
                      f"{'Var':>8} {'Kurt':>8} {'SW stat':>8} {'SW p':>8} {'BC':>8} "
                      f"{'Dip stat':>9} {'Dip p':>8} {'Peaks':>6} {'FWHM':>6}  {'Shape'}\n")
        f.write(col_header)
        f.write('-' * 138 + '\n')
        for (seq, _orig, n_trial, n_pp, seq_mean, seq_sd, seq_mode,
             variance, kurt, sw_stat, sw_p, bc,
             dip_stat, dip_p, n_peaks, fwhm, shape) in stats_rows:
            mean_s = f"{seq_mean:>7.2f}" if not np.isnan(seq_mean) else f"{'nan':>7}"
            sd_s   = f"{seq_sd:>7.2f}"   if not np.isnan(seq_sd)   else f"{'nan':>7}"
            mode_s = f"{seq_mode:>5}"    if not isinstance(seq_mode, float) else f"{'nan':>5}"
            fwhm_s = f"{fwhm:>6.1f}"     if not np.isnan(fwhm)     else f"{'nan':>6}"
            f.write(f"{seq:<20} {n_trial:>8} {n_pp:>6} {mean_s} {sd_s} {mode_s} "
                    f"{_f(variance,8)} {_f(kurt,8,2)} {_f(sw_stat,8)} {_f(sw_p,8)} {_f(bc,8)} "
                    f"{_f(dip_stat,9,4)} {_f(dip_p,8)} {n_peaks:>6} {fwhm_s}  {shape}\n")

        f.write('\n')
        f.write('Track B notes:\n')
        f.write('  Mean/Mode : descriptive centrality of individual response lengths (trial-level)\n')
        f.write('  Var / Kurt: trial-level variance and excess kurtosis (Fisher)\n')
        f.write('  SW        : Shapiro-Wilk normality test. p < 0.05 rejects normality\n')
        f.write('  BC        : Bimodality coefficient. BC > 0.555 suggests multimodality\n')
        f.write('  Dip       : Hartigan\'s Dip Test on trial-level data. p < 0.05 rejects unimodality\n')
        f.write('  FWHM      : Full Width at Half Maximum of the count histogram (in length units)\n')
        f.write('  Shape     : geometry-first — Multimodal (n_peaks>=2), Sharp/Leptokurtic (FWHM<=2), Broad/Gaussian\n')
        f.write('  Peak det. : find_peaks with height >= 25% of max, min distance = 4 bins\n')

        # ══════════════════════════════════════════════════════════════════════
        # TRACK A — INFERENTIAL TESTS (per-participant means, N=participants)
        # ══════════════════════════════════════════════════════════════════════
        f.write('\n')
        f.write(f'TRACK A — INFERENTIAL TESTS (per-participant means, N={n_participants})\n')
        f.write('=' * 130 + '\n')

        # ── Section 1: Pairwise dispersion tests ──────────────────────────────
        try:
            f.write('\nSECTION 1 — Pairwise dispersion tests (Brown-Forsythe / Levene center=median)\n')
            f.write(f"            Unit of analysis: per-participant mean response length (N={n_participants})\n")
            f.write('-' * 130 + '\n')
            f.write(f"{'Pair':<45} {'W':>10} {'p':>10}  Interpretation\n")
            f.write('-' * 130 + '\n')
            for pair in comparison_pairs:
                s1, s2 = pair[0], pair[1]
                if s1 not in pp_arrs_by_name or s2 not in pp_arrs_by_name:
                    f.write(f"  {s1} vs {s2} — skipped (sequence not in dataset)\n")
                    continue
                a1, a2 = pp_arrs_by_name[s1], pp_arrs_by_name[s2]
                if len(a1) < 2 or len(a2) < 2:
                    f.write(f"  {s1} vs {s2} — skipped (n < 2)\n")
                    continue
                w, p = stats.levene(a1, a2, center='median')
                sig   = '*' if p < 0.05 else ''
                direction = f"{s1} < {s2}" if np.var(a1) < np.var(a2) else f"{s1} > {s2}"
                f.write(f"  {f'{s1} vs {s2}':<43} {w:>10.4f} {p:>10.4f}  {direction} variance, p={p:.3f} {sig}\n")
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 1 failed: {e}\n')

        # ── Section 2: Control homogeneity ────────────────────────────────────
        try:
            f.write('\nSECTION 2 — Control homogeneity test (Brown-Forsythe across control Rep-2/3/4)\n')
            f.write(f"            Unit of analysis: per-participant mean response length (N={n_participants})\n")
            f.write('-' * 130 + '\n')
            ctrl_names  = ['control Repetition-2', 'control Repetition-3', 'control Repetition-4']
            ctrl_arrays = [pp_arrs_by_name[c] for c in ctrl_names
                           if c in pp_arrs_by_name and len(pp_arrs_by_name[c]) >= 2]
            if len(ctrl_arrays) >= 2:
                w_ctrl, p_ctrl = stats.levene(*ctrl_arrays, center='median')
                sig_ctrl = '*' if p_ctrl < 0.05 else ''
                f.write(f"  Groups tested : {', '.join(ctrl_names)}\n")
                f.write(f"  W = {w_ctrl:.4f},  p = {p_ctrl:.4f}  {sig_ctrl}\n")
                if p_ctrl >= 0.05:
                    f.write('  Interpretation: controls have statistically equal spread (H0 not rejected).\n')
                    f.write('  They form a coherent ANS baseline.\n')
                else:
                    f.write('  Interpretation: controls differ in spread (H0 rejected).\n')
                    f.write('  The ANS baseline is NOT homogeneous — control variances differ significantly.\n')
            else:
                f.write('  Not enough control groups found in the dataset.\n')
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 2 failed: {e}\n')

        # ── Section 3: Centrality — t-test + Cohen's d ────────────────────────
        try:
            f.write('\nSECTION 3 — Centrality: one-sample t-test vs μ=12 and Cohen\'s d\n')
            f.write(f"            Unit of analysis: per-participant mean response length (N={n_participants})\n")
            f.write('-' * 130 + '\n')
            f.write(f"{'Sequence':<20} {'N_pp':>6} {'Mean_pp':>9} {'SD_pp':>8} "
                    f"{'t':>8} {'p':>8} {'Cohen d':>9}  Interpretation\n")
            f.write('-' * 130 + '\n')
            for (seq, orig, *_rest) in stats_rows:
                pp_arr = pp_arrs_by_name[orig]
                if len(pp_arr) < 2:
                    f.write(f"  {seq:<18} — skipped (n_pp < 2)\n")
                    continue
                m_pp  = np.mean(pp_arr)
                sd_pp = np.std(pp_arr, ddof=1)
                t_s, t_p = stats.ttest_1samp(pp_arr, popmean=12)
                t_sig = '*' if t_p < 0.05 else ''
                if sd_pp > 0:
                    d = (m_pp - 12.0) / sd_pp
                    if   abs(d) < 0.2: interp = 'negligible deviation from 12'
                    elif abs(d) < 0.5: interp = f"small {'above' if d>0 else 'below'} 12"
                    elif abs(d) < 0.8: interp = f"medium {'above' if d>0 else 'below'} 12"
                    else:              interp = f"large {'above' if d>0 else 'below'} 12"
                else:
                    d, interp = float('nan'), 'SD=0'
                d_s = f"{d:>9.3f}" if not np.isnan(d) else f"{'nan':>9}"
                f.write(f"  {seq:<18} {len(pp_arr):>6} {m_pp:>9.3f} {sd_pp:>8.3f} "
                        f"{t_s:>8.3f} {t_p:>8.4f}{t_sig} {d_s}  {interp}\n")
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 3 failed: {e}\n')

        # ══════════════════════════════════════════════════════════════════════
        # TRACK B CONTINUED — Weber's Law + Rep-2 bimodal (shape/descriptive)
        # ══════════════════════════════════════════════════════════════════════
        f.write('\n')
        f.write('TRACK B CONTINUED — Shape statistics computed on trial-level data\n')
        f.write('=' * 130 + '\n')

        # ── Section 4: Weber's Law ────────────────────────────────────────────
        try:
            f.write('\nSECTION 4 — Weber\'s Law / scalar variability test (structured sequences, trial-level SD)\n')
            f.write('-' * 130 + '\n')
            weber_seqs    = ['Repetition-2', 'Repetition-3', 'Repetition-4']
            weber_ngroups = {'Repetition-2': 6, 'Repetition-3': 4, 'Repetition-4': 3}
            f.write(f"{'Sequence':<20} {'n_groups':>9} {'SD':>8} {'Weber frac':>11} {'CV':>8}\n")
            f.write('-' * 60 + '\n')
            sd_vals, ng_vals = [], []
            for ws in weber_seqs:
                if ws not in arrs_by_name:
                    f.write(f"  {ws:<18} — not found in dataset\n")
                    continue
                arr_w = arrs_by_name[ws]
                if len(arr_w) < 2:
                    f.write(f"  {ws:<18} — n < 2\n")
                    continue
                ng = weber_ngroups[ws]
                sd = np.std(arr_w, ddof=1)
                m  = np.mean(arr_w)
                wf = sd / ng if ng > 0 else float('nan')
                cv = sd / m  if m  > 0 else float('nan')
                f.write(f"  {ws:<18} {ng:>9} {sd:>8.4f} {wf:>11.4f} {cv:>8.4f}\n")
                sd_vals.append(sd)
                ng_vals.append(ng)
            f.write('\n')
            if len(sd_vals) == 3:
                r, p_r = stats.pearsonr(sd_vals, ng_vals)
                f.write(f"  Pearson r(SD, n_groups) = {r:.4f},  p = {p_r:.4f}\n")
                f.write('  [Note: 3-point correlation — treat p-value with caution.]\n')
                try:
                    slope, intercept, r2_root, p_lin, _se = stats.linregress(ng_vals, sd_vals)
                    f.write(f"  OLS SD ~ n_groups: slope={slope:.4f}, intercept={intercept:.4f}, "
                            f"R²={r2_root**2:.4f}, p={p_lin:.4f}\n")
                except Exception as ols_e:
                    f.write(f'  OLS failed: {ols_e}\n')
                f.write('  Weber fractions: see gradient in table above.\n')
                _d4 = [((np.mean(pp_arrs_by_name[ws]) - 12.0) / np.std(pp_arrs_by_name[ws], ddof=1))
                       if ws in pp_arrs_by_name and len(pp_arrs_by_name[ws]) >= 2
                          and np.std(pp_arrs_by_name[ws], ddof=1) > 0
                       else float('nan') for ws in weber_seqs]
                _v4 = [(n, d) for n, d in zip(ng_vals, _d4) if not np.isnan(d)]
                if len(_v4) >= 2:
                    _rho4, _p4 = stats.spearmanr([v[0] for v in _v4], [v[1] for v in _v4])
                    f.write(f"  Spearman rho(n_groups, Cohen_d) = {_rho4:.4f},  p = {_p4:.4f}\n")
                else:
                    f.write('  Spearman rho skipped — not enough valid Cohen\'s d values.\n')
            else:
                f.write('  Not enough sequences for regression.\n')
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 4 failed: {e}\n')

        # ── Section 4.b: Weber's Law — control sequences ──────────────────────
        try:
            f.write('\nSECTION 4.b — Weber\'s Law / scalar variability test (control sequences, trial-level SD)\n')
            f.write('-' * 130 + '\n')
            weber_control_seqs    = ['control Repetition-2', 'control Repetition-3', 'control Repetition-4']
            weber_control_ngroups = {'control Repetition-2': 12, 'control Repetition-3': 12, 'control Repetition-4': 12}
            f.write(f"{'Sequence':<25} {'n_groups':>9} {'SD':>8} {'Weber frac':>11} {'CV':>8}\n")
            f.write('-' * 65 + '\n')
            sd_vals_c, ng_vals_c = [], []
            for ws in weber_control_seqs:
                if ws not in arrs_by_name:
                    f.write(f"  {ws:<23} — not found in dataset\n")
                    continue
                arr_w = arrs_by_name[ws]
                if len(arr_w) < 2:
                    f.write(f"  {ws:<23} — n < 2\n")
                    continue
                ng = weber_control_ngroups[ws]
                sd = np.std(arr_w, ddof=1)
                m  = np.mean(arr_w)
                wf = sd / ng if ng > 0 else float('nan')
                cv = sd / m  if m  > 0 else float('nan')
                f.write(f"  {ws:<23} {ng:>9} {sd:>8.4f} {wf:>11.4f} {cv:>8.4f}\n")
                sd_vals_c.append(sd)
                ng_vals_c.append(ng)
            f.write('\n')
            if len(sd_vals_c) == 3:
                r_c, p_r_c = stats.pearsonr(sd_vals_c, ng_vals_c)
                f.write(f"  Pearson r(SD, n_groups) = {r_c:.4f},  p = {p_r_c:.4f}\n")
                f.write('  [Note: 3-point correlation — treat p-value with caution.]\n')
                try:
                    slope_c, intercept_c, r2_root_c, p_lin_c, _se_c = stats.linregress(ng_vals_c, sd_vals_c)
                    f.write(f"  OLS SD ~ n_groups: slope={slope_c:.4f}, intercept={intercept_c:.4f}, "
                            f"R²={r2_root_c**2:.4f}, p={p_lin_c:.4f}\n")
                except Exception as ols_e:
                    f.write(f'  OLS failed: {ols_e}\n')
                f.write('  Weber fractions: see gradient in table above.\n')
                _d4b = [((np.mean(pp_arrs_by_name[ws]) - 12.0) / np.std(pp_arrs_by_name[ws], ddof=1))
                        if ws in pp_arrs_by_name and len(pp_arrs_by_name[ws]) >= 2
                           and np.std(pp_arrs_by_name[ws], ddof=1) > 0
                        else float('nan') for ws in weber_control_seqs]
                _v4b = [(n, d) for n, d in zip(ng_vals_c, _d4b) if not np.isnan(d)]
                if len(_v4b) >= 2:
                    _rho4b, _p4b = stats.spearmanr([v[0] for v in _v4b], [v[1] for v in _v4b])
                    f.write(f"  Spearman rho(n_groups, Cohen_d) = {_rho4b:.4f},  p = {_p4b:.4f}\n")
                else:
                    f.write('  Spearman rho skipped — not enough valid Cohen\'s d values.\n')
            else:
                f.write('  Not enough sequences for regression.\n')
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 4.b failed: {e}\n')

        try:
            f.write('\nSECTION 4.c — Weber\'s Law / scalar variability test (control sequences, trial-level SD)\n')
            f.write('-' * 130 + '\n')
            weber_control_seqs    = ['control Repetition-2', 'control Repetition-3', 'control Repetition-4']
            weber_control_ngroups = {'control Repetition-2': 6, 'control Repetition-3': 4, 'control Repetition-4': 3}
            f.write(f"{'Sequence':<25} {'n_groups':>9} {'SD':>8} {'Weber frac':>11} {'CV':>8}\n")
            f.write('-' * 65 + '\n')
            sd_vals_c, ng_vals_c = [], []
            for ws in weber_control_seqs:
                if ws not in arrs_by_name:
                    f.write(f"  {ws:<23} — not found in dataset\n")
                    continue
                arr_w = arrs_by_name[ws]
                if len(arr_w) < 2:
                    f.write(f"  {ws:<23} — n < 2\n")
                    continue
                ng = weber_control_ngroups[ws]
                sd = np.std(arr_w, ddof=1)
                m  = np.mean(arr_w)
                wf = sd / ng if ng > 0 else float('nan')
                cv = sd / m  if m  > 0 else float('nan')
                f.write(f"  {ws:<23} {ng:>9} {sd:>8.4f} {wf:>11.4f} {cv:>8.4f}\n")
                sd_vals_c.append(sd)
                ng_vals_c.append(ng)
            f.write('\n')
            if len(sd_vals_c) == 3:
                r_c, p_r_c = stats.pearsonr(sd_vals_c, ng_vals_c)
                f.write(f"  Pearson r(SD, n_groups) = {r_c:.4f},  p = {p_r_c:.4f}\n")
                f.write('  [Note: 3-point correlation — treat p-value with caution.]\n')
                try:
                    slope_c, intercept_c, r2_root_c, p_lin_c, _se_c = stats.linregress(ng_vals_c, sd_vals_c)
                    f.write(f"  OLS SD ~ n_groups: slope={slope_c:.4f}, intercept={intercept_c:.4f}, "
                            f"R²={r2_root_c**2:.4f}, p={p_lin_c:.4f}\n")
                except Exception as ols_e:
                    f.write(f'  OLS failed: {ols_e}\n')
                f.write('  Weber fractions: see gradient in table above.\n')
                _d4c = [((np.mean(pp_arrs_by_name[ws]) - 12.0) / np.std(pp_arrs_by_name[ws], ddof=1))
                        if ws in pp_arrs_by_name and len(pp_arrs_by_name[ws]) >= 2
                           and np.std(pp_arrs_by_name[ws], ddof=1) > 0
                        else float('nan') for ws in weber_control_seqs]
                _v4c = [(n, d) for n, d in zip(ng_vals_c, _d4c) if not np.isnan(d)]
                if len(_v4c) >= 2:
                    _rho4c, _p4c = stats.spearmanr([v[0] for v in _v4c], [v[1] for v in _v4c])
                    f.write(f"  Spearman rho(n_groups, Cohen_d) = {_rho4c:.4f},  p = {_p4c:.4f}\n")
                else:
                    f.write('  Spearman rho skipped — not enough valid Cohen\'s d values.\n')
            else:
                f.write('  Not enough sequences for regression.\n')
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 4.c failed: {e}\n')

        # ── Section 4.c: Weber's Law — structured sequences, n_groups=12 ──────
        try:
            f.write('\nSECTION 4.d — Weber\'s Law / scalar variability test (structured sequences, n_groups=12, trial-level SD)\n')
            f.write('-' * 130 + '\n')
            weber_seqs_c    = ['Repetition-2', 'Repetition-3', 'Repetition-4']
            weber_ngroups_c = {'Repetition-2': 12, 'Repetition-3': 12, 'Repetition-4': 12}
            f.write(f"{'Sequence':<20} {'n_groups':>9} {'SD':>8} {'Weber frac':>11} {'CV':>8}\n")
            f.write('-' * 60 + '\n')
            sd_vals_sc, ng_vals_sc = [], []
            for ws in weber_seqs_c:
                if ws not in arrs_by_name:
                    f.write(f"  {ws:<18} — not found in dataset\n")
                    continue
                arr_w = arrs_by_name[ws]
                if len(arr_w) < 2:
                    f.write(f"  {ws:<18} — n < 2\n")
                    continue
                ng = weber_ngroups_c[ws]
                sd = np.std(arr_w, ddof=1)
                m  = np.mean(arr_w)
                wf = sd / ng if ng > 0 else float('nan')
                cv = sd / m  if m  > 0 else float('nan')
                f.write(f"  {ws:<18} {ng:>9} {sd:>8.4f} {wf:>11.4f} {cv:>8.4f}\n")
                sd_vals_sc.append(sd)
                ng_vals_sc.append(ng)
            f.write('\n')
            if len(sd_vals_sc) == 3:
                r_sc, p_r_sc = stats.pearsonr(sd_vals_sc, ng_vals_sc)
                f.write(f"  Pearson r(SD, n_groups) = {r_sc:.4f},  p = {p_r_sc:.4f}\n")
                f.write('  [Note: 3-point correlation — treat p-value with caution.]\n')
                try:
                    slope_sc, intercept_sc, r2_root_sc, p_lin_sc, _se_sc = stats.linregress(ng_vals_sc, sd_vals_sc)
                    f.write(f"  OLS SD ~ n_groups: slope={slope_sc:.4f}, intercept={intercept_sc:.4f}, "
                            f"R²={r2_root_sc**2:.4f}, p={p_lin_sc:.4f}\n")
                except Exception as ols_e:
                    f.write(f'  OLS failed: {ols_e}\n')
                f.write('  Weber fractions: see gradient in table above.\n')
                _d4d = [((np.mean(pp_arrs_by_name[ws]) - 12.0) / np.std(pp_arrs_by_name[ws], ddof=1))
                        if ws in pp_arrs_by_name and len(pp_arrs_by_name[ws]) >= 2
                           and np.std(pp_arrs_by_name[ws], ddof=1) > 0
                        else float('nan') for ws in weber_seqs_c]
                _v4d = [(n, d) for n, d in zip(ng_vals_sc, _d4d) if not np.isnan(d)]
                if len(_v4d) >= 2:
                    _rho4d, _p4d = stats.spearmanr([v[0] for v in _v4d], [v[1] for v in _v4d])
                    f.write(f"  Spearman rho(n_groups, Cohen_d) = {_rho4d:.4f},  p = {_p4d:.4f}\n")
                else:
                    f.write('  Spearman rho skipped — not enough valid Cohen\'s d values.\n')
            else:
                f.write('  Not enough sequences for regression.\n')
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 4.c failed: {e}\n')

        # ── Section 4b: Spearman rho(n_groups, Cohen's d) ────────────────────
        try:
            spearman_seqs    = ['Repetition-2', 'Repetition-3', 'Repetition-4']
            spearman_ngroups = {'Repetition-2': 6, 'Repetition-3': 4, 'Repetition-4': 3}
            ng_sp, d_sp = [], []
            for ws in spearman_seqs:
                if ws not in pp_arrs_by_name:
                    continue
                pp_w = pp_arrs_by_name[ws]
                if len(pp_w) < 2:
                    continue
                sd_w = np.std(pp_w, ddof=1)
                if sd_w > 0:
                    ng_sp.append(spearman_ngroups[ws])
                    d_sp.append((np.mean(pp_w) - 12.0) / sd_w)
            if len(ng_sp) == 3:
                rho, p_rho = stats.spearmanr(ng_sp, d_sp)
                f.write(f"\n  Spearman rho(n_groups, Cohen_d) = {rho:.4f},  p = {p_rho:.4f}\n")
            else:
                f.write('\n  Spearman rho skipped — fewer than 3 structured sequences found.\n')
        except Exception as e:
            f.write(f'  [!] Spearman rho failed: {e}\n')

        # ── Section 5: Rep-2 bimodal peak locations ───────────────────────────
        try:
            from scipy.signal import find_peaks as _fp
            f.write('\nSECTION 5 — Rep-2 bimodal peak locations (trial-level frequency histogram)\n')
            f.write('-' * 130 + '\n')
            rep2_name = 'Repetition-2'
            if rep2_name in freqs_by_name:
                freq2  = freqs_by_name[rep2_name]
                h2_max = freq2.max()
                peaks2, _ = _fp(freq2, height=0.15 * h2_max, distance=2)
                f.write(f"  Peak detection: height >= 15% of max ({0.15*h2_max:.1f}), distance >= 2\n")
                f.write(f"  Detected {len(peaks2)} peak(s):\n")
                for pk in peaks2:
                    count = int(freq2[pk])
                    f.write(f"    length = {pk:>2},  count = {count:>4}\n")
            else:
                f.write(f'  {rep2_name} not found in dataset.\n')
            sections_ok += 1
        except Exception as e:
            f.write(f'  [!] Section 5 failed: {e}\n')

        # ── Summary ───────────────────────────────────────────────────────────
        f.write('\n' + '=' * 130 + '\n')
        f.write(f'{sections_ok} / 7 sections completed successfully.\n')
        
def plot_all_length(data,path,sequence_list=seq_name_list,nb_rows=5,nb_cols=5,figsize=(25,25)):
    ### Plotting 
    # Create the figure and axes
    fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=figsize)
    max_y=140
    plot_index=0 #this is the index for all_mean_timings which differs in the length and increments if will take from index of plots
    name_index=0 #tracks the name of the sequence to display

    sequence_list_renamed = [
            "Rep-Local" if x == "control NoGlobal nested" else 
            "Rep-Global" if x == "control NoLocal nested" else x 
            for x in sequence_list
        ]
    for index, ax in enumerate(axes.flat):
        # Holders
        holder_length=[]
        length_frequency=[]

        # Get the length of answers for one type of sequence
        builder_frequency=np.zeros(18)
        for index, row in data[data['seq_name']==sequence_list[plot_index]].iterrows():
            this_length=len(row['sequences_response'])
            holder_length.append(this_length)
            if this_length<18:
                builder_frequency[this_length]+=1
        length_frequency.append(builder_frequency)
        


        ax.set_title(f'{sequence_list_renamed[name_index]} : {dict_expressions[sequence_list[name_index]]}', fontweight = "bold", fontsize = 18)
        ax.bar(np.arange(0,18,1),builder_frequency, align='center')
        top3_idx = np.argsort(builder_frequency)[-3:]
        for idx in top3_idx:
            if builder_frequency[idx] > 0:
                ax.text(idx, builder_frequency[idx] + 1, str(int(builder_frequency[idx])),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(np.arange(0,19,1),labels=np.arange(0,19,1))
        ax.set_ylim(0,max_y)
        # -- GO to next iteration
        plot_index+=1
        name_index+=1

    # Save and show the plot
    plt.savefig(f'{path}/all_length_distribution_subplots.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
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
        plt.savefig(f'{path}/interclick/individual/median/median_interclicks_subplots_{seq_name_list[index]}.pdf', bbox_inches='tight', dpi=_adaptive_dpi())    
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
    

def chunking_base_interclick(data, path, break_duration=30):
    """
    Tags chunking trial by trial for each participant with a correct answer, identifying chunk boundaries based on interclick times.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the sequence data with interclick times and performance metrics.
    path (str): Path to save the generated plots.
    break_duration (int, optional): Duration to define a break between chunks. Defaults to 30.
    
    The function processes each sequence for each participant, determining chunk boundaries where interclick times exceed the mean.
    For each trial, it generates an array indicating chunk boundaries and aggregates these to visualize common boundaries across trials.
    
    Steps:
    1. Filter the data for each sequence and successful performance.
    2. Calculate interclick time differences to determine chunk boundaries.
    3. Aggregate chunk boundaries across trials.
    4. Plot and save the sum of interclick times to visualize chunk boundaries.
    
    Example:
    For a sequence with 3 items repeated as [0,0,1,0,0,1,0,0,1,0,0], the function would identify chunking as ABC.ABC.ABC.ABC.
    
    The output is a plot showing the sum of interclick times, saved as a pdf file in the specified path.
    """
    for index_name in range(len(seq_name_list)):
        one_sequence_holder = []
        subset_data = data[(data['seq_name'] == seq_name_list[index_name]) & (data['performance'] == "success")]
        
        for index, row in subset_data.iterrows():
            row_interclick = row['interclick_time']
            mean_inter = np.mean(row_interclick)
            one_trial_holder = []
            for num_i in range(len(row_interclick)):
                if num_i == 0:
                    one_trial_holder.append(0)
                elif row_interclick[num_i] - row_interclick[num_i - 1] <= break_duration:
                    one_trial_holder.append(0)
                else:
                    one_trial_holder.append(1)
            one_sequence_holder.append(one_trial_holder)
    
        # Sum all those arrays. Peaks signify boundaries (how will we define peaks?)
        sum_holder = np.sum(one_sequence_holder, axis=0)
    
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(range(0, len(sum_holder)), sum_holder, color='skyblue')
        plt.xlabel('Position in Sequence')
        plt.ylabel(f'Sum of interclick over the mean \n(within trial)',rotation=0, labelpad=y_label_pad)
        plt.title(f'{seq_name_list[index_name]} - {list_seq_expression[index_name]}')
        plt.xticks(ticks=[i - 0.5 for i in range(0, 12)], labels=[i for i in alpha_seq_expression[index_name]])
        plt.xlim(xmin=-1, xmax=11)
            
        plt.grid(True)
        #/Users/et/Documents/UNICOG/2-Experiments/memocrush/Figures/interclick/differentials
        plt.savefig(f'{path}/interclick/differentials/{index_name}_differential_interclick_{seq_name_list[index_name]}.pdf', bbox_inches='tight', dpi=_adaptive_dpi())
        plt.show()

        
def chunking_base_interclick_target(data,name,target=None):
    """
    The idea is to tag chunking trial by trial for each participant with a correct answer.
    To determine chunk boundaries, we observe that the difference in interclick times is quick
    (inferior or equal to the mean) for intrachunk elements, and small pauses determine chunk boundaries (superior to the mean).
    
    For a given sequence, for each trial:
    - We obtain an array of size 11. 
    - For example for 3 items rep: [0,0,1,0,0,1,0,0,1,0,0] would mean a chunking of type ABC.ABC.ABC.ABC
    """
    one_sequence_holder = []
    subset_data = data[(data['seq_name'] == name) & (data['performance'] == "success")]
    
    for index, row in subset_data.iterrows():
        row_interclick=row['interclick_time']
        mean_inter = np.mean(row_interclick)
        one_trial_holder = []
        for num in row_interclick:
            if num <= mean_inter:
                one_trial_holder.append(0)
            else:
                one_trial_holder.append(1)
        one_sequence_holder.append(one_trial_holder)

        if one_trial_holder == target:
            print(f'index: {index}. \ninterclicks: {row_interclick}.\ntarget: {target}.\nmean: {mean_inter}\n')
            print('-------------')
            
    if target==None:
        print(pd.Series(one_sequence_holder).value_counts()[:40])

    # Sum all those arrays. Peaks signify boundaries (how will we define peaks?)
    sum_holder = np.sum(one_sequence_holder, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(sum_holder) + 1), sum_holder, color='skyblue')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Sum of Interclick Times',rotation=0, labelpad=y_label_pad)
    plt.title(name)
    plt.xticks(range(1, len(sum_holder) + 1))
    plt.grid(True)
    plt.show()

    return sum_holder

def error_rate_order_first_tokens(data, sequence_selection=range(len(seq_name_list))):
    """
    Calculate and print the error rates for the first n items of sequences in the dataset.

    This function analyzes sequences within the provided data, calculating the error rates
    for the first n items in each sequence. It prints the error rates along with relevant
    sequence information. Also generates error-bars

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing the sequence data. It must have the following columns:
        - 'seq_name' : The name or identifier of the sequence.
        - 'sequences_structure' : The structure of the sequence as a list or array.
        - 'comparable_temp' : The comparable sequence items to be analyzed.
    order (bool): If True, will compare the first chunks of sequences (original and comparable response),
                    if False, will compare the set of first chunks (order has no importance in this case).

    Notes:
    ------
    The function prints a summary of the error rates for the first n items in each sequence,
    where n is determined by the unique items in the sequence up to the first repeated item.
    It also provides additional context through printed messages based on specific conditions
    for each sequence.

    The `print_bar` list is used to format the printed output and give additional warnings
    or highlights for specific sequences.
    
    """

    print('*** Comment: It is interesting to compare together => Repetition-4, Mirror-Rep, Mirror-NoRep, subprogram-V1 and their controls (they all start with the same items)\n\n')
    print('This is the Error Rates. Set indicates error rate on unordered items of the first chunk. Differential is the difference between the measure of exact chunk matching and unordered chunk matching\n\n\n')
    
    print_bar = [0, 1,
                 0, 1, 
                 0, 2,
                 0, 0, 3,
                 0, 3,
                 0, 1,
                 0, 1,
                 0, 1,
                 0, 1,
                 0, 1,
                 0, 1,
                 0, 1]

    # How many index to go back to compare the sequence
    comparison_index=[0,1,
                      0,1,
                      0,1,
                      0,1,2,
                      0,1,
                      0,1,
                      0,1,
                      0,1,
                      0,1,
                      0,1,
                      0,1,
                      0,1]
    
    all_error_rates_seq=[]
    all_error_rates_set=[]

    all_mean_per_participant_error_rates=[]
    all_sem_per_participant_error_rates=[]
    
    all_mean_per_participant_error_rates_set=[]
    all_sem_per_participant_error_rates_set=[]
    
    # For each sequence, specify the value of n=number of token, to compute the error
    
    number_confusion=[2,2, # Repetition-2
                      3,3, # Repetition-3
                      4,4, # Repetition-4
                      6,6,6, # Nested Repetition (hardly relevant)
                      4,4, # Play-4
                      3,3, # subprogram-V1
                      4,4, # subprogram-V2
                      4,4, # index-i (not relevant)
                      4,4, # Play-1 (hardly relevant)
                      3,3, # Insertion / Suppression (hardly relevant)
                      4,4, # Mirror-Rep
                      4,4] # Mirror-NoRep
    
    # use a dictionnary to be able to choose which sequences are considered
    number_confusion={
        seq_name_list[0]:2,
        seq_name_list[1]:2,
        
        seq_name_list[2]:3,
        seq_name_list[3]:3,
        
        seq_name_list[4]:4,
        seq_name_list[5]:4,
        
        seq_name_list[6]:6,
        seq_name_list[7]:6,
        seq_name_list[8]:6,
        
        seq_name_list[9]:4,
        seq_name_list[10]:4,
        
        seq_name_list[11]:3,
        seq_name_list[12]:3,
        
        seq_name_list[13]:4,
        seq_name_list[14]:4,
        
        seq_name_list[15]:4,
        seq_name_list[16]:4,
        
        seq_name_list[17]:4,
        seq_name_list[18]:4,
        
        seq_name_list[19]:3,
        seq_name_list[20]:3,
        
        seq_name_list[21]:4,
        seq_name_list[22]:4,
        
        seq_name_list[23]:4,
        seq_name_list[24]:4,
    }
    
    # Gather complexity numbers
    holder_complexity=[]
    
    if len(sequence_selection)<25:
        print(f'sequences considered: {[seq_name_list[i] for i in sequence_selection]}')
        
    for index_name in sequence_selection:
        
        name=seq_name_list[index_name]
        subset = data[data['seq_name'] == name]
        holder_complexity.append(subset['LoT Complexity'].iloc[0])
        
        # -- For comparison, get the shown first chunk
        original_first_chunk = subset['sequences_structure'].to_numpy()[0][:number_confusion[name]]
        
        # For each sequence
        error_rates_seq=[]
        error_rates_set=[]
        
        # -- For each participant
        for IDs in subset['participant_ID'].unique():
            # Define the subset dataframe for the given sequence and given participant
            subset_participant=subset[subset['participant_ID']==IDs]
            # -- number of trials
            nb_trials=len(subset_participant)
        
            # -- Test if there is at least one trial for this sequence (participants who did exp1 don't have trials on sequences of exp2)
            if nb_trials!=0:
        
                # Reset number of success
                nb_success=0

                
                # -- Total success for the first chunk of the sequence
                nb_success=np.sum(subset_participant['comparable_temp'].apply(lambda x: x[:number_confusion[name]] == original_first_chunk).astype(int))

                # -- Total success for the first unordered SET of items of the chunk of the sequence
                nb_success_set=np.sum(subset_participant['comparable_temp'].apply(lambda x: set(x[:number_confusion[name]]) == set(original_first_chunk)).astype(int))
                    
                # -- Divided by number of trials (2)
                error_rates_seq.append(100*(1-nb_success/nb_trials))
                error_rates_set.append(100*(1-nb_success_set/nb_trials))
                
            

        # -- Put all participants success rates together in one big array per sequence
        all_error_rates_seq.append(error_rates_seq)
        all_error_rates_set.append(error_rates_set)
        
        
        # -- Compute mean error rate and Standard error of the mean on first chunk
        mean_error_holder=np.mean(error_rates_seq)
        sem_holder=np.std(error_rates_seq)/np.sqrt(len(error_rates_seq))
        
        # Same for unordered sets
        mean_error_holder_set=np.mean(error_rates_set)
        sem_holder_set=np.std(error_rates_set)/np.sqrt(len(error_rates_set))

        # -- Append them to a bigger array
        all_mean_per_participant_error_rates.append(mean_error_holder)
        all_sem_per_participant_error_rates.append(sem_holder) 
        all_mean_per_participant_error_rates_set.append(mean_error_holder_set)
        all_sem_per_participant_error_rates_set.append(sem_holder_set) 
        
        size_subset = len(data[data['seq_name'] == name])
        seq_expression = subset['sequences_structure'].to_numpy()[0]
        
        print(f"## {name} : {subset['sequences_structure'].to_numpy()[0]} __ {original_first_chunk} (size: {number_confusion[name]})\n#")
        print(f'CHUNK: {round(mean_error_holder,2)}, -- SEM : {round(sem_holder,2)}')
        print(f'Unordered set: {round(mean_error_holder_set,2)} -- SEM: {round(sem_holder_set,2)}\n')

        # Adjust length of samples

        if comparison_index[index_name] !=0:
            #################
            # -- Prepare data
            ##
            # > Construct arrays
            #
            # All error rates
            sample1_all=all_error_rates_seq[index_name]
            sample2_all=all_error_rates_seq[index_name-1]

            # Mean Error rates
            sample1_means=all_mean_per_participant_error_rates[index_name]
            sample2_means=all_mean_per_participant_error_rates[index_name-comparison_index[index_name]]

            # All per participant, mean error rates
            sample1_perParticipant_means=all_mean_per_participant_error_rates_set[index_name]
            sample2_perParticipant_means=all_mean_per_participant_error_rates_set[index_name-comparison_index[index_name]]

            # All SET error rates
            sample1_set_error=all_error_rates_set[index_name]
            sample2_set_error=all_error_rates_set[index_name-1]

            # > Ensure data is of the same size
            #
            len_1=np.shape(sample1_all)[0]
            len_2=np.shape(sample2_all)[0]

            if len_1!=len_2:
                min_length=min(len_1,len_2)
                sample1_all=sample1_all[:min_length]
                sample2_all=sample2_all[:min_length]
            
            len_1=np.shape(sample1_set_error)[0]
            len_2=np.shape(sample2_set_error)[0]

            if len_1!=len_2:
                min_length=min(len(sample1_set_error),len(sample2_set_error))
                sample1_set_error=sample1_set_error[:min_length]
                sample2_set_error=sample2_set_error[:min_length]

            ###################
            # --  Compute Stats
            ##
            comparison_err=sample1_means-sample2_means
            print(f'Difference in err rate: {round(comparison_err,2)}')
        
            stat, p_value = wilcoxon(sample1_all, sample2_all)
            diffs = np.array(sample1_all) - np.array(sample2_all)
            non_zero_diffs = diffs[diffs != 0]
            n = len(non_zero_diffs)
            if n > 0:
                mean_W = n * (n + 1) / 4
                std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                z = (stat - mean_W) / std_W
                r = abs(z) / np.sqrt(n)
                print(f'Cumulative Accuracy - Wilcoxon-stat : {stat}, p-value : {p_value}, effect size r = {round(r, 3)}\n')
            else:
                print(f'Cumulative Accuracy - Wilcoxon-stat : {stat}, p-value : {p_value}, effect size: NA (all diffs zero)\n')
            
            comparison_err_set=sample1_perParticipant_means-sample2_perParticipant_means
            print(f'Difference in err rate SET: {round(comparison_err_set,2)}')

            stat, p_value = wilcoxon(sample1_set_error, sample2_set_error)
            diffs_set = np.array(sample1_set_error) - np.array(sample2_set_error)
            non_zero_diffs_set = diffs_set[diffs_set != 0]
            n_set = len(non_zero_diffs_set)
            if n_set > 0:
                mean_W_set = n_set * (n_set + 1) / 4
                std_W_set = np.sqrt(n_set * (n_set + 1) * (2 * n_set + 1) / 24)
                z_set = (stat - mean_W_set) / std_W_set
                r_set = abs(z_set) / np.sqrt(n_set)
                print(f'SET Accuracy - Wilcoxon-stat : {stat}, p-value : {p_value}, effect size r = {round(r_set, 3)}\n')
            else:
                print(f'SET Accuracy - Wilcoxon-stat : {stat}, p-value : {p_value}, effect size: NA (all diffs zero)\n')
            
        
        if print_bar[index_name] != 0:
                print('---------------------------------------------------------\n\n')
    
                if print_bar[index_name] == 2:
                    print('**Warning : This metric is not relevant for nested sequences**')
                if print_bar[index_name] == 3:
                    print(' >> Interesting case')

    # Adding Pearson R for the correlation
    pearson_corr, p_value = stats.pearsonr(all_mean_per_participant_error_rates, holder_complexity)
    pearson_corr_set, p_value_set = stats.pearsonr(all_mean_per_participant_error_rates_set, holder_complexity)
    
    print(f'[LoT Complexity // Absolute Error first chunk ] : Pearson Correlation is : {pearson_corr}, p_value: {p_value}')
    print(f'[LoT Complexity // Error first chunk UNORDERED ] : Pearson Correlation is : {pearson_corr_set}, p_value: {p_value_set}')
    
    
    # Calculate and print the overall average error rates and SEMs for sequences
    overall_mean_error_rate_seq = np.mean(all_mean_per_participant_error_rates)
    overall_sem_error_rate_seq = np.mean(all_sem_per_participant_error_rates)
    
    overall_mean_error_rate_set = np.mean(all_mean_per_participant_error_rates_set)
    overall_sem_error_rate_set = np.mean(all_sem_per_participant_error_rates_set)
    
    print('\nOverall Averaged Mean Error Rates and SEMs')
    print('-----------------------------------------')
    print(f'Average Error Rate (First Chunks): {overall_mean_error_rate_seq:.2f}')
    print(f'SEM (First Chunks): {overall_sem_error_rate_seq:.2f}')
    print(f'Average Error Rate (Sets): {overall_mean_error_rate_set:.2f}')
    print(f'SEM (Sets): {overall_sem_error_rate_set:.2f}')

def compute_token_errors(data):
    
    """
    Computes and prints statistics on token errors for each participant in the dataset.

    The function performs the following steps:
    1. Iterates over each unique participant in the data.
    2. For each participant, calculates the number and types of token errors:
        - Total token errors (`TokenErr`).
        - Forgotten tokens (`TokenErr_forg` without `TokenErr_add`).
        - Added tokens (`TokenErr_add` without `TokenErr_forg`).
        - Substituted tokens (both `TokenErr_forg` and `TokenErr_add`).
    3. Computes the error rates as a percentage of the number of responses for each participant.
    4. Aggregates the results across all participants and calculates the mean and standard error of the mean (SEM) for each error type.
    5. Prints the mean error rates and SEMs for all participants.

    Parameters:
    - data (pd.DataFrame): A dataset containing token error information. It must include the following columns:
        - 'participant_ID': Unique identifier for each participant.
        - 'TokenErr': Indicator for any token error.
        - 'TokenErr_forg': Indicator for a forgotten token.
        - 'TokenErr_add': Indicator for an added token.

    Prints:
    - Mean and SEM for total token errors, forgotten tokens, added tokens, and substituted tokens across all participants.
    """
    # Initialize lists outside the loop
    all_tokenErr_total_participant = []
    all_tokenErr_forg_participant = []
    all_tokenErr_added_participant = []
    all_tokenErr_substitution_participant = []

    for participant in data['participant_ID'].unique():
        subset_participant = data[data['participant_ID'] == participant].copy()
        nb_responses = len(subset_participant)

        # Get number of token errors per type
        tokenErr_total = len(subset_participant[subset_participant['TokenErr']])
        tokenErr_forg = len(subset_participant[(subset_participant['TokenErr_forg']) & (~subset_participant['TokenErr_add'])])
        tokenErr_added = len(subset_participant[(subset_participant['TokenErr_add']) & (~subset_participant['TokenErr_forg'])])
        tokenErr_substitution = len(subset_participant[(subset_participant['seq'].apply(set) != subset_participant['sequences_response'].apply(set)) & 
                                (subset_participant['seq'].apply(len) == subset_participant['sequences_response'].apply(len))])

        #len(subset_participant[subset_participant['TokenErr_add'] & subset_participant['TokenErr_forg']])

        # Add values to the holders
        all_tokenErr_total_participant.append(tokenErr_total / nb_responses)
        all_tokenErr_forg_participant.append(tokenErr_forg / nb_responses)
        all_tokenErr_added_participant.append(tokenErr_added / nb_responses)
        all_tokenErr_substitution_participant.append(tokenErr_substitution / nb_responses)

    # Convert lists to numpy arrays and apply rounding afterward
    all_tokenErr_total_percentage = np.round(100 * np.array(all_tokenErr_total_participant), 3)
    all_tokenErr_forg_percentage = np.round(100 * np.array(all_tokenErr_forg_participant), 3)
    all_tokenErr_added_percentage = np.round(100 * np.array(all_tokenErr_added_participant), 3)
    all_tokenErr_substitution_percentage = np.round(100 * np.array(all_tokenErr_substitution_participant), 3)

    # Compute means
    mean_all_tokenErr_total_percentage = np.round(np.mean(all_tokenErr_total_percentage, axis=0), 3)
    mean_all_tokenErr_forg_percentage = np.round(np.mean(all_tokenErr_forg_percentage, axis=0), 3)
    mean_all_tokenErr_added_percentage = np.round(np.mean(all_tokenErr_added_percentage, axis=0), 3)
    mean_all_tokenErr_substitution_percentage = np.round(np.mean(all_tokenErr_substitution_percentage, axis=0), 3)

    # Compute standard error of the mean (sem)
    sem_mean_all_tokenErr_total_percentage = np.round(np.std(all_tokenErr_total_percentage, axis=0) / np.sqrt(len(all_tokenErr_total_percentage)), 3)
    sem_all_tokenErr_forg_percentage = np.round(np.std(all_tokenErr_forg_percentage, axis=0) / np.sqrt(len(all_tokenErr_forg_percentage)), 3)
    sem_all_tokenErr_added_percentage = np.round(np.std(all_tokenErr_added_percentage, axis=0) / np.sqrt(len(all_tokenErr_added_percentage)), 3)
    sem_all_tokenErr_substitution_percentage = np.round(np.std(all_tokenErr_substitution_percentage, axis=0) / np.sqrt(len(all_tokenErr_substitution_percentage)), 3)

    
    # Print results
    print('################ ALL SEQUENCES ################')
    print(f'Per participant mean Token Error recorded in the responses, i.e., set(original)!=set(response) {mean_all_tokenErr_total_percentage}% sem: {sem_mean_all_tokenErr_total_percentage}. From which:')
    print(f'--- Token was forgotten {mean_all_tokenErr_forg_percentage}%, sem: {sem_all_tokenErr_forg_percentage}')
    print(f'--- Token was added {mean_all_tokenErr_added_percentage}%, sem: {sem_all_tokenErr_added_percentage}')
    print(f'--- Token was substituted {mean_all_tokenErr_substitution_percentage}%, sem: {sem_all_tokenErr_substitution_percentage}')
    print('###############################################\n')
    
    return all_tokenErr_total_percentage, all_tokenErr_forg_percentage, all_tokenErr_added_percentage,all_tokenErr_substitution_percentage
    
def analyze_token_errors(data_input, seq_name_list=seq_name_list, control_name_list = [], structured_name_list = []):
    """
    Analyzes token errors in a dataset, computes their proportions, and calculates Pearson correlations.

    Parameters:
    - data_input (pd.DataFrame): The main dataset containing sequences and token error information.

    Returns:
    - df_tokenErr (pd.DataFrame): A DataFrame containing error proportions and complexities for each sequence.
    - correlation_results (dict): A dictionary with Pearson correlation results between number of tokens / LoT complexity 
                                  and different types of token errors.
    
    This function performs the following steps:
    1. Computes total and specific types of token errors across all sequences.
    2. Calculates the proportions of these errors relative to the number of responses.
    3. Prints the overall results for all sequences.
    4. Iterates over each sequence in `seq_name_list` to compute and print token error statistics.
    5. Collects token error data and sequence complexity into holders.
    6. Constructs a DataFrame `df_tokenErr` with the collected data.
    7. Computes Pearson correlations between the number of tokens / LoT complexity and different types of token errors.
    8. Prints the correlation results and returns the DataFrame and correlation results dictionary.
    """
    
    # All Sequences
    data = data_input.copy()
    
    compute_token_errors(data)

    # Per individual sequences
    # Holder for Token Err values
    
    holder_token_nb = []
    holder_complexity = []
    holder_tokenErr = []
    holder_tokenForg = []
    holder_tokenAdded = []
    holder_tokenSubstitution = []

    
        
    for name in seq_name_list:
        
        data = data_input[data_input['seq_name'] == name].copy()
        token_nb = len(set(data['seq'].iloc[0]))
        complexity = data['LoT Complexity'].iloc[0]
        
        holder_token_nb.append(token_nb)
        holder_complexity.append(complexity)
        
        # Initialize lists outside the loop
        all_tokenErr_total_participant = []
        all_tokenErr_forg_participant = []
        all_tokenErr_added_participant = []
        all_tokenErr_substitution_participant = []
        
        for participant in data['participant_ID'].unique():

            subset_participant = data[data['participant_ID'] == participant].copy()
            nb_responses = len(subset_participant)
            

            # Get number of token errors per type
            tokenErr_total = len(subset_participant[subset_participant['TokenErr']])
            tokenErr_forg = len(subset_participant[(subset_participant['TokenErr_forg']) & (~subset_participant['TokenErr_add'])])
            tokenErr_added = len(subset_participant[(subset_participant['TokenErr_add']) & (~subset_participant['TokenErr_forg'])])
            tokenErr_substitution = len(subset_participant[(subset_participant['seq'].apply(set) != subset_participant['sequences_response'].apply(set)) & 
                                (subset_participant['seq'].apply(len) == subset_participant['sequences_response'].apply(len))])

            # Add values to the holders
            all_tokenErr_total_participant.append(tokenErr_total / nb_responses)
            all_tokenErr_forg_participant.append(tokenErr_forg / nb_responses)
            all_tokenErr_added_participant.append(tokenErr_added / nb_responses)
            all_tokenErr_substitution_participant.append(tokenErr_substitution / nb_responses)

        # Convert lists to numpy arrays and apply rounding afterward
        all_tokenErr_total_percentage = np.round(100 * np.array(all_tokenErr_total_participant), 3)
        all_tokenErr_forg_percentage = np.round(100 * np.array(all_tokenErr_forg_participant), 3)
        all_tokenErr_added_percentage = np.round(100 * np.array(all_tokenErr_added_participant), 3)
        all_tokenErr_substitution_percentage = np.round(100 * np.array(all_tokenErr_substitution_participant), 3)

        # Compute means
        mean_all_tokenErr_total_percentage = np.round(np.mean(all_tokenErr_total_percentage, axis=0), 3)
        mean_all_tokenErr_forg_percentage = np.round(np.mean(all_tokenErr_forg_percentage, axis=0), 3)
        mean_all_tokenErr_added_percentage = np.round(np.mean(all_tokenErr_added_percentage, axis=0), 3)
        mean_all_tokenErr_substitution_percentage = np.round(np.mean(all_tokenErr_substitution_percentage, axis=0), 3)

        # Compute standard error of the mean (sem)
        sem_mean_all_tokenErr_total_percentage = np.round(np.std(all_tokenErr_total_percentage, axis=0) / np.sqrt(len(all_tokenErr_total_percentage)), 3)
        sem_all_tokenErr_forg_percentage = np.round(np.std(all_tokenErr_forg_percentage, axis=0) / np.sqrt(len(all_tokenErr_forg_percentage)), 3)
        sem_all_tokenErr_added_percentage = np.round(np.std(all_tokenErr_added_percentage, axis=0) / np.sqrt(len(all_tokenErr_added_percentage)), 3)
        sem_all_tokenErr_substitution_percentage = np.round(np.std(all_tokenErr_substitution_percentage, axis=0) / np.sqrt(len(all_tokenErr_substitution_percentage)), 3)

        # Fill the holders
        holder_tokenErr.append(mean_all_tokenErr_total_percentage)
        holder_tokenForg.append(mean_all_tokenErr_forg_percentage)
        holder_tokenAdded.append(mean_all_tokenErr_added_percentage)
        holder_tokenSubstitution.append(mean_all_tokenErr_substitution_percentage)

        # Print results
        print(f'____________________>{name.upper()}<_____________')
        print(f'Token Error recorded in the responses, i.e., set(original)!=set(response) {round(mean_all_tokenErr_total_percentage, 3)}%, sem: {sem_mean_all_tokenErr_total_percentage}. From which:')
        print(f'--- Token was forgotten {round(mean_all_tokenErr_forg_percentage, 3)}%, sem: {sem_all_tokenErr_forg_percentage}')
        print(f'--- Token was added {round(mean_all_tokenErr_added_percentage, 3)}%, sem: {sem_all_tokenErr_added_percentage}')
        print(f'--- Token was substituted {round(mean_all_tokenErr_substitution_percentage, 3)}%, sem: {sem_all_tokenErr_substitution_percentage}')
        print('_______________________________________________\n')
    
    df_tokenErr = pd.DataFrame({
        'seq_name': seq_name_list,
        'token_nb': holder_token_nb,
        'LoT_complexity': holder_complexity,
        'tokenErr': holder_tokenErr,
        'tokenForg': holder_tokenForg,
        'tokenAdd': holder_tokenAdded,
        'tokenSubstitution': holder_tokenSubstitution
    })

    # Compute correlations
    correlation_results = {}
    
    pearson_corr, p_value = stats.pearsonr(df_tokenErr['token_nb'], df_tokenErr['tokenErr'])
    correlation_results['token_nb_tokenErr'] = (pearson_corr, p_value)
    print(f'[token_nb // Token Error ] : Pearson Correlation is : {pearson_corr}, p_value: {p_value}')

    pearson_corr, p_value = stats.pearsonr(df_tokenErr['token_nb'], df_tokenErr['tokenForg'])
    correlation_results['token_nb_tokenForg'] = (pearson_corr, p_value)
    print(f'[token_nb // Token Forgetting ] : Pearson Correlation is : {pearson_corr}, p_value: {p_value}')

    pearson_corr, p_value = stats.pearsonr(df_tokenErr['token_nb'], df_tokenErr['tokenAdd'])
    correlation_results['token_nb_tokenAdd'] = (pearson_corr, p_value)
    print(f'[token_nb // Token Addition ] : Pearson Correlation is : {pearson_corr}, p_value: {p_value}')

    pearson_corr, p_value = stats.pearsonr(df_tokenErr['LoT_complexity'], df_tokenErr['tokenErr'])
    correlation_results['LoT_complexity_tokenErr'] = (pearson_corr, p_value)
    print(f'[LoT_complexity // Token Error ] : Pearson Correlation is : {pearson_corr}, p_value: {p_value}')

    pearson_corr, p_value = stats.pearsonr(df_tokenErr['LoT_complexity'], df_tokenErr['tokenForg'])
    correlation_results['LoT_complexity_tokenForg'] = (pearson_corr, p_value)
    print(f'[LoT_complexity // Token Forgetting ] : Pearson Correlation is : {pearson_corr}, p_value: {p_value}')

    pearson_corr, p_value = stats.pearsonr(df_tokenErr['LoT_complexity'], df_tokenErr['tokenAdd'])
    correlation_results['LoT_complexity_tokenAdd'] = (pearson_corr, p_value)
    print(f'[LoT_complexity // Token Addition ] : Pearson Correlation is : {pearson_corr}, p_value: {p_value}')
    
    #####################################
    # Now comparing structure VS controls
    if not control_name_list or not structured_name_list:
        control_name_list = [name for name in seq_name_list if 'control' in name]
        structured_name_list = [name for name in seq_name_list if 'control' not in name]

    # 3. Final safety check: are they still empty?
    if not control_name_list or not structured_name_list:
        print("WARNING: One of the sequence lists is empty. Statistics will be skipped.")
        # Return early or handle the error safely
    else:
        print('Structured sequence list:', structured_name_list)
        print('Control sequence list:', control_name_list)
    
    control_dataset = data_input[data_input['seq_name'].isin(control_name_list)]
    structured_dataset = data_input[data_input['seq_name'].isin(structured_name_list)]
    
    print('\nSTRUCTURED DATASET\n')
    structured_tokenErr_total_percentage, structured_tokenErr_forg_percentage, structured_tokenErr_added_percentage,structured_tokenErr_substitution=compute_token_errors(structured_dataset)
    
    print('\nCONTROL DATASET \n')
    control_tokenErr_total_percentage, control_tokenErr_forg_percentage, control_tokenErr_added_percentage,control_tokenErr_substitution=compute_token_errors(control_dataset)

    print('------ ALL Token Errors')
    stat, p_value = wilcoxon(structured_tokenErr_total_percentage,control_tokenErr_total_percentage)
    print(f'\nCompared per participant mean token errors STRUCTURED // CONTROL datasets')
    print(f'Wilcoxon stat: {stat}, p_value : {p_value}\n')
    
    print('------ Token Forgetting')
    stat, p_value = wilcoxon(structured_tokenErr_forg_percentage,control_tokenErr_forg_percentage)
    print(f'\nCompared per participant mean token errors STRUCTURED // CONTROL datasets')
    print(f'Wilcoxon stat: {stat}, p_value : {p_value}\n')
    
    print('------ Token Addition')
    stat, p_value = wilcoxon(structured_tokenErr_added_percentage,control_tokenErr_added_percentage)
    print(f'\nCompared per participant mean token errors STRUCTURED // CONTROL datasets')
    print(f'Wilcoxon stat: {stat}, p_value : {p_value}\n')
    
    print('------ Token Substitution')
    stat, p_value = wilcoxon(structured_tokenErr_substitution,control_tokenErr_substitution)
    print(f'\nCompared per participant mean token errors STRUCTURED // CONTROL datasets')
    print(f'Wilcoxon stat: {stat}, p_value : {p_value}\n')
    
    #return df_tokenErr, correlation_results

# ---------------------------------------
# ***** Transition probabilities ********
# ---------------------------------------   

def calculate_transition_probabilities(seq):
    # We output [[tp(0,0), tp(0,1), tp(0,2), ..., tp(0,5)],
    #            ....
    #           [[tp(5,0), tp(5,1), ..., tp(5,5)]]
    #
    # 0. Create an array of zeros
    transition_counter=np.zeros((6,6))
    probabilities=np.zeros((6,6))
    nb_total_transitions=0
    
    # 1. Compute the number of Transitions (0,1), (1,0) etc.
    for i in range(len(seq)-1):
        transition_counter[seq[i],seq[i+1]]+=1
        nb_total_transitions+=1
        
    # 2. Fill in the Transition Probabilities matrix
    for i in range(np.shape(transition_counter)[0]):
        for j in range(len(transition_counter[i])):
            probabilities[i,j]=transition_counter[i,j]/nb_total_transitions
            
    return probabilities

def average_response_transition_probabilities(data,path):
    # Compute the mean transition probabilities in answers for one particular sequence
    averaged_tp=[]
    original_tps=[]
    
    for index_name in range(len(seq_name_list)):
        name=seq_name_list[index_name]
        original_tps.append(calculate_transition_probabilities([int(char) for char in real_mapping[name]]))
        subset_data_sequence=data[data['seq_name']==name]
    
        # Holder for all computed complexities
        sum_array=np.zeros((6,6))
    
        # Now compute all the transition probability arrays
        for i in range(len(subset_data_sequence)):
            sum_array+=calculate_transition_probabilities(subset_data_sequence['response_structure'].iloc[i])
        
        averaged_tp.append(sum_array/len(subset_data_sequence))
    
    # Compute how much the averaged responses' TPs diverge from originals
    holder_heatmap_TP=np.array(averaged_tp)-np.array(original_tps)
    holder_heatmap_TP_structured=[]
    holder_heatmap_TP_controls=[]
    
    index_choice=[0,1,0,1,0,1,0,1,2,0,1,0,1,0,1,0,1,0,1,2,2,0,1,0,1,0,1]
    for index in range(len(seq_name_list)):
        if index_choice[index]==2:
            pass
        elif index_choice[index]==1:
            holder_heatmap_TP_controls.append(holder_heatmap_TP[index])
        elif index_choice[index]==0:
            holder_heatmap_TP_structured.append(holder_heatmap_TP[index])

    holder_heatmap_TP_structured=np.array(holder_heatmap_TP_structured)
    holder_heatmap_TP_controls=np.array(holder_heatmap_TP_controls)
    
    # Labels for the axes
    labels = ['A', 'B', 'C', 'D', 'E', 'F']

   # Create a diverging color palette for the heatmap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # In your loop where you create the heatmap:
    for i in range(len(holder_heatmap_TP)):
        name=seq_name_list[i]
        plt.figure(figsize=(5, 5))
        ax = sns.heatmap(holder_heatmap_TP[i], annot=True, fmt='.3f', linewidth=0.5, cmap=cmap, center=0)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=0)
        plt.title(f"TP: {alpha_seq_expression[i].upper()}", pad=padding_size)
        plt.savefig(f'{path}/TP_heatmap/{i}_TP_heatmap_{name}.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
        plt.show()
    
    # Heatmap for averaged TP for responses to all sequences
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(np.mean(holder_heatmap_TP,axis=0), annot=True, fmt='.3f', linewidth=0.5, cmap=cmap, center=0)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    plt.title(f"TP: Average over all sequences", pad=padding_size)
    plt.savefig(f'{path}/TP_heatmap/{i+1}_TP_heatmap_ALL.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
    
    # Heatmap for averaged TP for responses to CONTROL sequences
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(np.mean(holder_heatmap_TP_controls,axis=0), annot=True, fmt='.3f', linewidth=0.5, cmap=cmap, center=0)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    plt.title(f"TP: Average over CONTROL sequences", pad=padding_size)
    plt.savefig(f'{path}/TP_heatmap/{i+2}_TP_heatmap_controls.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
    
    # Heatmap for averaged TP for responses to STRUCTURED sequences
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(np.mean(holder_heatmap_TP_structured,axis=0), annot=True, fmt='.3f', linewidth=0.5, cmap=cmap, center=0)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    plt.title(f"TP: Average over STRUCTURED sequences", pad=padding_size)
    plt.savefig(f'{path}/TP_heatmap/{i+3}_TP_heatmap_structured.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
    
    # Run Normality test : STRUCTURED
    print('Kolmogorov-Smirnov test for normal distribution of responses TP on STRUCTURED sequences')
    stat, p_value = kstest(holder_heatmap_TP_structured.flatten(), 'norm')
    print(f'statistic: {stat}, p-value: {p_value}')
    print('Sample size: ',len(holder_heatmap_TP_structured)) 
    
    # Run Normality test : CONTROLS
    print('Kolmogorov-Smirnov test for normal distribution of responses TP on CONTROL sequences')
    stat, p_value = kstest(holder_heatmap_TP_controls.flatten(), 'norm')
    print(f'statistic: {stat}, p-value: {p_value}')
    print('Sample size: ',len(holder_heatmap_TP_controls)) 
    
    # Run a statistical test
    stat, p_value = wilcoxon(holder_heatmap_TP_structured.flatten(),holder_heatmap_TP_controls.flatten())
    print(f'\nCompared responses TP to structured vs control sequences')
    print(f'Wilcoxon stat: {stat}, p_value : {p_value}')

# ---------------------------------------
# ***** Primacy / Recency Effect ********
# ---------------------------------------  
def primacy_recency(data):
    # -- We want to check how accurate are the first three and last three items of the sequence --
    #
    # Set counters
    counter_primacy=[0,0,0]
    counter_recency=[0,0,0]

    # Loop through the rows and compare
    for index,row in data.iterrows():
        for k in range(3):
            counter_primacy[k]+=(row['seq'][k]==row['sequences_response'][k])
            counter_recency[-k-1]+=(row['seq'][-k-1]==row['sequences_response'][-k-1])

    # Turn into percentages
    primacy_effect_overall=np.array(counter_primacy)/len(data)
    recency_effect_overall=np.array(counter_recency)/len(data)
    
    print('Overall Primacy Effects (1st, 2nd, 3rd) positions accuracy: ', primacy_effect_overall)
    print('Overall Recency Effects (Last, 2nd to last, 3rd to last) positions accuracy: ', recency_effect_overall)
    
    for name in seq_name_list:
        print(f'\n --- {name.upper()}')
        subset_data=data[data['seq_name']==name]
        
        # Set counters
        counter_primacy=[0,0,0]
        counter_recency=[0,0,0]

        # Loop through the rows and compare
        for index,row in subset_data.iterrows():
            for k in range(3):
                counter_primacy[k]+=(row['seq'][k]==row['sequences_response'][k])
                counter_recency[-k-1]+=(row['seq'][-k-1]==row['sequences_response'][-k-1])

        # Turn into percentages
        primacy_effect_subset=np.round(np.array(counter_primacy)/len(subset_data),3)
        recency_effect_subset=np.round(np.array(counter_recency)/len(subset_data),3)
        
        print(f'Primacy Effects: ', primacy_effect_subset)
        print(f'Recency Effects: ', recency_effect_subset)

def calculate_primacy_recency(data):
    # -- We want to check how accurate are the first three and last three items of the sequence --
    #
    # For each participant
    primacy_effect=[]
    recency_effect=[]
    
    for participant in data['participant_ID'].unique():
        subset_participant=data[data['participant_ID']==participant].copy()
        # Set counters
        counter_primacy=[0,0,0]
        counter_recency=[0,0,0]

        # Loop through the rows and compare
        for index,row in subset_participant.iterrows():
            for k in range(3):
                counter_primacy[k]+=(row['seq'][k]==row['sequences_response'][k])
                counter_recency[-k-1]+=(row['seq'][-k-1]==row['sequences_response'][-k-1])

        # Turn into percentages for the current participant
        primacy_effect_current = np.array(counter_primacy) / len(subset_participant)
        recency_effect_current = np.array(counter_recency) / len(subset_participant)

        # Store the results for all participants
        primacy_effect = np.vstack((primacy_effect, primacy_effect_current)) if len(primacy_effect) > 0 else primacy_effect_current
        recency_effect = np.vstack((recency_effect, recency_effect_current)) if len(recency_effect) > 0 else recency_effect_current
        
       
    
    mean_primacy_effect=np.round(np.mean(primacy_effect,axis=0),3)
    mean_recency_effect=np.round(np.mean(recency_effect,axis=0),3)
    
    # Compute standard error of the mean
    sem_primacy =np.round(np.std(primacy_effect, axis=0) / np.sqrt(len(primacy_effect)),3)
    sem_recency =np.round(np.std(recency_effect, axis=0) / np.sqrt(len(recency_effect)),3)
    
    return (mean_primacy_effect,mean_recency_effect,sem_primacy,sem_recency)
    
    
def primacy_recency_per_participant(data, seq_name_list=seq_name_list):
    mean_primacy_effect,mean_recency_effect,sem_primacy,sem_recency= calculate_primacy_recency(data[data['seq_name'].isin(seq_name_list)])
    print(f'Overall Primacy Effects (1st, 2nd, 3rd) positions accuracy:\n {mean_primacy_effect}, SEM = {sem_primacy}')
    print(f'Overall Recency Effects (Last, 2nd to last, 3rd to last) positions accuracy:\n {mean_recency_effect}, SEM = {sem_recency}', )
        
    # Now for structured versus controls
    print('\n\n******************** Structured VS Controls *********************\n')
    control_name_list=[name for name in seq_name_list if 'control' in name]
    structured_name_list=[name for name in seq_name_list if not 'control' in name]
    
    control_dataset = data[data['seq_name'].isin(control_name_list)]
    structured_dataset = data[data['seq_name'].isin(structured_name_list)]
    
    mean_primacy_effect,mean_recency_effect,sem_primacy,sem_recency= calculate_primacy_recency(control_dataset)
    print(' --------------- Control Sequences')
    print(f'Primacy Effects:\n {mean_primacy_effect}, SEM = {sem_primacy}')
    print(f'Recency Effects:\n {mean_recency_effect}, SEM = {sem_recency}\n', )
    
    mean_primacy_effect,mean_recency_effect,sem_primacy,sem_recency= calculate_primacy_recency(structured_dataset)
    print(' --------------- Structured Sequences')
    print(f'Primacy Effects:\n {mean_primacy_effect}, SEM = {sem_primacy}')
    print(f'Recency Effects:\n {mean_recency_effect}, SEM = {sem_recency}\n', )
    
    # Now for each Sequences
    print('\n\n******************** Per Sequence *********************\n')
    
    for name in seq_name_list:
        print(f'\n --- {name.upper()}')
        subset_data=data[data['seq_name']==name]
        
        mean_primacy_effect,mean_recency_effect,sem_primacy,sem_recency= calculate_primacy_recency(subset_data)
        
        print(f'Primacy Effects:\n {mean_primacy_effect}, SEM = {sem_primacy}')
        print(f'Recency Effects:\n {mean_recency_effect}, SEM = {sem_recency}\n', )
    
    
# ---------------------------------------
# ***** First items accuracy ********
# ---------------------------------------      

def compute_accuracy(data,max_positions=12, structure=False):
    """
    Compute the accuracy of the first `n` items in a sequence for each participant.

    Parameters:
    - data: DataFrame containing the experimental data.
    - max_positions (int): Maximum number of positions to evaluate in the sequence (default is 12).
    - structure (bool): If True, use 'response_structure' column; if False, use 'sequences_response' column.

    Returns:
    - np.array: An array of accuracies for each participant across the first `n` items.
    """
    accuracy = []
    for participant in data['participant_ID'].unique():
        subset_participant = data[data['participant_ID'] == participant]
        total_nb_responses = len(subset_participant)
        participant_accuracy = []
        for i in range(1, max_positions + 1):
            correct_count = 0
            valid_responses = 0
            for index, row in subset_participant.iterrows():
                response = row['response_structure'] if structure else row['sequences_response']
                origin=row['sequences_structure'] if structure else row['seq']
                if len(origin) >= i and len(response) >= i:
                    valid_responses += 1
                    if origin[:i] == response[:i]:
                        correct_count += 1
            participant_accuracy.append(correct_count / total_nb_responses if total_nb_responses > 0 else 0)
        accuracy.append(participant_accuracy)
    return np.array(accuracy)

def compute_reverse_accuracy(data,max_positions=12, structure=False):
    """
    Compute the accuracy of the LAST `n` items in a sequence for each participant.

    Parameters:
    - data: DataFrame containing the experimental data.
    - max_positions (int): Maximum number of positions to evaluate in the sequence (default is 12).
    - structure (bool): If True, use 'response_structure' column; if False, use 'sequences_response' column.

    Returns:
    - np.array: An array of accuracies for each participant across the LAST `n` items.
    ex: [0.7, 0.6, 0.5,...,0] means [70% accurate on last item, 60% accurate on last AND 2nd to last items, 50% accurate on last THREE items etc.]
    """
    accuracy = []
    for participant in data['participant_ID'].unique():
        subset_participant = data[data['participant_ID'] == participant]
        total_nb_responses = len(subset_participant)
        participant_accuracy = []
        for i in range(1, max_positions + 1):
            correct_count = 0
            valid_responses = 0
            for index, row in subset_participant.iterrows():
                response = row['response_structure'] if structure else row['sequences_response']
                origin=row['sequences_structure'] if structure else row['seq']
                if len(origin) >= i and len(response) >= i:
                    valid_responses += 1
                    if origin[-i:] == response[-i:]:
                        correct_count += 1
            participant_accuracy.append(correct_count / total_nb_responses if total_nb_responses > 0 else 0)
        accuracy.append(participant_accuracy)
    return np.array(accuracy)

def compute_position_accuracy(data,max_positions=12, structure=False):
    """
    Compute the accuracy of the n-th item in a sequence for each participant.

    Parameters:
    - data: DataFrame containing the experimental data.
    - max_positions (int): Maximum number of positions to evaluate in the sequence (default is 12).
    - structure (bool): If True, use 'response_structure' column; if False, use 'sequences_response' column.

    Returns:
    - np.array: An array of accuracies for each participant for the n-th item.
    """
    accuracy = []
    for participant in data['participant_ID'].unique():
        subset_participant = data[data['participant_ID'] == participant]
        total_nb_responses = len(subset_participant)
        participant_accuracy = []
        for i in range(1, max_positions + 1):
            correct_count = 0
            valid_responses = 0
            for index, row in subset_participant.iterrows():
                response = row['response_structure'] if structure else row['sequences_response']
                origin=row['sequences_structure'] if structure else row['seq']
                if len(origin) >= i and len(response) >= i:
                    # print(f"Origin length: {len(origin)}, Response length: {len(response)}, Current index: {i}")
                    valid_responses += 1
                    if origin[i-1] == response[i-1]:
                        correct_count += 1
            participant_accuracy.append(correct_count / total_nb_responses if total_nb_responses > 0 else 0)
        accuracy.append(participant_accuracy)
    return np.array(accuracy)

def compute_reverse_position_accuracy(data,max_positions=12, structure=False):
    """
    Compute the accuracy of the n-th item in a sequence for each participant STARTING FROM THE END.

    Parameters:
    - data: DataFrame containing the experimental data.
    - max_positions (int): Maximum number of positions to evaluate in the sequence (default is 12).
    - structure (bool): If True, use 'response_structure' column; if False, use 'sequences_response' column.

    Returns:
    - np.array: An array of accuracies for each participant for the n-th item starting from the end.
    """
    accuracy = []
    for participant in data['participant_ID'].unique():
        subset_participant = data[data['participant_ID'] == participant]
        total_nb_responses = len(subset_participant)
        participant_accuracy = []
        for i in range(1, max_positions + 1):
            correct_count = 0
            valid_responses = 0
            for index, row in subset_participant.iterrows():
                response = row['response_structure'] if structure else row['sequences_response']
                origin=row['sequences_structure'] if structure else row['seq']
                if len(origin) >= i and len(response) >= i:
                    # print(f"Origin length: {len(origin)}, Response length: {len(response)}, Current index: {i}")
                    valid_responses += 1
                    if origin[-i] == response[-i]:
                        correct_count += 1
            participant_accuracy.append(correct_count / total_nb_responses if total_nb_responses > 0 else 0)
        accuracy.append(participant_accuracy)
    return np.array(accuracy)

def accuracies_general(data, max_positions=12, structure=False):
    """
    Calculate and display the mean accuracy and standard error of the mean (SEM) 
    for each position in a sequence for each participant, as well as for the first 
    and last `n` positions.

    This function provides a breakdown of accuracy statistics in three categories:
    - Accuracy by each position: Mean and SEM of accuracy at each position.
    - Accuracy for the first `n` positions: Mean and SEM of accuracy up to each of 
      the first `n` positions in a sequence.
    - Accuracy for the last `n` positions: Mean and SEM of accuracy for the final 
      `n` positions in a sequence.

    Parameters:
    - data: DataFrame containing the experimental data.
    - max_positions (int): The maximum number of positions to evaluate (default is 12).
    - structure (bool): If True, use 'response_structure' column; if False, use 
      'sequences_response' column.

    Returns:
    - None. Prints formatted results to the console.
    """
    
    # Compute and display accuracy by position
    print(" == Accuracies by Position: Mean per Participant ==\n")
    mean_accuracy_perParticipant_perPosition = compute_position_accuracy(data, max_positions=max_positions, structure=structure)
    mean_accuracy_perPosition = np.mean(mean_accuracy_perParticipant_perPosition, axis=0)
    sem_accuracy_perParticipant_perPosition = np.round(np.std(mean_accuracy_perParticipant_perPosition, axis=0) / np.sqrt(len(mean_accuracy_perParticipant_perPosition)), 3)
    general_accuracy_perPosition=np.mean(mean_accuracy_perParticipant_perPosition)
    sem_general_accuracy_perPosition=np.round(np.std(mean_accuracy_perParticipant_perPosition) / np.sqrt(len(mean_accuracy_perParticipant_perPosition)), 3)
    print("Mean Accuracy by Position:\n", mean_accuracy_perPosition)
    print("\nSEM by Position:\n", sem_accuracy_perParticipant_perPosition)
    print("\nGeneral mean accuracy all positions:\n", general_accuracy_perPosition)
    print("\nSEM:\n", sem_general_accuracy_perPosition)
    print("\n" + "-"*30 + "\n")
    
    # Compute and display accuracy for the first `n` positions
    print("== Accuracies on First `n` Positions: Mean per Participant ==\n")
    mean_accuracy_perParticipant_perPosition = compute_accuracy(data, max_positions=max_positions, structure=structure)
    mean_accuracy_perPosition = np.mean(mean_accuracy_perParticipant_perPosition, axis=0)
    sem_accuracy_perParticipant_perPosition = np.round(np.std(mean_accuracy_perParticipant_perPosition, axis=0) / np.sqrt(len(mean_accuracy_perParticipant_perPosition)), 3)
    print("Mean Accuracy on First `n` Positions:\n", mean_accuracy_perPosition)
    print("\nSEM on First `n` Positions:\n", sem_accuracy_perParticipant_perPosition)
    print("\n" + "-"*30 + "\n")
    
    # Compute and display positional accuracy for the `n` positions starting from last
    print("== Accuracies on `n-th` Positions starting from LAST: Mean per Participant ==\n")
    mean_accuracy_perParticipant_perPosition = compute_reverse_position_accuracy(data, max_positions=max_positions, structure=structure)
    mean_accuracy_perPosition = np.mean(mean_accuracy_perParticipant_perPosition, axis=0)
    sem_accuracy_perParticipant_perPosition = np.round(np.std(mean_accuracy_perParticipant_perPosition, axis=0) / np.sqrt(len(mean_accuracy_perParticipant_perPosition)), 3)
    print("Mean Accuracy `n-th` Positions starting from last:\n", mean_accuracy_perPosition)
    print("\nSEM:\n", sem_accuracy_perParticipant_perPosition)
    print("\n" + "-"*30 + "\n")
    
    # Compute and display accuracy for the last `n` positions
    print("== Accuracies on Last `n` Positions: Mean per Participant ==\n")
    mean_accuracy_perParticipant_perPosition = compute_reverse_accuracy(data, max_positions=max_positions, structure=structure)
    mean_accuracy_perPosition = np.mean(mean_accuracy_perParticipant_perPosition, axis=0)
    sem_accuracy_perParticipant_perPosition = np.round(np.std(mean_accuracy_perParticipant_perPosition, axis=0) / np.sqrt(len(mean_accuracy_perParticipant_perPosition)), 3)
    print("Mean Accuracy on Last `n` Positions:\n", mean_accuracy_perPosition)
    print("\nSEM on Last `n` Positions:\n", sem_accuracy_perParticipant_perPosition)
    print("\n" + "-"*30 + "\n")
    
    

    
    

def accuracy_histograms(data,save_path, max_positions=12, structure=False):
    """
    Generate and display individual histograms for the accuracy of each sequence.

    Parameters:
    - data: DataFrame containing the experimental data.
    - max_positions (int): Maximum number of positions to evaluate in the sequence (default is 12).
    - structure (bool): If True, use 'response_structure' column; if False, use 'sequences_response' column.

    Each histogram represents accuracy for the first `n` items of a sequence with error bars (SEM).
    """
    
    for index, seq_name in enumerate(seq_name_list):
        subset_data = data[data['seq_name'] == seq_name]
    
        accuracy = compute_accuracy(subset_data, max_positions, structure)
        mean_accuracy = np.mean(accuracy, axis=0)
        sem_accuracy = np.std(accuracy, axis=0) / np.sqrt(len(accuracy))
    
        # Plotting the histogram with error bars
        plt.figure(figsize=(10, 2))
        plt.bar(range(1, max_positions + 1), mean_accuracy, yerr=sem_accuracy, capsize=5, alpha=0.75)
        plt.title(f'Accuracy for Sequence: {seq_name}')
        plt.xlabel('Number of First Items')
        plt.ylabel('Accuracy',rotation=0, labelpad=y_label_pad)
        plt.ylim(0, 1)
        plt.xticks(range(1, max_positions + 1))
        # Using index to create a unique index_name for saving the figure
        if structure:
            plt.savefig(f'{save_path}/first_items_accuracy/first_items_accuracy_structure/{index}_{seq_name}_first_items_accuracy_structure.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
        else:
            plt.savefig(f'{save_path}/first_items_accuracy/first_items_accuracy_regular/{index}_{seq_name}_first_items_accuracy.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
        plt.show()


    
    

def to_percent(y, position):
    return f'{y:.0f}%'

def deconstruct_subprogram(data, path,index_name=13 ):
    # We will now classify the answers of participants in those different categories. The challenge is to have highly exclusive categories.
    #
    # No-Rule_1: Cannot find 2 occurences of "ABC".
    # No-Rule_2: Cannot find 2 occurences of "ABC" and full vocabulary is present
    # No_Rule_3: Cannot find 2 occurences of "ABC" and full vocabulary is present in the right order.
    # -----------------------------------------------------------------------------------------------
    # LoT-Rule_1: 2 validated strings that fits "ABC" 
    # LoT-Rule_2: 2 validated strings that fits "ABC-X" with X in {D,E,F}
    # LoT-Rule_3: Original == Response
    #
    print('No-Rule_1: Cannot find 2 occurences of "ABC".')
    print('No-Rule_2: Cannot find 2 occurences of "ABC" and full vocabulary is present.')
    print('No_Rule_3: Cannot find 2 occurences of "ABC" and full vocabulary is present in the right order.')
    print('----------------------------------------------')
    print('LoT-Rule_1: 2 validated strings that fits "ABC".')
    print('LoT-Rule_2: 2 validated strings that fits "ABC-X" with X in {D,E,F}.')
    print('LoT-Rule_3: Original == Response.')
    
    ####### Execution
    # Select the subset of subprogram-2 responses
    name=seq_name_list[index_name]
    subset_sub=data[data['seq_name']==name]

    # Initialize a counter array that represents [No-Rule_1, No-Rule_2, No-Rule_3, LoT-Rule_1, LoT-Rule_2, LoT-Rule_3]
    counter_rule=[0,0,0,0,0,0]

    # Search the dataset and categorize each response
    for index,row in subset_sub.iterrows():
        # Define variables
        response=row['comparable_temp']
        performance=row['performance']
        full_vocabulary=set(row['sequences_structure'])
        full_vocabulary_str=''.join(map(str,full_vocabulary))
        
        # If row['performance']=='success'. counter_rule[5]+=1
        if performance=='success':
            #counter_rule[3]+=1 # LoT-Rule_1
            #counter_rule[4]+=1 # LoT-Rule_2
            counter_rule[5]+=1 # LoT-Rule_3
            continue # Skip to the next iteration
            
        # Else search for two occurences of the chunk ABC. If not found counter_rule[0]+=1
        response_str=''.join(map(str,response))
        abc_count=response_str.count('012')
        if abc_count<2:
            counter_rule[0]+=1 # No-Rule_1
            
            # Search if full vocabulary is present. If found counter_rule[1]+=1
            if ''.join(map(str,set(response)))==full_vocabulary_str:
                counter_rule[1]+=1 # No-Rule_2

            # Search if full vocabulary appears in the right order. If it does counter_rule[2]+=1
                if ''.join(map(str,pd.unique(response)))==full_vocabulary_str:
                    counter_rule[2]+=1 # No-Rule_3

        # Else (two occurences of the chunk ABC are found). counter_rule[3]+=1
        else:
            counter_rule[3]+=1 # LoT-Rule_1

            # Search if there are two strings that fits "ABC-X" with X in {D,E,F}. If found: counter_rule[4]+=1
            counter_chunk=0
            
            for x in ['3','4','5']:
                if response_str.count(f"012{x}") > 0:
                    counter_chunk+=1
                    
            if counter_chunk >1:
                counter_rule[4]+=1
    
    # We need to remove all the occurences that are counted both times in each categories
    new_counter_rule=[0,0,0,0,0,0]
    new_counter_rule[0]=100*(counter_rule[0]-counter_rule[1])/len(subset_sub)
    new_counter_rule[1]=100*(counter_rule[1]-counter_rule[2])/len(subset_sub)
    new_counter_rule[2]=100*(counter_rule[2])/len(subset_sub)
    new_counter_rule[3]=100*(counter_rule[3]-counter_rule[4])/len(subset_sub)
    new_counter_rule[4]=100*(counter_rule[4])/len(subset_sub)
    new_counter_rule[5]=100*(counter_rule[5])/len(subset_sub)
    
    
        
    # Plot the distribution of responses in defined categories as a histogram
    categories = ['No-Rule_1', 'No-Rule_2', 'No-Rule_3', 'LoT-Rule_1', 'LoT-Rule_2', 'LoT-Rule_3']
    plt.figure(figsize=(10, 6))
    plt.bar(categories, new_counter_rule, color=['skyblue'] * 3 + ['Maroon'] * 3)
    plt.xlabel('Categories')
    plt.ylabel('Percentage of All Responses',rotation=0, labelpad=y_label_pad)
    plt.title(f'Distribution of Responses in Defined Categories for {name}')
    # Apply the percentage format to y-tick labels
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.savefig(f'{path}/learning_level_analysis/{index_name}_{name}_deconstruct_learning.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
    plt.show()
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
        
# ---------------------------------------
# ***** Simple Regression + T-test ******
# --------------------------------------- 
def simple_linear_regression(data):

    
    # Defining variables
    X=data[['LoT Complexity']]
    y=data[['distance_dl']]

    # Splitting the Dataset into Training and Testing set
    #X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

    # Create the model
    model=LinearRegression()

    # Train the model
    model.fit(X,y)

    # Return key components of the model
    return model.coef_[0][0], model.intercept_[0]

def aggregate_participants_OLS(data):
    # -- Todo
    # 1. I want to have one OLS per participant
    # 2. Extract Betas from the individual OLS
    # 3. T-Test on extracted betas
    
    # -- List Participants' IDs
    id=[data["participant_ID"][0]]
    for i in range(len(data)-1):
        if data.iloc[i]["participant_ID"] not in id:
            id.append(data.iloc[i]["participant_ID"])

    # -- Get Betas for every participant
    all_betas=[]
    for participant in id:
        subset_data=data[data['participant_ID']==participant]
        beta, intercept=simple_linear_regression(subset_data)
        all_betas.append(beta)


    return all_betas
    
def t_test_on_OLS(betas,display_text=True):
    if display_text:
        print("""Conducting a t-test on aggregate betas. Betas come from a Ordinary Least Square regression of the dependent variable (LoT Complexity)
                on the independent variable (Damerau-Levenshtein distance). We then ran a t-test to observe if our betas distribution is 
                significantly different from a normal distribution.\n""")
    # Conduct t-test on beta coefficients
    t_stat, p_value = stats.ttest_ind(betas,0)
    
    if display_text:
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
        print("Mean betas:", np.mean(betas))
    return t_stat, p_value

def extract_pearsonR(data):
    # Extract the relevant columns
    dl_distance = data['distance_dl']
    lot_complexity = data['LoT Complexity']
    
    # Calculate Pearson's r
    r, p_value = stats.pearsonr(dl_distance, lot_complexity)

    print("We computed Pearson R over dl_distance and Lot_complexity.")
    print(f"Pearson's r: {r}")
    print(f"P-value: {p_value}")

def plot_comparison_AIC_models(path, aic_arr, title="Δ(AIC) of different complexity models", selected_models=None):
    """
    Plots a horizontal bar chart comparing AIC values of different models, 
    ordered from best (lowest AIC) at the top to worst (highest AIC) at the bottom.

    Args:
        path (str): The directory path to save the plot.
        aic_arr (dict): A dictionary where keys are model names and values are AIC scores.
    """
    # Filter the AIC dictionary based on selected models (if provided)
    if selected_models is not None:
        aic_arr = {k: v for k, v in aic_arr.items() if k in selected_models}
        
    # Sort models by AIC value in ascending order
    sorted_items = sorted(aic_arr.items(), key=lambda x: x[1])  # Sort by AIC value
    model_names, aic_values = zip(*sorted_items)  # Unzip into two lists

    # Calculate delta AIC (AIC relative to the minimum AIC)
    min_aic = np.min(aic_values)
    delta_aic_values = [value - min_aic for value in aic_values]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(model_names)), delta_aic_values, color='grey')

    # Set y-axis ticks and labels (already ordered from best to worst)
    ax.invert_yaxis() 
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    # Customize plot appearance
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.title(title, fontsize=18, pad=15)

    # Save the plot
    plt.savefig(f'{path}/models/comparison_explanation/AIC_complexity_models.jpg', bbox_inches='tight', dpi=_adaptive_dpi())
    plt.show()  # Close figure to prevent memory issues.


def plot_distribution_dl_per_seq(data, path):
    import os
    os.makedirs(path, exist_ok=True)

    plt.rcParams["figure.facecolor"] = "white"

    all_bins = np.arange(0, 13)
    global_max = max(
        data[data["seq_name"] == seq]["distance_dl"].dropna()
            .value_counts().reindex(all_bins, fill_value=0).max()
        for seq in seq_name_list
    )

    for i, seq in enumerate(seq_name_list):
        seq_data = data[data["seq_name"] == seq]["distance_dl"].dropna()
        value_counts = seq_data.value_counts().reindex(all_bins, fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        color = plot_colors[i % len(plot_colors)]

        ax.bar(all_bins, value_counts.values, color=color,
               edgecolor='white', linewidth=bar_frame_width, width=0.8)
        ax.set_xticks(all_bins)
        ax.set_ylim(0, global_max)

        ax.set_xlabel("Damerau-Levenshtein Distance", fontsize=title_size, labelpad=padding_size)
        ax.set_ylabel("Count", fontsize=title_size, labelpad=padding_size)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.title(f"DL Distance Distribution — {seq}", size=title_size, pad=padding_size)

        safe_name = seq.replace(" ", "_").replace("/", "_")
        plt.savefig(f'{path}/{i}_{safe_name}.png', bbox_inches='tight', dpi=_adaptive_dpi())
        plt.close(fig)

    print(f"Saved {len(seq_name_list)} distribution plots to: {path}")


'''
# ---------------------------------------
# ********* Versions changelogs *********
# ---------------------------------------
Current: Version 2.8

*** 18.03.2025: Version 2.7
- Modified plot_mean_dl to harmonize scales (added one parameter x_interval)
- Modified plot_common_interclick to be able to set the colors of labels (added one parameter: colors) + Added a vlines bool parameter to display vlines or not + plot_title param for filename
- Completely Reworked plot_common_interclick_only_full_correct so that it matches plot_common_interclick

*** 05.08.2024: Version 2.7
- changing the error_rate_order_first_tokens() function to include an ORDER parameter that allows to compare sets rather than absolute reproduction of first chunk.

*** 18.07.2024: Version 2.5
- Changing the plot_regression function so that the points are colored.
- Added analyze_token_error() function

*** 17.07.2024: Version 2.4
- We go back to previous version on plot_mean_error_rates() functions and similar functions to use the histograms.
- Adding a model for simple regression and t-test on individual dl_distance as a function of LoT Complexity
- Adding Pearson's R

*** 17.07.2024: Version 2.3
- added: chunking_base_interclick() and chunking_base_interclick_target()

*** 14.07.2024: Version 2.2
- Adapting plot_mean_error_rates, plot_median_dl, plot_mean_dl to obtain a more elegant visual.



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