from modules.params import *
from modules.functions import *
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from scipy.stats import friedmanchisquare
from scipy.stats import shapiro
from scipy.stats import kstest, norm

##############################################################################################################################

def eliminate_outliers(lst):
        if len(lst) < 2:  # No outliers to remove if list is too small
            return lst
        lst_series = pd.Series(lst)
        Q1 = lst_series.quantile(0.25)
        Q3 = lst_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lst_series[(lst_series >= lower_bound) & (lst_series <= upper_bound)].tolist()

def eliminate_outliers_from_list(series):
    """
    Eliminate outliers from a column of lists using the IQR method.
    
    Parameters:
    -----------
    series : pandas.Series
        Series where each entry is a list of numerical values.
    
    Returns:
    --------
    series_no_outliers : pandas.Series
        Series where outliers have been removed from each list.
    """

    return series.apply(eliminate_outliers)



def test_normality(data, dl=True, distance_column='distance_dl', participant_column='participant_ID', condition_column='seq_name', kolmogorov=False):
    """
    Test if the DL-distance is normally distributed for each condition using the Kolmogorov-Smirnov test 
    after eliminating outliers and averaging the values per condition per participant.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data to be tested.
    dl: bool, optional
        if True test normality of mean DL per participant per condition. If False, tests the same but for error rates. Default is 'True'
    distance_column : str, optional
        Column name for the DL-distance values. Default is 'distance_dl'.
    participant_column : str, optional
        Column name for participant IDs. Default is 'participant_ID'.
    condition_column : str, optional
        Column name for the conditions. Default is 'seq_name'.
    kolmogorov: bool, optional
        If True, use Kolmogorov to assess normality. Else, uses Shapiro-Wilk. Default to False.

    Returns:
    --------
    results : dict
        A dictionary with condition names as keys and the Shapiro-Wilk test results (statistic
        and p-value) as values.

    How to Interpret and Report the Results:
    ----------------------------------------
    The Shapiro-Wilk test checks the null hypothesis that the data was drawn from a normal distribution.

    - **p-value > 0.05**: Fail to reject the null hypothesis. The data does not significantly deviate from normality.
    - **p-value ≤ 0.05**: Reject the null hypothesis. The data significantly deviates from normality.

    When reporting the results, include:
    - The test statistic.
    - The p-value.
    - Interpretation of the results in the context of the study.

    Example of results reporting:
    "The Kolmogorov-Smirnov test for normality was conducted on the DL-distance values per condition per participant 
    after eliminating outliers. For 'Repetition-2', the test statistic was 0.975 and the p-value was 0.25, 
    indicating no significant deviation from normality (p > 0.05). For ConditionB, the test statistic was 0.950 
    and the p-value was 0.03, indicating a significant deviation from normality (p ≤ 0.05)."
    """
    

    # Create a dictionary to store Shapiro-Wilk test results
    results = {}

    if kolmogorov:
        if dl:
            print('DL-DISTANCE: Result of Kolomogorov-Smirnov test for Normality')
        else:
            print('ERROR RATE: Result of Kolomogorov-Smirnov test for Normality')
    else:
        if dl:
            print('DL-DISTANCE: Result of Shapiro-Wilk test for Normality')
        else:
            print('ERROR RATE: Result of Shapiro-Wilk test for Normality')

    
    # Iterate over each condition
    for index_name in range(len(seq_name_list)):
        # Define sequence name for this iteration
        name=seq_name_list[index_name]

        # Get the data for current sequence
        subset = data[data[condition_column] == name]
        
        # Holder for all the means DL
        participant_means=[]
    
        # -- For each participant
        for IDs in subset[participant_column].unique():
            # Define the subset dataframe for the given sequence and given participant
            subset_participant=subset[subset[participant_column]==IDs]
            
            # -- number of trials
            nb_trials=len(subset_participant)
        
            # -- Test if there is at least one trial for this sequence (participants who did exp1 don't have trials on sequences of exp2)
            if nb_trials!=0:
                
                if dl:
                    # -- Mean DL for the given sequence for a particular participant
                    mean_dl=np.mean(subset_participant[distance_column].to_numpy())
                    
                    # -- Append the mean to the bigger array
                    participant_means.append(mean_dl)
                else:
                    # Reset number of success
                    nb_success=0
                    
                    # -- Total success number
                    nb_success=len(subset_participant[subset_participant['performance']=='success'])
                    
                    # -- Mean Error rate for the given sequence for a particular participant
                    participant_means.append(100*(1-nb_success/nb_trials))

        if kolmogorov:
            # Eliminate outliers
            cleaned_participant_means=np.array(eliminate_outliers(pd.Series(participant_means)))
            
            # Perform Kolmogorov-Smirnov
            stat, p_value = kstest(cleaned_participant_means, 'norm')
        else:
            # Use Shapiro-Wilk
            stat, p_value = shapiro(participant_means)

        
            
        #stat, p_value = shapiro(participant_means)

        
        # Print the results
        print(f'\n[{name}]: statistic: {stat}, p-value: {p_value}')
        if kolmogorov:
            print('Sample size: ',len(cleaned_participant_means)) 
        else:
            print('Sample size: ',len(participant_means))
        
        # print means if it's error rate
        if dl:
            pass
        else:
            print('mean error rate (with outliers)= ',np.mean(participant_means))
            print('mean error rate (NO outliers)= ',np.mean(cleaned_participant_means))
        




#######################################################################################################################################
def prepare_data_for_test(data, condition1, condition2, dl=True, distance_column='distance_dl', participant_column='participant_ID', condition_column='seq_name'):
    subset1 = data[data[condition_column] == condition1]
    subset2 = data[data[condition_column] == condition2]
    
    participant_means1 = []
    participant_means2 = []
    
    for participant in data[participant_column].unique():
        participant_data1 = subset1[subset1[participant_column] == participant]
        participant_data2 = subset2[subset2[participant_column] == participant]
        
        if not participant_data1.empty and not participant_data2.empty:
            if dl:
                mean1 = np.mean(participant_data1[distance_column])
                mean2 = np.mean(participant_data2[distance_column])
            else:
                success1 = len(participant_data1[participant_data1['performance'] == 'success'])
                success2 = len(participant_data2[participant_data2['performance'] == 'success'])
                mean1 = 100 * (1 - success1 / len(participant_data1))
                mean2 = 100 * (1 - success2 / len(participant_data2))
                
            participant_means1.append(mean1)
            participant_means2.append(mean2)
    
    return np.array(participant_means1), np.array(participant_means2)


def construct_comparison_pairs():
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
    
    # Define which sequences will be compared
    condition_pairs=[]
    holder_condition=[]
    
    for index_name in range(len(seq_name_list)):
    
        # Construct the condition_pairs from the comparison_index array
        if comparison_index[index_name]==0:
            holder_condition.append(seq_name_list[index_name])
    
        elif comparison_index[index_name]==2:
            holder_condition.append(seq_name_list[index_name-2])
            holder_condition.append(seq_name_list[index_name])
            condition_pairs.append(holder_condition)
            holder_condition=[]
        
        else:
            holder_condition.append(seq_name_list[index_name])
            condition_pairs.append(holder_condition)
            holder_condition=[]

    # -- We want to add the following comparisons
    #Index-i / CRep-2
    condition_pairs.append([seq_name_list[15],seq_name_list[1]])
    #CIndex-i /CRep-2
    condition_pairs.append([seq_name_list[16],seq_name_list[1]])
    #Play-4/ CRep-4
    condition_pairs.append([seq_name_list[9],seq_name_list[5]])
    #CPlay-4 /CRep-4
    condition_pairs.append([seq_name_list[10],seq_name_list[5]])


    return condition_pairs
            
def compute_friedman_test(data, dl=True, distance_column='distance_dl', participant_column='participant_ID', condition_column='seq_name'):
    """
    Compute the Friedman test for multiple pairs of conditions.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data to be tested.
    condition_pairs : list of tuples
        List of tuples, where each tuple contains two conditions to compare.
    dl : bool, optional
        If True, prepare data for DL-distance. If False, prepare data for error rates. Default is True.
    distance_column : str, optional
        Column name for the DL-distance values. Default is 'distance_dl'.
    participant_column : str, optional
        Column name for participant IDs. Default is 'participant_ID'.
    condition_column : str, optional
        Column name for the conditions. Default is 'seq_name'.

    Returns:
    --------
    stat : float
        Friedman test statistic.
    p_value : float
        P-value of the test.
    """
    


    condition_pairs=construct_comparison_pairs()
    if dl:
        print('----> Friedman Test on mean DL-Distances')
    else:
        print('-----> Friedman Test on mean Error Rates')
    
    for i in range(len(condition_pairs)):
        all_data = []
        
        condition1=condition_pairs[i][0]
        condition2=condition_pairs[i][1]
        
        group1, group2 = prepare_data_for_test(data, condition1, condition2, dl, distance_column, participant_column, condition_column)
        all_data.append(group1)
        all_data.append(group2)
        
        stat, p_value = friedmanchisquare(all_data[0],all_data[1])
        print(f'\nCompared {condition1} // {condition2}')
        print(f'Friedman stat: {stat}, p_value : {p_value}')
    

def compute_paired_ttest(data, dl=True, distance_column='distance_dl', participant_column='participant_ID', condition_column='seq_name'):
    """
    Compute the paired t-test for two conditions.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data to be tested.
    condition1 : str
        Name of the first condition to compare.
    condition2 : str
        Name of the second condition to compare.
    dl : bool, optional
        If True, prepare data for DL-distance. If False, prepare data for error rates. Default is True.
    distance_column : str, optional
        Column name for the DL-distance values. Default is 'distance_dl'.
    participant_column : str, optional
        Column name for participant IDs. Default is 'participant_ID'.
    condition_column : str, optional
        Column name for the conditions. Default is 'seq_name'.

    Returns:
    --------
    stat : float
        Paired t-test statistic.
    p_value : float
        P-value of the test.
    """

    condition_pairs=construct_comparison_pairs()
    
    if dl:
        print('----> Paired T-Test on mean DL-Distances')
    else:
        print('-----> Paired T-Test on mean Error Rates')
    
    for i in range(len(condition_pairs)):
        all_data = []
    
        condition1=condition_pairs[i][0]
        condition2=condition_pairs[i][1]
        
        group1, group2 = prepare_data_for_test(data, condition1, condition2, dl, distance_column, participant_column, condition_column)
        all_data.append(group1)
        all_data.append(group2)
        
        # Mean values
        mean1=np.mean(all_data[0])
        mean2=np.mean(all_data[1])
        
        # Standard deviation
        std_sample1=np.std(all_data[0])
        std_sample2=np.std(all_data[1])
        
        
        # SEM values
        sem_sample1=stats.sem(all_data[0], nan_policy='omit')
        sem_sample2=stats.sem(all_data[1], nan_policy='omit')
        
        stat, p_value = ttest_rel(all_data[0],all_data[1])
        print(f'\nCompared {condition1} // {condition2}')
        print(f'T-Test stat: {stat}, p_value : {p_value}')
        print(f'Mean. {condition1}: {round(mean1,3)}, {condition2}: {round(mean2,3)}')
        print(f'Standard Deviation. {condition1}: {round(std_sample1,3)}, {condition2}: {round(std_sample2,3)}')
        print(f'SEM. {condition1}: {round(sem_sample1,3)}, {condition2}: {round(sem_sample2,3)}')
    
def compute_wilcoxon_signed_rank_test_for_pair(data,seq1, seq2):
    """
    seq1 (str) : name of first sequence for comparison pair. Example : seq_name_list[0],
    seq2 (str) : name of second sequence for comparison pair. Example : seq_name_list[1]
    """
    all_data = []
    group1, group2 = prepare_data_for_test(data, seq1, seq2)
    all_data.append(group1)
    all_data.append(group2)

    # Mean values
    mean1=np.mean(all_data[0])
    mean2=np.mean(all_data[1])
    
    # Standard deviation
    std_sample1=np.std(all_data[0])
    std_sample2=np.std(all_data[1])
    
    
    # SEM values
    sem_sample1=stats.sem(all_data[0], nan_policy='omit')
    sem_sample2=stats.sem(all_data[1], nan_policy='omit')
    
    
    
    stat, p_value = wilcoxon(all_data[0],all_data[1])
    n = len([diff for diff in (np.array(all_data[0]) - np.array(all_data[1])) if diff != 0])  # exclude zero diffs
    
    # Compute the standardized Z value from the Wilcoxon statistic
    mean_W = n * (n + 1) / 4
    std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (stat - mean_W) / std_W
    r = abs(z) / np.sqrt(n)

    print(f'\nCompared {seq1} // {seq2}')
    print(f'Wilcoxon stat: {stat}, p_value : {p_value}')
    print(f'Effect size r = {round(r, 3)}')
    print(f'Mean. {seq1}: {round(mean1,3)}, {seq2}: {round(mean2,3)}')
    print(f'Standard Deviation. {seq1}: {round(std_sample1,3)}, {seq2}: {round(std_sample2,3)}')
    print(f'SEM. {seq1}: {round(sem_sample1,3)}, {seq2}: {round(sem_sample2,3)}')

def compute_wilcoxon_signed_rank_test(data, dl=True, distance_column='distance_dl', participant_column='participant_ID', condition_column='seq_name'):
    """
    Compute the Wilcoxon signed-rank test for two conditions.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data to be tested.
    condition1 : str
        Name of the first condition to compare.
    condition2 : str
        Name of the second condition to compare.
    dl : bool, optional
        If True, prepare data for DL-distance. If False, prepare data for error rates. Default is True.
    distance_column : str, optional
        Column name for the DL-distance values. Default is 'distance_dl'.
    participant_column : str, optional
        Column name for participant IDs. Default is 'participant_ID'.
    condition_column : str, optional
        Column name for the conditions. Default is 'seq_name'.

    Returns:
    --------
    stat : float
        Wilcoxon signed-rank test statistic.
    p_value : float
        P-value of the test.
        
    HOW TO REPORT:
    --------------
    Purpose of the Test: Explain why the test was conducted.
    Descriptive Statistics: Provide descriptive statistics of the paired samples (e.g., means, medians, standard deviations).
    Test Results: Report the test statistic (W) and the p-value.
    Conclusion: Interpret the results in the context of the research question.
    
    Example of report:
    -----------------
    A Wilcoxon signed-rank test was conducted on paired conditions (structured and controls) to determine whether mean error-rates differences were significant. 
    The Wilcoxon signed-rank test serves the same purpose as a t-test but it is used to compare 2 non-parametric dependent samples. 

    The Wilcoxon signed-rank test was conducted to determine if there was a statistically significant difference in test scores before and after the new teaching method.
    The results indicated a statistically significant increase in test scores (W = 0.0, p = 0.002), 
    suggesting that the new teaching method had a positive effect on student performance.
    
    """
    if dl:
        print('----> Wilcoxon Signed Rank test on mean DL-Distances')
    else:
        print('-----> Wilcoxon Signed Rank test on mean Error Rates')
        all_mean_errors, all_sem_errorRates,all_error_rates_participants=mean_error_rate_SEM(data)
        
    
        
    condition_pairs=construct_comparison_pairs()
    for i in range(len(condition_pairs)):

        all_data = []

        condition1=condition_pairs[i][0]
        condition2=condition_pairs[i][1]
        
        if dl:
            group1, group2 = prepare_data_for_test(data, condition1, condition2, dl, distance_column, participant_column, condition_column)
        else:
            index1=seq_name_list.index(condition_pairs[i][0])
            index2=seq_name_list.index(condition_pairs[i][1])
            group1= all_error_rates_participants[index1]
            group2= all_error_rates_participants[index2]
            
        all_data.append(group1)
        all_data.append(group2)

        # Mean values
        mean1=np.mean(all_data[0])
        mean2=np.mean(all_data[1])
        
        # Standard deviation
        std_sample1=np.std(all_data[0])
        std_sample2=np.std(all_data[1])
        
        
        # SEM values
        sem_sample1=stats.sem(all_data[0], nan_policy='omit')
        sem_sample2=stats.sem(all_data[1], nan_policy='omit')
        
        
        
        stat, p_value = wilcoxon(all_data[0],all_data[1])
        n = len([diff for diff in (np.array(all_data[0]) - np.array(all_data[1])) if diff != 0])  # exclude zero diffs
        
        # Compute the standardized Z value from the Wilcoxon statistic
        mean_W = n * (n + 1) / 4
        std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (stat - mean_W) / std_W
        r = abs(z) / np.sqrt(n)

        print(f'\nCompared {condition1} // {condition2}')
        print(f'Wilcoxon stat: {stat}, p_value : {p_value}')
        print(f'Effect size r = {round(r, 3)}')
        print(f'Mean. {condition1}: {round(mean1,3)}, {condition2}: {round(mean2,3)}')
        print(f'Standard Deviation. {condition1}: {round(std_sample1,3)}, {condition2}: {round(std_sample2,3)}')
        print(f'SEM. {condition1}: {round(sem_sample1,3)}, {condition2}: {round(sem_sample2,3)}')

# --------------------------------------------------------------------------------------------------------------------------------
def compute_control_vs_others_wilcoxon(data, list_group_structured,list_group_control, dl=True, distance_column='distance_dl', participant_column='participant_ID', condition_column='seq_name'):
    """
    Compute the Wilcoxon signed-rank test comparing the aggregated mean of sequences with "control" in their names to the aggregated mean of other sequences.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data to be tested.
    list_group_structured : list of str
        List of sequence names in the group of structured sequences.
    list_group_control : list of str
        List of sequence names in the group of control sequences.
    dl : bool, optional
        If True, prepare data for DL-distance. If False, prepare data for error rates. Default is True.
    distance_column : str, optional
        Column name for the DL-distance values. Default is 'distance_dl'.
    participant_column : str, optional
        Column name for participant IDs. Default is 'participant_ID'.
    condition_column : str, optional
        Column name for the conditions. Default is 'seq_name'.

    Returns:
    --------
    result : dict
        Dictionary containing test results for the aggregated comparison.
    """
    if dl:
        print('----> Wilcoxon Signed Rank test on aggregated mean DL-Distances')
    else:
        print('-----> Wilcoxon Signed Rank test on aggregated mean Error Rates')
        all_mean_errors, all_sem_errorRates, all_error_rates_participants = mean_error_rate_SEM(data)

    # Separate "control" sequences from others
    control_sequences = list_group_control
    other_sequences = list_group_structured

    # Prepare data for control sequences
    if dl:
        print('Comparing Damerau-Levenshtein distances ...\n\n')
        all_control_data=[]
        all_other_data=[]
        # ensuring data are of the same size
        for index_name in range(len(control_sequences)):
            name1=control_sequences[index_name]
            name2=other_sequences[index_name]
            temp_data_control=data[data[condition_column]==name1][distance_column]
            temp_data_other=data[data[condition_column]==name2][distance_column]
            if len(temp_data_control)!=len(temp_data_other):
                min_length=min(len(temp_data_control),len(temp_data_other))
                print(f'Sample sizes are different for [{name1}] and [{name2}]. Reducing highest to : ',min_length)
                temp_data_control=temp_data_control[:min_length]
                temp_data_other=temp_data_other[:min_length]
            all_control_data.append(temp_data_control)
            all_other_data.append(temp_data_other)
        # Flatten the arrays
        control_data=np.array([item for sublist in all_control_data for item in sublist])
        other_data=np.array([item for sublist in all_other_data for item in sublist])
        
        #control_data = data[data[condition_column].isin(control_sequences)][distance_column]
        #other_data = data[data[condition_column].isin(other_sequences)][distance_column]
    else:
        print('Comparing Error Rates ...\n\n')
        control_data = np.array([mean_error_rate_SEM(data[data[condition_column] == seq]) for seq in control_sequences])
        other_data = np.array([mean_error_rate_SEM(data[data[condition_column] == seq]) for seq in other_sequences])

    
    

    # Aggregate means
    aggregated_control_mean = np.mean(control_data)
    aggregated_other_mean = np.mean(other_data)

    # Calculate SEM
    sem_control = stats.sem(control_data)
    sem_other = stats.sem(other_data)

    # Absolute difference
    absolute_difference = abs(aggregated_control_mean - aggregated_other_mean)

    # Wilcoxon test
    stat, p_value = wilcoxon(control_data, other_data)

    # Compute effect size r = Z / sqrt(N), where N = number of non-zero diffs
    diffs = control_data - other_data
    non_zero_diffs = diffs[diffs != 0]
    n = len(non_zero_diffs)
    if n > 0:
        # Expected mean and std under H0
        mean_W = n * (n + 1) / 4
        std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (stat - mean_W) / std_W
        r = abs(z) / np.sqrt(n)
    else:
        z = np.nan
        r = np.nan

    result = {
        'aggregated_control_mean': round(aggregated_control_mean, 3),
        'aggregated_other_mean': round(aggregated_other_mean, 3),
        'sem_control': round(sem_control, 3),
        'sem_other': round(sem_other, 3),
        'absolute_difference': round(absolute_difference, 3),
        'stat': stat,
        'p_value': p_value,
        'effect_size_r': round(r, 3)
    }

    print(f'\nAggregated Comparison: Control vs. Other')
    print(f'Wilcoxon stat: {stat}, p_value: {p_value}')
    print(f'Effect size r = {round(r, 3)}')
    print(f'Aggregated Mean Control: {round(aggregated_control_mean, 3)}, SEM: {round(sem_control, 3)}')
    print(f'Aggregated Mean Other: {round(aggregated_other_mean, 3)}, SEM: {round(sem_other, 3)}')
    print(f'Absolute Difference: {round(absolute_difference, 3)}')




    
    
# --------------------------------------------------------------------------------------------------------------------------------
def print_base_stats(data):
    """
    This function will give us information on Error Rates and Damerau-Levenshtein distance.
    For both, it will output:
        - Overall Mean
        - Mean for structured sequences
        - Mean for control sequences
        - Mean for experiment 1 sequences
        - Mean for experiment 2 sequences

    For Damerau-Levenshtein distance, it will additionally output the DL-distance of each sequence to its easiest relative arrangement.
    Example: the easiest arrangement of Rep-2 is '000000 111111'. (Are participants producing closer approximations of sequences than this
    rough representation?) Plus, the mean of all these values (This gives us the performance of an agent that would only be able to recognize
    which token was used and count how many occurrences of this item were in the sequence. The agent groups all occurrences of a token together).
    """
    
    def generate_easiest(seq_structure):
        """
        Generate the easiest sequence structure based on given sequence structure.
        
        Parameters:
        seq_structure (str): A string representing the sequence structure.

        Returns:
        list: A list of integers representing the easiest sequence structure.
        """
        return sorted(seq_structure)

    seq_name_list = data['seq_name'].unique().tolist()
    all_structures = [data[data['seq_name'] == name]['sequences_structure'].iloc[0] for name in seq_name_list]
    easiest_structures = list(map(generate_easiest, all_structures))

    # Generate distance to easiest array
    dl_to_easiest = [dl_distance(regular, easiest) for regular, easiest in zip(all_structures, easiest_structures)]
    dict_dl_to_easiest = dict(zip(seq_name_list, dl_to_easiest))

    # Subset data
    subset_structured = data[data['seq_name'].apply(lambda x: 'control' not in x)]
    subset_control = data[data['seq_name'].apply(lambda x: 'control' in x)]
    subset_exp1 = data[data['which_exp'] == 'base']
    subset_exp2 = data[data['which_exp'] == 'extension']
    subset_structured_exp1 = subset_exp1[subset_exp1['seq_name'].apply(lambda x: 'control' not in x)]
    subset_structured_exp2 = subset_exp2[subset_exp2['seq_name'].apply(lambda x: 'control' not in x)]
    subset_control_exp1 = subset_exp1[subset_exp1['seq_name'].apply(lambda x: 'control' in x)]
    subset_control_exp2 = subset_exp2[subset_exp2['seq_name'].apply(lambda x: 'control' in x)]

    # Calculate means
    overall_mean = np.mean(data['distance_dl'])
    structured_mean = np.mean(subset_structured['distance_dl'])
    control_mean = np.mean(subset_control['distance_dl'])
    exp1_mean = np.mean(subset_exp1['distance_dl'])
    exp2_mean = np.mean(subset_exp2['distance_dl'])
    structured_mean_exp1 = np.mean(subset_structured_exp1['distance_dl'])
    control_mean_exp1 = np.mean(subset_control_exp1['distance_dl'])
    structured_mean_exp2 = np.mean(subset_structured_exp2['distance_dl'])
    control_mean_exp2 = np.mean(subset_control_exp2['distance_dl'])

    # Calculate standard errors of the mean
    overall_sem = np.std(data['distance_dl'], ddof=1) / np.sqrt(len(data['distance_dl']))
    structured_sem = np.std(subset_structured['distance_dl'], ddof=1) / np.sqrt(len(subset_structured['distance_dl']))
    control_sem = np.std(subset_control['distance_dl'], ddof=1) / np.sqrt(len(subset_control['distance_dl']))
    exp1_sem = np.std(subset_exp1['distance_dl'], ddof=1) / np.sqrt(len(subset_exp1['distance_dl']))
    exp2_sem = np.std(subset_exp2['distance_dl'], ddof=1) / np.sqrt(len(subset_exp2['distance_dl']))
    exp1_structured_sem = np.std(subset_structured_exp1['distance_dl'], ddof=1) / np.sqrt(len(subset_structured_exp1['distance_dl']))
    exp1_control_sem = np.std(subset_control_exp1['distance_dl'], ddof=1) / np.sqrt(len(subset_control_exp1['distance_dl']))
    exp2_structured_sem = np.std(subset_structured_exp2['distance_dl'], ddof=1) / np.sqrt(len(subset_structured_exp2['distance_dl']))
    exp2_control_sem = np.std(subset_control_exp2['distance_dl'], ddof=1) / np.sqrt(len(subset_control_exp2['distance_dl']))

    # Print results
    print('----------------------------')
    print('Damerau-Levenshtein distance')
    print('----------------------------\n')
    print(f"Overall Mean: {round(overall_mean, 3)}, SEM: {round(overall_sem, 3)}")
    print(f"Structured Sequences' Mean: {round(structured_mean, 3)}, SEM: {round(structured_sem, 3)}")
    print(f"Control Sequences' Mean: {round(control_mean, 3)}, SEM: {round(control_sem, 3)}")
    print(f"Experiment BASE Mean: {round(exp1_mean, 3)}, SEM: {round(exp1_sem, 3)}")
    print(f"Experiment EXTENSION Mean: {round(exp2_mean, 3)}, SEM: {round(exp2_sem, 3)}")
    print('\n** Experiment-1**')
    print(f"__Structured Sequences' mean: {round(structured_mean_exp1, 3)}, SEM: {round(exp1_structured_sem, 3)}")
    print(f"__Control Sequences' mean: {round(control_mean_exp1, 3)}, SEM: {round(exp1_control_sem, 3)}")
    print('\n** Experiment-2**')
    print(f"__Structured Sequences' mean: {round(structured_mean_exp2, 3)}, SEM: {round(exp2_structured_sem, 3)}")
    print(f"__Control Sequences' mean: {round(control_mean_exp2, 3)}, SEM: {round(exp2_control_sem, 3)}")

    print('\n\n----------------------------')
    print('MEAN Error Rates')
    print('----------------------------\n')
    
    # Error Rates
    all_mean_errors, all_sem_errorRates,all_error_rates = mean_error_rate_SEM(data)
    all_structured_error, all_structured_error_sem,all_error_rates = mean_error_rate_SEM(subset_structured)
    all_control_error, all_control_error_sem,all_error_rates = mean_error_rate_SEM(subset_control)
    all_exp1_error, all_exp1_error_sem,all_error_rates = mean_error_rate_SEM(subset_exp1)
    all_exp2_error, all_exp2_error_sem,all_error_rates = mean_error_rate_SEM(subset_exp2)
    all_structured_error_exp1, all_structured_error_exp1_sem,all_error_rates = mean_error_rate_SEM(subset_structured_exp1)
    all_control_error_exp1, all_control_error_exp1_sem,all_error_rates = mean_error_rate_SEM(subset_control_exp1)
    all_structured_error_exp2, all_structured_error_exp2_sem,all_error_rates = mean_error_rate_SEM(subset_structured_exp2)
    all_control_error_exp2, all_control_error_exp2_sem,all_error_rates = mean_error_rate_SEM(subset_control_exp2)

    overall_error = np.mean(all_mean_errors)
    structured_error = np.mean(all_structured_error)
    control_error = np.mean(all_control_error)
    exp1_error = np.mean(all_exp1_error)
    exp2_error = np.mean(all_exp2_error)
    structured_error_exp1 = np.mean(all_structured_error_exp1)
    control_error_exp1 = np.mean(all_control_error_exp1)
    structured_error_exp2 = np.mean(all_structured_error_exp2)
    control_error_exp2 = np.mean(all_control_error_exp2)

    overall_mean_error_sem = np.std(all_mean_errors) / np.sqrt(len(all_mean_errors))
    structured_error_sem = np.std(all_structured_error) / np.sqrt(len(all_structured_error))
    control_error_sem = np.std(all_control_error) / np.sqrt(len(all_control_error))
    exp1_error_sem = np.std(all_exp1_error) / np.sqrt(len(all_exp1_error))
    exp2_error_sem = np.std(all_exp2_error) / np.sqrt(len(all_exp2_error))
    structured_error_exp1_sem = np.std(all_structured_error_exp1) / np.sqrt(len(all_structured_error_exp1))
    control_error_exp1_sem = np.std(all_control_error_exp1) / np.sqrt(len(all_control_error_exp1))
    structured_error_exp2_sem = np.std(all_structured_error_exp2) / np.sqrt(len(all_structured_error_exp2))
    control_error_exp2_sem = np.std(all_control_error_exp2) / np.sqrt(len(all_control_error_exp2))

    print(f"Overall ERROR: {round(overall_error, 3)}, SEM : {round(overall_mean_error_sem, 3)}")
    print(f"Structured Sequences' ERROR: {round(structured_error, 3)}, SEM : {round(structured_error_sem, 3)}")
    print(f"Control Sequences' ERROR: {round(control_error, 3)}, SEM : {round(control_error_sem, 3)}")
    print(f"Experiment BASE ERROR: {round(exp1_error, 3)}, SEM : {round(exp1_error_sem, 3)}")
    print(f"Experiment EXTENSION ERROR: {round(exp2_error, 3)}, SEM : {round(exp2_error_sem, 3)}")
    print('\n** Experiment-1**')
    print(f"__Structured Sequences' ERROR: {round(structured_error_exp1, 3)}, SEM : {round(structured_error_exp1_sem, 3)}")
    print(f"__Control Sequences' ERROR: {round(control_error_exp1, 3)}, SEM : {round(control_error_exp1_sem, 3)}")
    print('\n** Experiment-2**')
    print(f"__Structured Sequences' ERROR: {round(structured_error_exp2, 3)}, SEM : {round(structured_error_exp2_sem, 3)}")
    print(f"__Control Sequences' ERROR: {round(control_error_exp2, 3)}, SEM : {round(control_error_exp2_sem, 3)}")

    print('\n\n----------------------------')
    print('DL to easiest')
    print('----------------------------\n')
    for key, val in dict_dl_to_easiest.items():
        print(f'{val}   {key}')
    print(f'===> Mean DL to easiest: {np.mean(list(dict_dl_to_easiest.values()))}')
    print("""
          Example: easiest arrangement of Rep-2 is '000000 111111'. (Are participants producing closer approximations of sequences than this 
    rough representation?) Plus, the mean of all these values (This gives us the performance of an agent that would only be able to recognize
    which token was used and count how many occurrences of this item was in the sequence. The agent groups all occurrences of a token together)
          """)
    
# ------------------------------------------------------------------------------------------

def print_base_stats_single_exp(data):
    """
    This function will give us information on Error Rates and Damerau-Levenshtein distance.
    For both, it will output:
        - Overall Mean
        - Mean for structured sequences
        - Mean for control sequences
        - Mean for experiment 1 sequences
        - Mean for experiment 2 sequences

    For Damerau-Levenshtein distance, it will additionally output the DL-distance of each sequence to its easiest relative arrangement.
    Example: the easiest arrangement of Rep-2 is '000000 111111'. (Are participants producing closer approximations of sequences than this
    rough representation?) Plus, the mean of all these values (This gives us the performance of an agent that would only be able to recognize
    which token was used and count how many occurrences of this item were in the sequence. The agent groups all occurrences of a token together).
    """
    
    def generate_easiest(seq_structure):
        """
        Generate the easiest sequence structure based on given sequence structure.
        
        Parameters:
        seq_structure (str): A string representing the sequence structure.

        Returns:
        list: A list of integers representing the easiest sequence structure.
        """
        return sorted(seq_structure)

    seq_name_list = data['seq_name'].unique().tolist()
    all_structures = [data[data['seq_name'] == name]['sequences_structure'].iloc[0] for name in seq_name_list]
    easiest_structures = list(map(generate_easiest, all_structures))

    # Generate distance to easiest array
    dl_to_easiest = [dl_distance(regular, easiest) for regular, easiest in zip(all_structures, easiest_structures)]
    dict_dl_to_easiest = dict(zip(seq_name_list, dl_to_easiest))

    # Subset data
    subset_structured = data[data['seq_name'].apply(lambda x: 'control' not in x)]
    subset_control = data[data['seq_name'].apply(lambda x: 'control' in x)]

    # Calculate means
    overall_mean = np.mean(data['distance_dl'])
    structured_mean = np.mean(subset_structured['distance_dl'])
    control_mean = np.mean(subset_control['distance_dl'])


    # Calculate standard errors of the mean
    overall_sem = np.std(data['distance_dl'], ddof=1) / np.sqrt(len(data['distance_dl']))
    structured_sem = np.std(subset_structured['distance_dl'], ddof=1) / np.sqrt(len(subset_structured['distance_dl']))
    control_sem = np.std(subset_control['distance_dl'], ddof=1) / np.sqrt(len(subset_control['distance_dl']))

    # Print results
    print('----------------------------')
    print('Damerau-Levenshtein distance')
    print('----------------------------\n')
    print(f"Overall Mean: {round(overall_mean, 3)}, SEM: {round(overall_sem, 3)}")
    print(f"Structured Sequences' Mean: {round(structured_mean, 3)}, SEM: {round(structured_sem, 3)}")
    print(f"Control Sequences' Mean: {round(control_mean, 3)}, SEM: {round(control_sem, 3)}")

    print('\n\n----------------------------')
    print('MEAN Error Rates')
    print('----------------------------\n')
    
    # Error Rates
    all_mean_errors, all_sem_errorRates,all_error_rates = mean_error_rate_SEM(data)
    all_structured_error, all_structured_error_sem,all_error_rates = mean_error_rate_SEM(subset_structured)
    all_control_error, all_control_error_sem,all_error_rates = mean_error_rate_SEM(subset_control)

    overall_error = np.mean(all_mean_errors)
    structured_error = np.mean(all_structured_error)
    control_error = np.mean(all_control_error)

    overall_mean_error_sem = np.std(all_mean_errors) / np.sqrt(len(all_mean_errors))
    structured_error_sem = np.std(all_structured_error) / np.sqrt(len(all_structured_error))
    control_error_sem = np.std(all_control_error) / np.sqrt(len(all_control_error))

    print(f"Overall ERROR: {round(overall_error, 3)}, SEM : {round(overall_mean_error_sem, 3)}")
    print(f"Structured Sequences' ERROR: {round(structured_error, 3)}, SEM : {round(structured_error_sem, 3)}")
    print(f"Control Sequences' ERROR: {round(control_error, 3)}, SEM : {round(control_error_sem, 3)}")

    print('\n\n----------------------------')
    print('DL to easiest')
    print('----------------------------\n')
    for key, val in dict_dl_to_easiest.items():
        print(f'{val}   {key}')
    print(f'===> Mean DL to easiest: {np.mean(list(dict_dl_to_easiest.values()))}')
    print("""
          Example: easiest arrangement of Rep-2 is '000000 111111'. (Are participants producing closer approximations of sequences than this 
    rough representation?) Plus, the mean of all these values (This gives us the performance of an agent that would only be able to recognize
    which token was used and count how many occurrences of this item was in the sequence. The agent groups all occurrences of a token together)
          """)


def mean_error_rate_SEM(data):
    """
    Calculate the mean error rate and standard error of the mean (SEM) for each sequence across participants.

    Parameters:
    data (DataFrame): The dataset containing participant IDs, sequence names, and performance.

    Returns:
    tuple: Two lists containing the mean error rates and SEM for each sequence respectively.
    """
    all_error_rates_seq = []
    mean_per_participant_error_rates = []
    sem_per_participant_error_rates = []

    sequences = [seq_name for seq_name in seq_name_list if seq_name in data['seq_name'].unique()]

    # For each sequence
    for name in sequences:
        error_rates_seq = []
        # For each participant
        for participant_id in data['participant_ID'].unique():
            # Number of trials for the current participant and sequence
            nb_trials = len(data[(data['participant_ID'] == participant_id) & (data['seq_name'] == name)])
    
            # Check if there is at least one trial for this sequence
            if nb_trials != 0:
                # Total number of errors for the sequence
                nb_error = len(data[(data['participant_ID'] == participant_id) & (data['seq_name'] == name) & (data['performance'] != 'success')])
            
                # Calculate the error rate and append to the list
                error_rates_seq.append(100 * nb_error / nb_trials)
                
        # Append all participants' error rates for the current sequence
        all_error_rates_seq.append(error_rates_seq)
    
    # Calculate mean and SEM for each sequence error rates array
    for error_rates in all_error_rates_seq:
        mean_error_rate = np.mean(error_rates)
        sem_error_rate = np.std(error_rates) / np.sqrt(len(error_rates))
        mean_per_participant_error_rates.append(mean_error_rate)
        sem_per_participant_error_rates.append(sem_error_rate)
    
    return mean_per_participant_error_rates, sem_per_participant_error_rates, all_error_rates_seq


def remove_outliers(arr):
    # Convert list to numpy array for easy manipulation
    arr = np.array(arr)
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1, Q3 = np.percentile(arr, [25, 75])
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    return arr[(arr >= lower_bound) & (arr <= upper_bound)].tolist()

def stat_model_OLS(data):
    X = data['LoT Complexity']  # Independent variable
    Y = data['distance_dl']   # Dependent variable
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(Y, X).fit()
    r_squared = model.rsquared
    f_statistic = model.fvalue
    p_value = model.f_pvalue
    print(f'R-squared: {r_squared:.4f}')
    print(f'F-statistic: {f_statistic:.4f}')
    print(f'P-value of the F-statistic: {p_value}')

def compute_wilcoxon_seq_length(data, participant_column='participant_ID', list_sequences=seq_name_list_exp1_only,condition_column='seq_name',experiment=1):
    # First we get the mean length + SEM for each sequence type
    print('Computing base stats on LENGTH of response sequences...\n\n')
    # Pairs of sequences to compare for the stat test
    if experiment==1:
        index_pairs=[[0,1],
                    [2,3],
                    [4,5],
                    [6,5],
                    [6,7],
                    [6,8],
                    [6,3]]
    else:
        index_pairs=[[0,1],
            [2,3],
            [4,5],
            [6,5],
            [6,7],
            [6,8],
            [9,10],
             [11,12],
             [13,14],
             [15,16],
             [17,18],
             [19,20],
             [21,22],
             [23,24]]

    # Get the mean sequence length per participant
    all_seq_mean_length=[] # Shape: (9, nb_participants)


    # Go over each sequence
    for index_name in range(len(list_sequences)):

        name=list_sequences[index_name]
        
        # -- Get subset for given sequence name
        data_subset=data[data[condition_column]==name]
        
        # -- Retrieve the mean length for each participant
        mean_length_participants=[]
        participant_IDs=[participant for participant in data_subset[participant_column].unique()]
        for individual_id in participant_IDs:
            mean_length_participants.append(np.mean([seq_length for seq_length in data_subset[data_subset[participant_column]==individual_id]['sequences_response'].apply(lambda x:len(x))]))

        # -- Add the computed means to the array taking in all means
        all_seq_mean_length.append(mean_length_participants)

    for pairs in index_pairs:
        index_seq1, index_seq2 = pairs
        
        # Names of compared sequences
        condition1=list_sequences[index_seq1]
        condition2=list_sequences[index_seq2]

        # All Means
        all_mean1=all_seq_mean_length[index_seq1]
        all_mean2=all_seq_mean_length[index_seq2]
        
        # Data should be of the same size
        #
        if len(all_mean1)!=len(all_mean2):
            min_length=min(len(all_mean1),len(all_mean2))
            all_mean1=all_mean1[:min_length]
            all_mean2=all_mean2[:min_length]


        # Mean values
        mean1=np.mean(all_mean1)
        mean2=np.mean(all_mean2)
        
        # Standard deviation
        std_sample1=np.std(all_mean1)
        std_sample2=np.std(all_mean2)
        
        
        # SEM values
        sem_sample1=stats.sem(all_mean1, nan_policy='omit')
        sem_sample2=stats.sem(all_mean2, nan_policy='omit')
        
        # Wilcoxon Test
        stat, p_value = wilcoxon(all_mean1,all_mean2)

        # Normality test
        cleaned_participant_means1=np.array(eliminate_outliers(pd.Series(all_mean1)))
        cleaned_participant_means2=np.array(eliminate_outliers(pd.Series(all_mean2)))
            
        # Perform Shapiro-Wilk
        k_stat1, k_p_value1 = shapiro(cleaned_participant_means1)
        k_stat2, k_p_value2 = shapiro(cleaned_participant_means2)

        

        # --- Printing the values
        print('------------------------------------------------')
        print(f' --- Compared {condition1} // {condition2} ---')
        print('------------------------------------------------')
        print(f'Wilcoxon stat: {stat}, p_value : {p_value}')
        print(f'\n----- {condition1} ------')
        print(f'Mean : {round(mean1,3)}, SEM : {round(sem_sample1,3)}, Standard Deviation : {round(std_sample1,3)}')
        print(f'Normality Test (Shapiro-Wilk). p-value : {k_p_value1}, test stat : {k_stat1}')
        print(f'\n----- {condition2} ------')
        print(f'Mean : {round(mean2,3)}, SEM : {round(sem_sample2,3)}, Standard Deviation : {round(std_sample2,3)}')
        print(f'Normality Test (Shapiro-Wilk). p-value : {k_p_value2}, test stat : {k_stat2}')
        print('\n\n')


'''
# ---------------------------------------
# ********* Versions changelogs *********
# ---------------------------------------
Current: Version 1.1

*** 01.08.2024: Version 1.1
- Changing the print_base_stat() function to correct inefficiencies.
- modified compute_wilcoxon_signed_rank_test() function to include the case for error rates


*** 01.08.2024: Version 1.0
- Created the module.



'''
