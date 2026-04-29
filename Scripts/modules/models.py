from modules.functions import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def create_participant_df(df,exclude=[]):
    """
    Create a DataFrame with one row per participant, aggregating mean distance_dl values for each sequence.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'participant_ID', 'seq_name', and 'distance_dl' columns.
        exclude (array): Labels of columns to exclude from the dataframe.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a participant and each column represents the mean distance_dl for a specific sequence.
    """
    
    
    # -- Gather participants' IDs
    all_ID=[i for i in df['participant_ID'].unique()]

    # -- Create a dataset with one row per participant. 
    participant_df = pd.DataFrame(all_ID, columns=['participant_id'])
        
    for name in seq_name_list:
        # -- Create a subset for considered sequence name. Take only participant_ID and distance_dl
        subset_data=df[df['seq_name']==name].copy()[['participant_ID','distance_dl']]
        
        # -- Take the mean DL value for each participant and append it in holder for future column.
        column_holder=[]
        for id in all_ID:
            dl_values=[i for i in subset_data[subset_data['participant_ID']==id]['distance_dl']]
            column_holder.append(np.mean(dl_values))
        
        # -- Add to participant_df one column per sequence. Each column is the mean dl_distance for a give sequence.
        if not name in exclude:
            participant_df[f'{name}']=column_holder
    return participant_df

def perform_PCA(df, pc_nb=2, heatmap=True, decide=False):
    """
    Perform PCA to reduce dimensionality and optionally decide on the number of components by visualizing the explained variance.

    Args:
        df (pd.DataFrame): Input DataFrame containing participant data with 'participant_id' and other feature columns.
        pc_nb (int): Number of principal components to use if decide is False. Default is 2.
        heatmap (bool): If True, plot a heatmap of the feature loadings on the principal components. Default is True.
        decide (bool): If True, visualize the explained variance to help decide the number of components. Default is False.

    Returns:
        np.ndarray or None: Transformed data with reduced dimensions if decide is False. None otherwise.
        
    Example usage:
        # Perform PCA with 3 principal components
        x_pca = perform_PCA(participant_df, pc_nb=3, decide=False)
        
        # Visualize explained variance to decide on the number of components
        perform_PCA(participant_df, decide=True)
    """
    
    max_components=len(df.columns)-1
    if decide:
        remaining_variance = []
        for nb_components in range(2, max_components):
            pca = PCA(n_components=nb_components)
            pca.fit(df.drop('participant_id', axis=1))
            explained_variance = pca.explained_variance_ratio_
            remaining_variance.append(1 - np.sum(explained_variance))

        # Plot remaining variance
        plt.scatter(range(2, max_components), remaining_variance)
        plt.ylim(0, 1)
        plt.title('Non-Explained variance as a function of Component number')
        plt.xlabel('Number of Components')
        plt.ylabel('Remaining Variance')
        plt.show()

        # Calculate and plot the drop in explained variance
        drop = [remaining_variance[k] - remaining_variance[k + 1] for k in range(len(remaining_variance) - 1)]
        
        plt.scatter(range(3,len(drop)+3), drop)
        plt.xticks(range(3,len(drop)+3))
        plt.title('Percentage of explained variance gained by adding one component')
        plt.xlabel('Rank of added Components')
        plt.ylabel('Drop in Remaining Variance')
        plt.show()

    else:
        pca = PCA(n_components=pc_nb)
        pca.fit(df.drop('participant_id', axis=1))
        x_pca = pca.transform(df.drop('participant_id', axis=1))
        
        print('participant_df.shape : ', df.shape)
        print('x_pca.shape : ', x_pca.shape)
        print('\n ----------------------------------------')

        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        print('Explained variance: ', explained_variance)
        print('Portion of remaining variance: ', 1 - np.sum(explained_variance))
        print('\n ----------------------------------------')

        # Plot the data on the 2 main PC's axes
        plt.figure(figsize=(8, 6))
        plt.scatter(x_pca[:, 0], x_pca[:, 1])
        plt.xlabel('1st PC')
        plt.ylabel('2nd PC')
        plt.title('Data on the First Two Principal Components')
        plt.show()
        print('\n ----------------------------------------')
        
        if heatmap:
            df_comp = pd.DataFrame(pca.components_, columns=df.drop('participant_id', axis=1).columns)
            df_comp_transposed = df_comp.T

            plt.figure(figsize=(12, 6))
            sns.heatmap(df_comp_transposed, annot=True, cmap='viridis', fmt='.2f', annot_kws={"size": 8})
            plt.xlabel('Principal Components')
            plt.ylabel('Features')
            plt.title('Heatmap of Features vs. Principal Components')
            plt.show()
    
        return x_pca




    
def perform_Kmeans_clustering(x_pca, cluster_nb=5, decide=True):
    """
    Perform K-means clustering on PCA-transformed data and optionally decide on the number of clusters using the elbow method.

    Args:
        x_pca (np.ndarray): PCA-transformed data. Can also use data with reduced dimensionality (2-d) but with another method.
        cluster_nb (int): Number of clusters to use if decide is False. Default is 5.
        decide (bool): If True, use the elbow method to help decide the number of clusters. Default is True.

    Returns:
        None
        
    Example usage:
        x_pca = perform_PCA(participant_df, pc_nb=3, decide=False)
        perform_Kmeans_clustering(x_pca, cluster_nb=5, decide=True)
        
    """

    if decide:
        sse = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=101)
            kmeans.fit(x_pca)
            sse.append(kmeans.inertia_)

        # Plot SSE for the elbow method
        plt.plot(range(1, 11), sse)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Sum of Squared Errors')
        plt.title('Elbow Method for Determining Optimal Number of Clusters')
        plt.show()
        
        print("Run the function with 'decide=False' to perform the clustering once you have chosen the right number of clusters.")
    
    else:
        kmeans = KMeans(n_clusters=cluster_nb, init='k-means++', max_iter=300, n_init=10, random_state=101)
        kmeans.fit(x_pca)

        # Plot the K-means clustering result
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.xlabel('1st Principal Component')
        plt.ylabel('2nd Principal Component')
        plt.title('K-means Clustering on PCA-transformed Data')
        plt.show()

