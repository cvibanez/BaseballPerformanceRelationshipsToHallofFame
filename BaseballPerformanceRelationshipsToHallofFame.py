# Conrad Ibanez
# Project Title- Baseball Performance Data and Relationships to Hall of Fame Selection
# Dataset- Kaggle-https://www.kaggle.com/open-source-sports/baseball-databank. 
# This dataset is a collection of historical baseball data from 1871 to 2015.  It contains 20 files, and the main tables include a master table for player and biographical info, batting statistics, pitching statistics, and fielding statistics.  There are also tables that contain information for player salaries, awards, All-Star appearances, and Hall of Fame votes.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist 
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.offline import plot



def readTables():
    
    '''
    For this project, I am interested primarily on offensive metrics.
    Data will be taken from Master.csv containing player data, Batting.csv containing offensive stats, and HallOfFame.csv indicating players in the Hall of Fame (HoF).
    
    '''
    try:
        # Attempt to process the data from csv files.
        master_baseball_df = pd.read_csv("baseball/Master.csv")
        master_batting_df = pd.read_csv("baseball/Batting.csv")
        master_hall_of_fame_df = pd.read_csv("baseball/HallOfFame.csv")
        
    except Exception as e:
        print('Error has occurred processing file:',e)
    
    player_df = master_baseball_df[['playerID','nameFirst','nameLast']]    
    batting_df = master_batting_df[['playerID','G','AB','R','H','2B','3B','HR','RBI']]
    # Every entry should have the playerID but all other metrics can be filled with 0 since they are numbers
    batting_df = batting_df.fillna(0)

    # Each batting row corresponds to a year/season for a player
    # Player can have many batting rows for each year/season played
    # Need to group batting data by playerID and sum the data to get career totals to analyze for Hall of Fame (HoF) criteria
    
    batting_df_groups = batting_df.groupby('playerID',as_index=False)
    sum_batting_df = batting_df_groups.sum()
    
    # Calculate the number of years/seasons played
    count_batting_df = batting_df_groups.count()
    count_batting_df = count_batting_df[['playerID','G']]
    count_batting_df = count_batting_df.rename(columns={"G":"years"})

    # Get the player data together
    hof_batting_df = pd.merge(count_batting_df,sum_batting_df, on=['playerID'],how='outer')
    
    # Get the Hall of Fame (HoF) data together
    hall_of_fame_df = master_hall_of_fame_df[['playerID','yearid','inducted','category']]
    hall_of_fame_df = hall_of_fame_df[(hall_of_fame_df['inducted']=='Y') & (hall_of_fame_df['category']=='Player')]

    # Combine player data and Hall of Fame data together to have one dataframe for analysis
    # Handle missing data, clean up, and format data as necesary
    data_analysis_df = pd.merge(player_df,hof_batting_df, on=['playerID'],how='right')
    data_analysis_df = pd.merge(data_analysis_df,hall_of_fame_df, on=['playerID'],how='left')
    data_analysis_df['yearid'] = pd.to_datetime(data_analysis_df['yearid'], format='%Y')
    hof_fill_values = {'yearid':pd.NaT, 'inducted': 'N', 'category': 'none'}
    data_analysis_df = data_analysis_df.fillna(value=hof_fill_values)
    data_analysis_df['inducted_val'] = [1 if x =='Y' else 0 for x in data_analysis_df['inducted']] 
    
    #return the main data frame for analysis
    return data_analysis_df
    
def createStatisticsHeatmap(data_analysis_df):
    
    '''
    Creating statistics heatmap to show relationships between offensive metrics data
    '''
    
    print('-'*50, sep='')
    print('Displaying statistics heatmap for some offensive metrics')
    print('-'*50, sep='')
    
    hf_entry_analysis_df = data_analysis_df[['years','G','AB','H','HR','RBI','R','2B','3B','inducted_val']]
  
    # generating correlation heatmap 
    sns.heatmap(hf_entry_analysis_df.corr(), annot = True) 
      
    # posting correlation heatmap to output console
    plt.savefig('HOF_Statistics_Heatmap.jpg')
    plt.show() 
    
    
def exploreData(data_analysis):
    
    '''
    Exploring the data.
    '''
    print('-'*50, sep='')
    print('Exploring the data')
    print('-'*50, sep='')
    
    hof_member_df = data_analysis[data_analysis.inducted_val == 1]
    
    print('-'*35, sep='')
    print('Displaying Hall of Fame data metrics')
    print('-'*35, sep='')
    print('Shape of Hall of Fame dataset:', hof_member_df.shape)
    
    maxValues = hof_member_df.max()
    print('Maximum value in each column for Hall of Fame data : ')
    print(maxValues)
    
    #print(hof_member_df.max())
    
    minValues = hof_member_df.min()
    print('Minimum value in each column for Hall of Fame data : ')
    print(minValues)
    #print(hof_member_df.min())
    
    print('Max years:\n',hof_member_df[hof_member_df.years == hof_member_df.years.max()])
    print('Min years:\n',hof_member_df[hof_member_df.years == hof_member_df.years.min()])
    print('Max games:\n',hof_member_df[hof_member_df.G == hof_member_df.G.max()])
    print('Min games:\n',hof_member_df[hof_member_df.G == hof_member_df.G.min()])
    print('Max AB:\n',hof_member_df[hof_member_df.AB == hof_member_df.AB.max()])
    print('Min AB:\n',hof_member_df[hof_member_df.AB== hof_member_df.AB.min()])
    print('Max H:\n',hof_member_df[hof_member_df.H == hof_member_df.H.max()])
    print('Min H:\n',hof_member_df[hof_member_df.H== hof_member_df.H.min()])
    print('Max HR:\n',hof_member_df[hof_member_df.HR == hof_member_df.HR.max()])
    print('Min HR:\n',hof_member_df[hof_member_df.HR== hof_member_df.HR.min()])
    print('Max RBI:\n',hof_member_df[hof_member_df.RBI == hof_member_df.RBI.max()])
    print('Min RBI:\n',hof_member_df[hof_member_df.RBI== hof_member_df.RBI.min()])
    
    print('-'*35, sep='')
    print('Displaying All Players data metrics')
    print('-'*35, sep='')
    
    print('Shape of All Players data:', data_analysis.shape)
    maxValues = data_analysis.max()
    print('Maximum value in each column for All Players data : ')
    print(maxValues)
    
    print('Max years:\n',data_analysis[data_analysis.years == data_analysis.years.max()])
    print('Max games:\n',data_analysis[data_analysis.G == data_analysis.G.max()])
    print('Max AB:\n',data_analysis[data_analysis.AB == data_analysis.AB.max()])
    print('Max H:\n',data_analysis[data_analysis.H == data_analysis.H.max()])
    print('Max HR:\n',data_analysis[data_analysis.HR == data_analysis.HR.max()])
    print('Max RBI:\n',data_analysis[data_analysis.RBI == data_analysis.RBI.max()])
   
    print("\nSaving data_analysis_df to file. \n")
    writer = pd.ExcelWriter("data_analysis_df.xlsx")
    data_analysis.to_excel(writer, "analysis")
    writer.save()
    
    print("\nSaving hof_member_df to file. \n")
    writer = pd.ExcelWriter("hof_member_df.xlsx")
    hof_member_df.to_excel(writer, "analysis")
    writer.save()
    
    
def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)

def compute_gap(clustering, data, k_max=5, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))
    
    ondata_inertia = []
    for k in range(1, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


#Functions provided by the professor
def compute_ssq(data, k, kmeans):
    dist = np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)
    tot_withinss = sum(dist**2) # Total within-cluster sum of squares
    totss = sum(pdist(data)**2) / data.shape[0] # The total sum of squares
    betweenss = totss - tot_withinss # The between-cluster sum of squares
    return betweenss/totss*100

#Given a data (as nxm matrix) and an array of ks, this returns the SSQ (sum of squared distances)
#SSQ is also called as SSD or SSE
def ssq_statistics(data, ks, ssq_norm=True):
    ssqs = sp.zeros((len(ks),)) # array for SSQs (length ks)

    for (i,k) in enumerate(ks): # iterate over the range of k values
        kmeans = KMeans(n_clusters=k, random_state=1234).fit(data)
        if ssq_norm:
            ssqs[i] = compute_ssq(data, k, kmeans)
        else:
            # The sum of squared error (SSQ) for k
            ssqs[i] = kmeans.inertia_
    
    return ssqs

#Method to plot the gap
def plot_gap(gap, k_max):
    
    plt.clf()   #clear the plot
    plt.plot(range(1, k_max+1), gap, '-o')
    plt.ylabel('gap')
    plt.xlabel('k')
    plt.savefig('Plot-Gap.pdf')
    plt.show()


#Method to plot the inertia
def plot_inertia(reference_inertia, ondata_inertia, k_max):    

    plt.clf()#clear the plot
    plt.plot(range(1, k_max+1), reference_inertia,
             '-o', label='reference')
    plt.plot(range(1, k_max+1), ondata_inertia,
             '-o', label='data')
    plt.xlabel('k')
    plt.ylabel('log(inertia)')
    plt.savefig('Plot-Inertia.pdf')
    plt.show()
    
#Method to plot the SSQ
def plot_ssq(k_values, ssqs):    

    plt.clf()#clear the plot
    plt.figure()
    plt.plot(k_values, ssqs)
    plt.xlabel("Number of cluster")
    plt.ylabel("SSQ")
    plt.show()
    
def evaluate_kmeans(data_analysis):
    
    data_analysis_df = data_analysis[['AB','H','HR','RBI','inducted_val']]
    
    hof_entry_array = data_analysis_df.to_numpy()
    #hof_entry_target = hof_entry_array[:, -1] # for last column
    #print('target--', hof_entry_target)
    # Get all columns except for last
    data_analysis_array  = hof_entry_array[:,:-1]
    #print('array', hof_entry_array)
 
    k_values = [x for x in range(1,10)]
    ssqs=ssq_statistics(data_analysis_array, k_values)
 
    print('\n\nSSQ Plot for k= 1 to 10\n','-'*40, sep='')
    plot_ssq(k_values,ssqs)
    
 
    k_max = max(k_values)
    gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), data_analysis_array, k_max)
    print('\n\nGap plot for k= 10\n','-'*40, sep='')
    plot_gap(gap, k_max)  
    print('\nInertia plot for k= 10\n','-'*40, sep='')
    plot_inertia(reference_inertia, ondata_inertia, k_max)
    

def perform_reduction_analysis(data_analysis, target_names):
    # Reference code- https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
    
    data_analysis_df = data_analysis[['AB','H','HR','RBI','inducted_val']]
    data_analysis_array = data_analysis_df.to_numpy()
    analysis_target = data_analysis_array[:, -1] # for last column

    # Get all columns except and last
    data_analysis_array = data_analysis_array[:,:-1]   
    
    X = data_analysis_array
    y = analysis_target
    y = y.astype('int')
    pca = PCA(n_components=2)
    #X_r = pca.fit(X).transform(X)
    X_r = pca.fit_transform(X)
    
    print('X_r', X_r)
    
    data_analysis['pc_1'] = X_r[:,0] # sets this first principal component
    data_analysis['pc_2'] = X_r[:,1] # sets this second principal component
    
    hof_low_pc = data_analysis[(data_analysis['inducted_val']==1) & (data_analysis['pc_1']<2000)]
    print('hof_low_pc:', hof_low_pc.shape)
    
    non_hof_high_pc = data_analysis[(data_analysis['inducted_val']==0) & ((data_analysis['pc_1']> 9000) | (data_analysis['pc_2'] > 750) | (data_analysis['pc_2'] < -5000))]
    print('non_hof_high_pc:', non_hof_high_pc.shape)
    
    print("\nSaving PCA Analysis data pcaAnalysisData.xlsx \n")
    writer = pd.ExcelWriter("pcaAnalysisData.xlsx")
    data_analysis.to_excel(writer, "analysis")
    writer.save()
    
    print("\nSaving PCA Hall of Fame Outliers pcaHallOfFameOutliers.xlsx \n")
    writer = pd.ExcelWriter("pcaHallOfFameOutliers.xlsx")
    hof_low_pc.to_excel(writer, "analysis")
    writer.save()
    
    print("\nSaving PCA Non-Hall of Fame Outliers pcaNonHallOfFameOutliers.xlsx \n")
    writer = pd.ExcelWriter("pcaNonHallOfFameOutliers.xlsx")
    non_hof_high_pc.to_excel(writer, "analysis")
    writer.save()
    
    #lda = LinearDiscriminantAnalysis()
    #X_r2 = lda.fit(X, y).transform(X)
    
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))
    
    plt.figure()
    colors = ['navy', 'turquoise']
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset')
    

    plt.show()
    return hof_low_pc, non_hof_high_pc
    

def get_kmeans_clusters(X, number_clusters):
        # Reference https://www.kaggle.com/minc33/visualizing-high-dimensional-clusters
    # Assume data_analysis contains only numerical variables
    
    #Initialize our scaler
    scaler = StandardScaler()
    
    #Scale each column in numer
    X = pd.DataFrame(scaler.fit_transform(X))
       
    #Initialize our model
    kmeans = KMeans(n_clusters = number_clusters)
                                    
    #Fit our model
    kmeans.fit(X)
    
    #Find which cluster each data-point belongs to
    clusters = kmeans.predict(X)
    
    return clusters
         

def perform_kmeans(X):
    # Reference https://www.kaggle.com/minc33/visualizing-high-dimensional-clusters
    # Assume data_analysis contains only numerical variables
    
    number_clusters = 2
    
    #Find which cluster each data-point belongs to
    clusters = get_kmeans_clusters(X, number_clusters)
    
    #Add the cluster vector to our DataFrame, X
    X['Cluster'] = clusters
    
    # Map the cluster values to match target values
    X['Cluster'] = X['Cluster'].map({0:1,1:0})
    
    
    #plotX is a DataFrame containing 5000 values sampled randomly from X
    plotX = pd.DataFrame(np.array(X))

    #Rename plotX's columns since it was briefly converted to an np.array above
    plotX.columns = X.columns
    
    #PCA with one principal component
    pca_1d = PCA(n_components=1)
    
    #PCA with two principal components
    pca_2d = PCA(n_components=2)
    
    
    #This DataFrame holds that single principal component mentioned above
    PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))
    
    #This DataFrame contains the two principal components that will be used
    #for the 2-D visualization mentioned above
    PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))
    

    
    PCs_1d.columns = ["PC1_1d"]

    #"PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
    #And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    
    
    plotX = pd.concat([plotX,PCs_1d,PCs_2d], axis=1, join='inner')
    
    plotX["dummy"] = 0
    
    #Note that all of the DataFrames below are sub-DataFrames of 'plotX'.
    #This is because we intend to plot the values contained within each of these DataFrames.
    
    cluster0 = plotX[plotX["Cluster"] == 0]
    cluster1 = plotX[plotX["Cluster"] == 1]
    #cluster2 = plotX[plotX["Cluster"] == 2]
    
    #Instructions for building the 1-D plot

    #trace1 is for 'Cluster 0'
    trace1 = go.Scatter(
                        x = cluster0["PC1_1d"],
                        y = cluster0["dummy"],
                        mode = "markers",
                        name = "Cluster 0",
                        marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                        text = None)
    


    
    #trace2 is for 'Cluster 1'
    trace2 = go.Scatter(
                        x = cluster1["PC1_1d"],
                        y = cluster1["dummy"],
                        mode = "markers",
                        name = "Cluster 1",
                        marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                        text = None)


    data = [trace1, trace2]
    
    title = "Visualizing Clusters in One Dimension Using PCA"
    
    layout = dict(title = title,
                  xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                  yaxis= dict(title= '',ticklen= 5,zeroline= False)
                 )
    
    fig = dict(data = data, layout = layout)
    
    plot(fig, filename='1D-PCA.html')
    
    
    #Instructions for building the 2-D plot

    #trace1 is for 'Cluster 0'
    trace1 = go.Scatter(
                        x = cluster0["PC1_2d"],
                        y = cluster0["PC2_2d"],
                        mode = "markers",
                        name = "Cluster 0",
                        marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                        text = None)
    
    #trace2 is for 'Cluster 1'
    trace2 = go.Scatter(
                        x = cluster1["PC1_2d"],
                        y = cluster1["PC2_2d"],
                        mode = "markers",
                        name = "Cluster 1",
                        marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                        text = None)
    
    data = [trace1, trace2]
    
    title = "Visualizing Clusters in Two Dimensions Using PCA"
    
    layout = dict(title = title,
                  xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
                  yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
                 )
    
    fig = dict(data = data, layout = layout)
    
    plot(fig, filename='2D-PCA.html')
    
    return clusters
    
    
def analyze_kmeans(data_analysis, hof_pc_outliers):
    
    data_analysis_df = data_analysis[['AB','H','HR','RBI']]
    clusters = perform_kmeans(data_analysis_df)
    
    data_analysis['cluster_k2'] = clusters
    # Map the cluster values to match target values
    data_analysis['cluster_k2'] = data_analysis['cluster_k2'].map({0:1,1:0})
    # Set clusters to new mapping
    clusters = data_analysis['cluster_k2']
    
    print('-'*50, sep='')
    print('Analyzing kmeans results when k = 2')
    print('-'*50, sep='')
    
    confusion_data = confusion_matrix(data_analysis['inducted_val'].tolist(), clusters)
    # Used as reference: https://stats.stackexchange.com/questions/95731/how-to-calculate-purity
    print('confusion_matrix',confusion_data)
    max_values = 0
    for i in range(2):
        max_values += max(confusion_data[i])
    purity_score = max_values/data_analysis.shape[0]
    print("Purity score = ", purity_score)
    
    hof_outliers = hof_pc_outliers[['playerID','nameFirst','nameLast']]
    data_analysis_false_neg = data_analysis[(data_analysis['inducted_val']==1) & (data_analysis['cluster_k2']==0)]
    data_analysis_false_neg = data_analysis_false_neg[['playerID','nameFirst','nameLast']]
    
    print('Shape of Hall of Fame outliers (Players in HoF but do not have impressive numbers.):', hof_outliers.shape)
    print('Shape of dataframe containing Hall of Fame Players but in Cluster indicated not in HOF:', data_analysis_false_neg.shape)
    diff_df = pd.concat([hof_outliers,data_analysis_false_neg]).drop_duplicates(keep=False)
    print('Difference between the two sets:', diff_df.to_string())
    
    '''
    print("\nSaving data to hof_outliers.xlsx \n")
    writer = pd.ExcelWriter("hof_outliers.xlsx")
    hof_outliers.to_excel(writer, "analysis")
    writer.save()
    
    print("\nSaving data to data_analysis_false_neg.xlsx \n")
    writer = pd.ExcelWriter("data_analysis_false_neg.xlsx")
    data_analysis_false_neg.to_excel(writer, "analysis")
    writer.save()
    '''
    
    # Analyze other values when k=3
    print('-'*50, sep='')
    print('Analyzing kmeans results when k = 3 and combining two clusters into one to get two clusters only')
    print('-'*50, sep='')
    number_clusters = 3
    cluster_k3 = get_kmeans_clusters(data_analysis_df, number_clusters)
    data_analysis['cluster_k3'] = cluster_k3
    data_analysis['cluster_k3'] = data_analysis['cluster_k3'].map({0:1,1:0,2:2})
    # Map the cluster values to match target values
    data_analysis['cluster_k3_revised'] = data_analysis['cluster_k3'].map({0:1,1:0,2:0})
    cluster_k3 = data_analysis['cluster_k3_revised']
    
    confusion_data = confusion_matrix(data_analysis['inducted_val'].tolist(), cluster_k3)
    # Used as reference: https://stats.stackexchange.com/questions/95731/how-to-calculate-purity
    print('confusion_matrix',confusion_data)
    max_values = 0
    for i in range(2):
        max_values += max(confusion_data[i])
    purity_score = max_values/data_analysis.shape[0]
    print("Purity score = ", purity_score)
    
    print("\nSaving data to k_Means_Analysis.xlsx \n")
    writer = pd.ExcelWriter("k_Means_Analysis.xlsx")
    data_analysis.to_excel(writer, "kmeans_Analysis")
    writer.save()
    
    
def main():
    
    np.random.seed(20)  # Initiate the random seed
    
    target_names = ['Not Hall of Famer', 'Hall of Famer']
    
    data_analysis_df = readTables()
    
    createStatisticsHeatmap(data_analysis_df)
    
    exploreData(data_analysis_df)

    
    hof_pc_outliers, non_hof_pc_outliers = perform_reduction_analysis(data_analysis_df, target_names)
    
    analyze_kmeans(data_analysis_df, hof_pc_outliers)
    
    #evaluate_kmeans(data_analysis_df)
    
    
if __name__ == '__main__':
    main()