
# World Bank Project Clustering - v1.1


*MIT Megacity Logistics Lab*(c): Esteban Mascarino <estmasca@mit.edu>, Daniel Merchan <dmerchan@mit.edu>, Matthias Winkenbach <mwinkenb@mit.edu>

**Summary**: This script reads from the input tables for each city involved in the project (Bogota, Lima and Quito). Those tables summarize the information about population, road network and economic censis at a pixel level for each place. With that data, the code first applies Principal Conponent Analysis (PCA) in order to summarize the information of several numerical variables in a subset of transformed significant variables which explain a certain % of the total variability. Afterwards, by means of K-means we clusters every pixel into a predefined number of groups.

## Scripts

### Loading Modules and Connections

Loading required libraries


```python
# General libraries
from __future__ import division
import numpy as np
import math as m
import pandas as pd
pd.options.display.max_columns = 30
import scipy.linalg
import os

# Specific data science packages
import sklearn.cluster as skcl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.preprocessing import PolynomialFeatures as PolyFeat
from sklearn.preprocessing import StandardScaler as Standardize
from sklearn.preprocessing import scale

from sklearn.model_selection import KFold as KFold
from sklearn.cross_validation import cross_val_score as CV
from sklearn.model_selection import train_test_split as Split
from sklearn.feature_selection import f_regression as RegTest

from wpca import WPCA, EMPCA

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Visualization libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
%matplotlib inline
import seaborn as sns
from IPython.display import Image
from IPython.display import display
import pydotplus

# System libraries
from itertools import combinations
import itertools as it

# Library to convert coordinates from LatLon to UTM
import utm

# Libraries for data visualization
import matplotlib.pyplot as plt
%matplotlib inline

# Libraries for handling temporal data or monitoring processing time
import datetime as dt
from datetime import datetime, date
import time

#Geospatial packages
from geopandas import GeoDataFrame
from shapely.geometry import Point

#Specific data science packages
from sklearn.linear_model import LinearRegression as Lin_Reg
from sklearn.linear_model import LogisticRegression as Log_Reg
from sklearn.linear_model import Ridge as Ridge_Reg
from sklearn.linear_model import Lasso as Lasso_Reg
from sklearn.tree import DecisionTreeRegressor as Reg_Tree
from sklearn.ensemble import RandomForestRegressor as Reg_Forest
from sklearn.ensemble import RandomForestClassifier as Cls_Forest
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS as OLS

from sklearn.preprocessing import PolynomialFeatures as PolyFeat
from sklearn.preprocessing import StandardScaler as Standardize

from sklearn.model_selection import KFold as KFold
from sklearn.model_selection import cross_val_score as CV
from sklearn.model_selection import train_test_split as Split
from sklearn.feature_selection import f_regression as RegTest

from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans as KMeans
from sklearn.cluster import DBSCAN as DBSCAN
import sklearn.cluster as skcl

import statsmodels.api as sm
import statsmodels.formula.api as smf

#Visualization libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.path import Path
import seaborn as sns

#System libraries
from itertools import combinations
import itertools as it
```

    /Users/maggiewilson/anaconda/lib/python2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /Users/maggiewilson/anaconda/lib/python2.7/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


### Data Uploading

Reads the associated file from the "Input" folder.


```python
def read_file(file_csv):
    """
    Input:
    - file_csv: file path to a .csv file containing all the required information for each pixel.
    
    Output:
    - pixs: data frame of input values.
    """
    
    pixs = pd.read_csv(file_csv, index_col=["pixel_ID"])
    pixs.index.name=None #Needed to remove extra row created when using ID as index
    
    print 'Number of pixels:', len(pixs)
    return pixs
```

### Data Processing

Filters those pixels that does not reach the minimum population level required and also those that do not have any recorded road infrastructural data.


```python
def filter_pixels_list_pop(pixs, min_pop=1000):
    """
    Inputs:
    - pixs: data frame containing the pixels to be filtered.
    - min_pop: number representing the minimum number of people required in a 1 sq. km pixel.
    
    Output:
    - pixs_output: data frame containing the filtered pixels.
    """
    # Filter by population
    pixs_output = pixs[pixs['population'] >= min_pop]
    
    print 'Number of pixels before filtering:', len(pixs)
    print 'Number of pixels after filtering:', len(pixs_output)
    return pixs_output
```


```python
def filter_pixels_list_network(pixs, column_name, criteria, below = False):
    """
    Inputs:
    - pixs: data frame containing the pixels to be filtered.
    - column_name: column to be used for filtering
    - citeria: filtering criteria
    - below: if True, function will filter all values below the certain. The default is filter aboive 
    
    Output:
    - pixs_output: data frame containing the filtered pixels.
    """
    # Filter by population
    if below:
        #filter below
        pixs_output = pixs[pixs[column_name] < criteria]
    else:
        #filter above
        pixs_output = pixs[pixs[column_name] > criteria]
    
    print 'Number of pixels before filtering:', len(pixs)
    print 'Number of pixels after filtering:', len(pixs_output)
    return pixs_output
```


```python
def explore_predictors(pixs, predictors):
    for predictor in predictors:
        print predictor
        sns.distplot(pixs[predictor])
        plt.show()
    return
```

Removes unnecessary columns from the original data frame.


```python
def remove_columns(input_df,
                   columns_remove = []):
    """
    Inputs:
    - input_df: data frame containing the columns to be removed for PCA.
    - columns_remove: list containing the name of the columns to be removed.
    
    Output:
    - output_df: dataframe ready for PCA with unnecessary columns removed.
    """
    # Clone input_df in order to avoid mutation
    output_df = input_df[:]
    
    # Remove selected columns
    for i in columns_remove:
        del output_df[i]
    
    # Return result
    return output_df
```

### Data Export

Function for downloading already preprocessed data into specific files.


```python
def save_results(data, file_name='file_name.xlsx', save=True):
    
    """
    Save an Excel file from a dataframe.
    
    Input:
    - data (m x n): Dataframe.
    - file_name: File name.
    - save: Optional to save the data.
    
    Output:
    - Saved file.
    """

    # Save an Excel file named "file_name" in the working directory
    if save:
        data.to_excel(file_name, index=False)
    
    return
```

### Normalization & PCA

### Weighted PCA

Applies a PCA analysis after normalizing input data and returns the original data frame with the normalized principal components.

Takes the generalized PCA approach based upon the GSVD (Generalized Singular Value Decomposition) described by I.T. Jollife in sections 3.5 and 14.2.1 of Principal Component Analysis 2nd Edition (2002). Michael Greenacre provides further details about the generalized SVD in Apendix A of Theory and Applications of Correspondence Analysis (1984). Both books are available in ".../01 WorldBank shared MLL/04 Bibliography/Weighted PCA/Useful".

The function gsvd(a, m, w) was obtained from: https://github.com/Darwin2011/GSVD.


```python
def gsvd(a, m, w):
    """
    This function defines the generalized version of the SVD which will then be used for calculating a weighted PCA.
    :param a: Matrix to GSVD
    :param m: 1st Constraint, (u.T * m * u) = I
    :param w: 2nd Constraint, (v.T * w * v) = I
    :return: (u ,s, v)
    """

    (aHeight, aWidth) = a.shape
    (mHeight, mWidth) = m.shape
    (wHeight, wWidth) = w.shape

    assert(aHeight == mHeight)
    assert(aWidth == wWidth)

    mSqrt = scipy.linalg.sqrtm(m)
    wSqrt = scipy.linalg.sqrtm(w)


    mSqrtInv = np.linalg.inv(mSqrt)
    wSqrtInv = np.linalg.inv(wSqrt)

    _a = np.dot(np.dot(mSqrt, a), wSqrt)

    (_u, _s, _v) = np.linalg.svd(_a)

    u = np.dot(mSqrtInv, _u)
    v = np.dot(wSqrtInv, _v.T).T
    s = _s

    return (u, s, v)
```


```python
def gsvd_w_pc_analysis(data_all, exp_var = 95, pixel_identifier = ['index'],
                       obs_weights = None, var_weights = None):
    
    """
    Weighted principal component analysis (based upon the Generalized Single Value
    Decomposition) to transforms the initial set of (correlated) variables 
    into a set of mutually uncorrelated principal components. 
    
    Input:
    - data_all (m x n): Dataframe of data.
    - exp_var: Percentage of the variance to be explained.
    - obs_weights: list of weights for EACH observation in its corresponding order.
                    If None, all observations are equally weighted.
    - var_weights: list of weights for EACH numerical variable in its corresponding order.
                    If None, all numerical variables are equally weighted.
    Output:
    - data_output (m x n): Dataframe including initial data and
                            weighted principal components.
    - n_components_: Number of principal components.
    """
    # Create a copy to avoid mutating the original data frame
    data_all_copy = data_all[:]
    data_all_copy.reset_index(inplace = True)
    
    # Select the columns required for the PCA analysis
    cols_to_use = []
    for i in data_all_copy.columns.tolist():
        if i not in pixel_identifier:
            cols_to_use.append(i)
    data = data_all_copy[cols_to_use]
    
    # Standardize variables by removing the mean and scaling to unit variance
    data_std = Standardize().fit_transform(data)
    
    # Apply the GSVD based upon the selected weights
    if obs_weights != None:
        obs_w = np.diag(np.array(obs_weights))
    else:
        obs_w = np.diag(np.ones(data_std.T.shape[1]))
        
    if var_weights != None:
        var_w = np.diag(np.array(var_weights))
    else:
        var_w = np.diag(np.ones(data_std.T.shape[0]))
    
    u__,s__,v__ = gsvd(data_std.T, var_w, obs_w)
    
    # Generate the pairs of eigenvalues variance and eigenvectors
    eig_pairs = [(np.abs((s__[i]**2)/(data_std.T.shape[1]-1)), u__[:,i]) for i in range(len(s__))]
    eig_pairs.sort()
    eig_pairs.reverse()
    
    # Determine the number of PCs required
    tot_var = np.abs((s__**2)/(len(data)-1)).sum()
    cum_var = 0
    n_components_ = 0
    for i in eig_pairs:
        n_components_ += 1
        cum_var += i[0]
        if cum_var*100/tot_var >= exp_var:
            break
    
    # Generate the transformation matrix
    matrix_w = np.hstack(tuple(eig_pairs[i][1].reshape(len(eig_pairs[i][1]),1) for i in range(n_components_)))
    
    # Transfor the original data with the PCs
    data_shaped = data_std.dot(matrix_w)
    
    print 'Number of principal components:',n_components_
    print 'Explained variance:',round(cum_var*100/tot_var,1),'%'
    
    # Reshape the array and convert to DataFrame
    #data_shaped = data_transf.reshape((len(data),n_components_))
    pcomponents = pd.DataFrame({'Prin'+str(i+1):data_shaped[:,i] for i in range(n_components_)})

    # Concatanate datasets
    frames = [data_all_copy, pcomponents]
    PCAdata = pd.concat(frames,axis=1) 
    
    return PCAdata, n_components_
```

#### Unweighted PCA

Applies a PCA analysis after normalizing input data and returns the original data frame with the normalized principal components.


```python
def pc_analysis(data_all, exp_var = 95 , pixel_identifier = ['index']):
    
    """
    Principal component analysis to transforms the initial set of (correlated) variables 
    into a set of mutually uncorrelated principal components. 
    
    Input:
    - data_all (m x n): Dataframe of data.
    - exp_var: Percentage of the variance to be explained.
    
    Output:
    - data_output (m x n): Dataframe including initial data and principal components.
    - pca.n_components_: Number of principal components.
    """
    # Create a copy to avoid mutating the original data frame
    data_all_copy = data_all[:]
    data_all_copy.reset_index(inplace = True)
    
    # Select the columns required for the PCA analysis
    cols_to_use = list(set(data_all_copy.columns.tolist())-set(pixel_identifier))
    data = data_all_copy[cols_to_use]
    
    # Standardize variables by removing the mean and scaling to unit variance
    data_std = Standardize().fit_transform(data)
     
    # Select the number of components such that the amount of variance to be explained is greater than the percentage set
    pca = PCA(n_components=float(exp_var)/100., svd_solver = 'full')
   
    # Fit the model and apply the dimensionality reduction
    data_transf = pca.fit_transform(data_std)
    print 'Number of principal components:',pca.n_components_
    print 'Explained variance:',round(np.sum(pca.explained_variance_ratio_).tolist()*100.,1),'%'
    
    # Reshape the array and convert to DataFrame
    data_shaped = data_transf.reshape((len(data),pca.n_components_))
    pcomponents = pd.DataFrame({'Prin'+str(i+1):data_shaped[:,i] for i in range(pca.n_components_)})

    # Concatanate datasets
    frames = [data_all_copy, pcomponents]
    PCAdata = pd.concat(frames,axis=1) 
    
    return PCAdata, pca.n_components_
```

### K-means Clustering

Returns the optimal number of cluesters applying K-means. This function uses the concept of the F(k) statistic in order to come up with an estimation of that optimal number.


```python
def run_kmeans(x, k):
    """
    K-means clustering analysis
    
    Input:
    - x (m x n): Numpy array with predictors 
    - k: number of clusteres to fit
    
    Output:
    - x_kmeans (m x n): 
    - k_means: Clustering model
    """
    
    print 'K-means clustering for k=', k
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    
    return kmeans.labels_#, k_means
```


```python
def OptClust(data_all, cols_to_use, minK = 1, maxK = 10, k_add = 2, scaling = False):
    
    """
    Cluster observations that share similar characteriscs using k-means clustering and defining the optimal number of clusters. 
    
    Input:
    - data_all (m x n): Dataframe of data including routes data and principal components.
    - cols_to_use: List with the name of the columns containing the principal components to use for the clustering. 
    - maxK: Maximum number of clusters to be considered. Suggested = 10
    - scaling: Logical value. If True, the clustering data is scaled before clustering. If False, it is not.
    - k_add: integer representing the additional number of clusters to be calculated appart from the optimal number defined by the
                F(k) estimator.
    
    Output:
    - kopt: Optimal number of clusters.
    - ClustersData (m x n): Dataframe including initial data and clusters allocation for k_opt, k_opt+1, ..., k_opt+k_add. 
    """
    
    def Distortion(cluster):
        
        n = len(cluster)
        Nd = len(cluster[0])
        mydist = 0
        center = [np.mean([row[x] for row in cluster]) for x in range(Nd)]
        for i in range(n):
            mydist += (np.linalg.norm(np.array(cluster[i])-np.array(center)))**2
        
        return mydist

    def GiveFk(data, maxK):

        mySk = []
        alpha = [0]
        myFk = [1]
        Nd = len(data[0])
        for k in range(maxK):
            centr, clust_alloc, inert = skcl.k_means(data, k+1, init='k-means++', n_init=100)  #,n_init=1000
            cluster = [] 
            ClustDist = []
            for c in range(k+1):
                cluster.append([data[x] for x in range(len(clust_alloc)) if clust_alloc[x] == c])
                ClustDist.append(Distortion(cluster[c]))
            mySk.append(sum(ClustDist[x] for x in range(k+1)))

            if k>0:
                if k==1:
                    alpha.append(1-(3/(4*Nd)))
                else:
                    alpha.append(alpha[k-1]+((1-alpha[k-1])/6))            

                if mySk[k-1] == 0:
                    myFk.append(1)
                else:
                    myFk.append(mySk[k]/(alpha[k]*mySk[k-1]))

        return myFk
    
    # Select the columns containing the principal components and Route TLP
    
    data = data_all[cols_to_use]
    
    # Standardize the data
    if scaling:
        data = scale(data)
    else:
        data = data.as_matrix()
            
    myF = GiveFk(data, maxK)
    kopt = np.where(myF == min(myF[minK - 1:]))[0][0] + 1
                        
    # Compute optimal clustering
    centr = []
    clust_alloc = []
    inert = []
    centr_new, clust_alloc_new, inert_new = skcl.k_means(data, kopt, init='k-means++', n_init=100)  
    centr.append(centr_new)
    clust_alloc.append(clust_alloc_new)
    inert.append(inert_new)
    
    # Compute addditional clusterings
    for x in range(1,k_add+1):
        centr_new, clust_alloc_new, inert_new = skcl.k_means(data, kopt+x, init='k-means++', n_init=100)  
        centr.append(centr_new)
        clust_alloc.append(clust_alloc_new)
        inert.append(inert_new)
    
    # Create dataframe and concatenate datasets
    kmeans = pd.DataFrame({'kopt+'+str(i)+'_Clusters':clust_alloc[i][:] for i in range(k_add+1)})
    frames = [data_all, kmeans]    
    ClustersData = pd.concat(frames,axis=1)
    
    print 'Optimal number of clusters:',kopt
    return kopt, ClustersData
```

### Plotting Functions

Plots the associated radar for the centroids charts based on the selected variables to show.


```python
def calculate_centroids(data, cols_clusters, cols_to_delete = ['index']):
    """
    Input:
    - data: data frame containing all the clusters generated with K-means.
    - cols_clusters: list of columns containing the clusters numbers
    - cols_to_delete: list of columns to exclude from the analysis
    
    Output:
    - output_df: data frame containig the centroids for each cluster and clustering option
    """
    # List of columns from which to calculate the centroid
    cols_to_use = list(set(data.columns.tolist())-set(cols_clusters)-set(cols_to_delete))
    
    # Create output DF
    output_df = pd.DataFrame(columns=['number_of_clusters','cluster_id']+cols_to_use)
    for i in cols_clusters:
        df_to_concat = pd.pivot_table(data[cols_to_use+[i]],
                                        index=[i], aggfunc=np.mean)
        df_to_concat['cluster_id'] = df_to_concat.index
        df_to_concat.reset_index(drop=True)
        df_to_concat['number_of_clusters'] = df_to_concat.cluster_id.max()+1
        df_to_concat = df_to_concat[['number_of_clusters','cluster_id']+cols_to_use]
        output_df = pd.concat([output_df, df_to_concat], axis=0)
    
    output_df.reset_index(drop=True)
    
    # Return result
    return output_df
```


```python
def radar_chart(data, num_of_clusters_field, cluster_id_field, to_plot_fields, labels, city_name, save=True):
    
    """
    Create a radar chart for each cluster displaying the standardized variables values.
    
    Input:
    - df (m x n): Dataframe of data to be displayed.
    - num_of_clusters_field: string containing the name of the column describing the
                                number of groups of clusters
    - cluster_id_field: string containing the name of the field with the clusters ID per
                        group
    - to_plot_fields: list column names containing the variables to be plotted in the radar chart
    - labels: list of names to be plotted for each variable. It should be in the same order as the previous list.
    - city_name: string containing the name of the city for the graphs.
    - save: If True, the radar charts are saved. If False, not.
    
    Output:
    - radar charts which are saved as .png files (one for centroid)
    """
    colors = ['#BAB0AC', '#2bbc23', '#a6c1ed', '#f98e02', '#9f22d8', '#f2d709', '#ff0000']
    def _radar_factory(num_vars):
    
        theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
        theta += np.pi/2

        def unit_poly_verts(theta):
            x0, y0, r = [0.5] * 3
            verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
            return verts

        class RadarAxes(PolarAxes):
            name = 'radar'
            RESOLUTION = 1

            def fill(self, *args, **kwargs):
                closed = kwargs.pop('closed', True)
                return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                lines = super(RadarAxes, self).plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.concatenate((x, [x[0]]))
                    y = np.concatenate((y, [y[0]]))
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(theta * 180/np.pi, labels)

            def _gen_axes_patch(self):
                verts = unit_poly_verts(theta)
                return plt.Polygon(verts, closed=True, edgecolor='k')

            def _gen_axes_spines(self):
                spine_type = 'circle'
                verts = unit_poly_verts(theta)
                verts.append(verts[0])
                path = Path(verts)
                spine = Spine(self, spine_type, path)
                spine.set_transform(self.transAxes)
                return {'polar': spine}

        register_projection(RadarAxes)

        return theta
    
    def radar_graph(labels = [], values = [], cluster=0, tot_clusters=1, save=True):
         
        N = len(labels) 
        theta = _radar_factory(N)
        #max_val = max(max(optimum), max(values))

        # Define chart characteristics
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='radar')
        ax.set_ylim(0,1)
        ax.set_yticks(np.arange(0,1,0.2))
        ax.plot(theta, values, color=colors[cluster])
        ax.set_title('Radar-Cluster '+str(int(cluster))+' of '+str(int(tot_clusters)), weight='bold', size='large')
        #ax.plot(theta, optimum, color='r')
        ax.set_varlabels(labels)
        #plt.show()

        # Save the radar chart in the working directory
        if save:
            plt.savefig('Output/'+city_name+'/'+str(int(cluster))+' of '+str(int(tot_clusters))+' radar-cluster.png',
                        dpi=360)
            
        return

    data_pre_copy = data[[num_of_clusters_field,cluster_id_field]+to_plot_fields]
    
    # Iterate for each group of clusters
    for k in data_pre_copy[num_of_clusters_field].unique():
        # Normalize clustering centroids
        data_copy = data_pre_copy[(data_pre_copy[num_of_clusters_field] == k)]
        del data_copy[num_of_clusters_field]
        data_copy = data_copy.set_index(cluster_id_field)
        clusters = data_copy.index.tolist()
        for j in data_copy.columns.tolist():
            mini = data_copy[j].min()
            maxi = data_copy[j].max()
            data_copy[j] = (data_copy[j] - mini) / (maxi - mini)
    
    
        for i in clusters:    
            values = data_copy.loc[i].tolist()
            print "Cluster", str(i)
            radar_graph(labels, values, i, k)    
    
    return
```


```python
def plot_map(df, category_column):
    geo_point = [Point(xy) for xy in zip(df.lon, df.lat)]
    df = df.drop(['lon', 'lat'], axis=1)
    crs = {'init': 'epsg:4326'}
    geo_df = GeoDataFrame(df, crs=crs, geometry=geo_point)
    geo_df.plot(column = category_column, cmap = 'Paired', marker = 's', markersize=15)
    return
```

## Methodology Execution

### Reading input file from the city and preparing data

The user must supply the name of the city from which to perform the analysis based upon the following abreviations:
 - 'BOG': Bogota
 - 'LIM': Lima
 - 'UIO': Quito


```python
city_name = input('Please provide the abreviation for the city to analyze (write it between ''): ')
```

    Please provide the abreviation for the city to analyze (write it between ): 'LIM'


The code will automatically upload the required file.


```python
pixels_data = read_file('Input/'+city_name+'/'+city_name+'_pixels_population_roadsext_cf_ec.csv')
```

    Number of pixels: 4290



```python
predictors = ['population',
              'count_intersections',
              'streets_per_node_avg',
              'betweenness_centrality_avg',
              'closeness_centrality_avg',
              'primary_length_total_ext',
              'highway_length_total_ext',
              'fraction_oneway_ext',
              'clustering_coefficient_avg',
              'N_Est_Mfg',
              'N_Est_BFA',
              'N_Est_RW',
              'N_Emp_Mfg',
              'N_Emp_BFA',
              'N_Emp_RW','N_Emp_Tot', 'POI_Count']
```


```python
explore_predictors(pixels_data, predictors)
```

    population



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_1.png)


    count_intersections



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_3.png)


    streets_per_node_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_5.png)


    betweenness_centrality_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_7.png)


    closeness_centrality_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_9.png)


    primary_length_total_ext



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_11.png)


    highway_length_total_ext



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_13.png)


    fraction_oneway_ext



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_15.png)


    clustering_coefficient_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_17.png)


    N_Est_Mfg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_19.png)


    N_Est_BFA



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_21.png)


    N_Est_RW



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_23.png)


    N_Emp_Mfg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_25.png)


    N_Emp_BFA



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_27.png)


    N_Emp_RW



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_29.png)


    N_Emp_Tot



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_31.png)


    POI_Count



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_40_33.png)


Data is filtered for population and road network.


```python
pixels_data_filtered = filter_pixels_list_pop(pixels_data, min_pop = 1000)
```

    Number of pixels before filtering: 4290
    Number of pixels after filtering: 864



```python
pixels_data_filtered = filter_pixels_list_network(pixels_data_filtered,
                                                  'clustering_coefficient_avg', 0)
```

    Number of pixels before filtering: 864
    Number of pixels after filtering: 663



```python
pixels_data_filtered = filter_pixels_list_network(pixels_data_filtered,
                                                  'clustering_coefficient_avg', 0.2,
                                                  below= True)
```

    Number of pixels before filtering: 663
    Number of pixels after filtering: 654



```python
pixels_data_filtered = filter_pixels_list_network(pixels_data_filtered,
                                                  'betweenness_centrality_avg', 0.25,
                                                  below = True)
```

    Number of pixels before filtering: 654
    Number of pixels after filtering: 654



```python
pixels_data_filtered = filter_pixels_list_network(pixels_data_filtered,
                                                  'betweenness_centrality_avg', 0.02)
```

    Number of pixels before filtering: 654
    Number of pixels after filtering: 653



```python
pixels_data_filtered = filter_pixels_list_network(pixels_data_filtered,
                                                  'streets_per_node_avg', 0.99)
```

    Number of pixels before filtering: 653
    Number of pixels after filtering: 653



```python
pixels_data_filtered = filter_pixels_list_network(pixels_data_filtered,
                                                  'closeness_centrality_avg', 0.005,
                                                 below = True)
```

    Number of pixels before filtering: 653
    Number of pixels after filtering: 652



```python
explore_predictors(pixels_data_filtered, predictors)
```

    population



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_1.png)


    count_intersections



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_3.png)


    streets_per_node_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_5.png)


    betweenness_centrality_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_7.png)


    closeness_centrality_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_9.png)


    primary_length_total_ext



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_11.png)


    highway_length_total_ext



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_13.png)


    fraction_oneway_ext



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_15.png)


    clustering_coefficient_avg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_17.png)


    N_Est_Mfg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_19.png)


    N_Est_BFA



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_21.png)


    N_Est_RW



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_23.png)


    N_Emp_Mfg



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_25.png)


    N_Emp_BFA



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_27.png)


    N_Emp_RW



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_29.png)


    N_Emp_Tot



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_31.png)


    POI_Count



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_49_33.png)



```python
pixels_data_filtered_ready = pixels_data_filtered[predictors]

pixels_data_filtered_ready = remove_columns(pixels_data_filtered_ready, columns_remove =
                                           ['closeness_centrality_avg',
                                            'clustering_coefficient_avg'])
```


```python
pixels_data_filtered_ready
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>count_intersections</th>
      <th>streets_per_node_avg</th>
      <th>betweenness_centrality_avg</th>
      <th>primary_length_total_ext</th>
      <th>highway_length_total_ext</th>
      <th>fraction_oneway_ext</th>
      <th>N_Est_Mfg</th>
      <th>N_Est_BFA</th>
      <th>N_Est_RW</th>
      <th>N_Emp_Mfg</th>
      <th>N_Emp_BFA</th>
      <th>N_Emp_RW</th>
      <th>N_Emp_Tot</th>
      <th>POI_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>117</th>
      <td>1461</td>
      <td>114</td>
      <td>3.041322</td>
      <td>0.059484</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.164404</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>248</th>
      <td>3558</td>
      <td>57</td>
      <td>3.030769</td>
      <td>0.096963</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>312</th>
      <td>1768</td>
      <td>63</td>
      <td>3.260870</td>
      <td>0.076863</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.061655</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>313</th>
      <td>6137</td>
      <td>96</td>
      <td>3.140000</td>
      <td>0.091784</td>
      <td>0.000000</td>
      <td>4060.184994</td>
      <td>0.226727</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>378</th>
      <td>8452</td>
      <td>145</td>
      <td>3.278912</td>
      <td>0.060882</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.184630</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>443</th>
      <td>1416</td>
      <td>23</td>
      <td>3.521739</td>
      <td>0.131752</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.351102</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>444</th>
      <td>1963</td>
      <td>30</td>
      <td>3.027778</td>
      <td>0.118371</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039808</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>508</th>
      <td>10039</td>
      <td>66</td>
      <td>3.185714</td>
      <td>0.084119</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.452238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35</td>
    </tr>
    <tr>
      <th>574</th>
      <td>4149</td>
      <td>9</td>
      <td>2.200000</td>
      <td>0.186813</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>575</th>
      <td>1870</td>
      <td>42</td>
      <td>2.722222</td>
      <td>0.109391</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>635</th>
      <td>1016</td>
      <td>21</td>
      <td>3.095238</td>
      <td>0.152256</td>
      <td>0.000000</td>
      <td>1626.015841</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>638</th>
      <td>1563</td>
      <td>20</td>
      <td>3.150000</td>
      <td>0.158772</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>639</th>
      <td>2623</td>
      <td>19</td>
      <td>3.315789</td>
      <td>0.141899</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027211</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>640</th>
      <td>2120</td>
      <td>12</td>
      <td>2.562500</td>
      <td>0.197619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>700</th>
      <td>3186</td>
      <td>129</td>
      <td>3.079710</td>
      <td>0.069031</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.140474</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>701</th>
      <td>5925</td>
      <td>129</td>
      <td>3.147059</td>
      <td>0.054680</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.073444</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>703</th>
      <td>2573</td>
      <td>131</td>
      <td>3.131034</td>
      <td>0.050211</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>704</th>
      <td>2642</td>
      <td>25</td>
      <td>3.111111</td>
      <td>0.140171</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>705</th>
      <td>3445</td>
      <td>20</td>
      <td>3.190476</td>
      <td>0.134336</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>766</th>
      <td>5774</td>
      <td>191</td>
      <td>3.270833</td>
      <td>0.050461</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.265891</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>768</th>
      <td>7648</td>
      <td>181</td>
      <td>3.316940</td>
      <td>0.047453</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.180476</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>829</th>
      <td>2298</td>
      <td>75</td>
      <td>3.129870</td>
      <td>0.089241</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057823</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>830</th>
      <td>9489</td>
      <td>307</td>
      <td>3.319355</td>
      <td>0.036052</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.263815</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>832</th>
      <td>6887</td>
      <td>159</td>
      <td>3.257862</td>
      <td>0.053697</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.355541</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>833</th>
      <td>6820</td>
      <td>254</td>
      <td>3.205426</td>
      <td>0.037250</td>
      <td>798.744052</td>
      <td>0.000000</td>
      <td>0.180348</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>834</th>
      <td>7112</td>
      <td>199</td>
      <td>3.170616</td>
      <td>0.044921</td>
      <td>677.380694</td>
      <td>0.000000</td>
      <td>0.078081</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>835</th>
      <td>3161</td>
      <td>64</td>
      <td>3.057143</td>
      <td>0.113019</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055204</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>840</th>
      <td>1000</td>
      <td>26</td>
      <td>3.000000</td>
      <td>0.184672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.085998</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>894</th>
      <td>1423</td>
      <td>32</td>
      <td>3.000000</td>
      <td>0.143672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>897</th>
      <td>8544</td>
      <td>290</td>
      <td>3.250000</td>
      <td>0.036473</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.396344</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3653</th>
      <td>2246</td>
      <td>62</td>
      <td>3.014925</td>
      <td>0.077842</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.152148</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3654</th>
      <td>1136</td>
      <td>51</td>
      <td>2.948276</td>
      <td>0.094531</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.286316</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3656</th>
      <td>1393</td>
      <td>28</td>
      <td>3.107143</td>
      <td>0.117623</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.310952</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3657</th>
      <td>1177</td>
      <td>80</td>
      <td>3.219512</td>
      <td>0.059982</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.048138</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3709</th>
      <td>4097</td>
      <td>53</td>
      <td>3.254545</td>
      <td>0.082574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3710</th>
      <td>10769</td>
      <td>190</td>
      <td>3.325000</td>
      <td>0.045522</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3711</th>
      <td>10013</td>
      <td>258</td>
      <td>3.585271</td>
      <td>0.037401</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3712</th>
      <td>6538</td>
      <td>222</td>
      <td>3.646288</td>
      <td>0.042664</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3713</th>
      <td>18333</td>
      <td>215</td>
      <td>3.305085</td>
      <td>0.047301</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3714</th>
      <td>9502</td>
      <td>183</td>
      <td>3.447368</td>
      <td>0.045289</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039784</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3715</th>
      <td>3061</td>
      <td>154</td>
      <td>3.267516</td>
      <td>0.036386</td>
      <td>0.000000</td>
      <td>1845.681177</td>
      <td>0.082943</td>
      <td>171.0</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>30970.0</td>
      <td>0.0</td>
      <td>5443.0</td>
      <td>110070.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3717</th>
      <td>1963</td>
      <td>69</td>
      <td>3.281690</td>
      <td>0.071513</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>5304.0</td>
      <td>0.0</td>
      <td>520.0</td>
      <td>9091.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3719</th>
      <td>1644</td>
      <td>88</td>
      <td>3.217391</td>
      <td>0.064075</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080264</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3721</th>
      <td>2824</td>
      <td>32</td>
      <td>2.815789</td>
      <td>0.113008</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3723</th>
      <td>5009</td>
      <td>12</td>
      <td>2.857143</td>
      <td>0.226190</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.127762</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3726</th>
      <td>1175</td>
      <td>14</td>
      <td>2.764706</td>
      <td>0.144608</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3775</th>
      <td>1218</td>
      <td>22</td>
      <td>2.838710</td>
      <td>0.146830</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3776</th>
      <td>1421</td>
      <td>89</td>
      <td>3.000000</td>
      <td>0.067887</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3779</th>
      <td>6319</td>
      <td>180</td>
      <td>3.063725</td>
      <td>0.042501</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3780</th>
      <td>1442</td>
      <td>156</td>
      <td>2.994286</td>
      <td>0.044316</td>
      <td>0.000000</td>
      <td>1242.905239</td>
      <td>0.010531</td>
      <td>142.0</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>25666.0</td>
      <td>0.0</td>
      <td>4923.0</td>
      <td>100979.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3785</th>
      <td>1731</td>
      <td>13</td>
      <td>2.800000</td>
      <td>0.139927</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3788</th>
      <td>2032</td>
      <td>50</td>
      <td>2.962963</td>
      <td>0.108047</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3844</th>
      <td>2363</td>
      <td>13</td>
      <td>2.687500</td>
      <td>0.152381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3845</th>
      <td>2234</td>
      <td>154</td>
      <td>3.024096</td>
      <td>0.046908</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3907</th>
      <td>1506</td>
      <td>41</td>
      <td>3.404762</td>
      <td>0.089983</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4040</th>
      <td>4533</td>
      <td>72</td>
      <td>3.112500</td>
      <td>0.079286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.096302</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>1227</td>
      <td>54</td>
      <td>3.185185</td>
      <td>0.103847</td>
      <td>0.000000</td>
      <td>1542.810577</td>
      <td>0.186673</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4105</th>
      <td>6060</td>
      <td>165</td>
      <td>3.078652</td>
      <td>0.052909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.064199</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4106</th>
      <td>3668</td>
      <td>130</td>
      <td>3.175573</td>
      <td>0.061268</td>
      <td>0.000000</td>
      <td>3270.108195</td>
      <td>0.088558</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4171</th>
      <td>3753</td>
      <td>48</td>
      <td>3.244898</td>
      <td>0.116279</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055028</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>652 rows × 15 columns</p>
</div>



### Running PCA

Running weighted PCA with the cleaned data and at least 90% of explanatory power. Assigning weight of 20% to population, 30% to infrastructure variables, and 50% to socioeconomic variables.


```python
pca_data, n_components = gsvd_w_pc_analysis(pixels_data_filtered_ready, exp_var=90, var_weights = [1/5,    1/20,1/20,1/20,1/20,1/20,1/20,    1/16,1/16,1/16,1/16,1/16,1/16,1/16,1/16])
```

    Number of principal components: 7
    Explained variance: 93.0 %



```python
pca_data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>population</th>
      <th>count_intersections</th>
      <th>streets_per_node_avg</th>
      <th>betweenness_centrality_avg</th>
      <th>primary_length_total_ext</th>
      <th>highway_length_total_ext</th>
      <th>fraction_oneway_ext</th>
      <th>N_Est_Mfg</th>
      <th>N_Est_BFA</th>
      <th>N_Est_RW</th>
      <th>N_Emp_Mfg</th>
      <th>N_Emp_BFA</th>
      <th>N_Emp_RW</th>
      <th>N_Emp_Tot</th>
      <th>POI_Count</th>
      <th>Prin1</th>
      <th>Prin2</th>
      <th>Prin3</th>
      <th>Prin4</th>
      <th>Prin5</th>
      <th>Prin6</th>
      <th>Prin7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>117</td>
      <td>1461</td>
      <td>114</td>
      <td>3.041322</td>
      <td>0.059484</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.164404</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>38</td>
      <td>7.114727</td>
      <td>1.648241</td>
      <td>-0.716472</td>
      <td>-1.662589</td>
      <td>1.690011</td>
      <td>1.488966</td>
      <td>0.175915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>248</td>
      <td>3558</td>
      <td>57</td>
      <td>3.030769</td>
      <td>0.096963</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>48</td>
      <td>8.239472</td>
      <td>-0.497781</td>
      <td>4.218135</td>
      <td>-3.305417</td>
      <td>1.012448</td>
      <td>0.217245</td>
      <td>-0.444929</td>
    </tr>
    <tr>
      <th>2</th>
      <td>312</td>
      <td>1768</td>
      <td>63</td>
      <td>3.260870</td>
      <td>0.076863</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.061655</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>8.235277</td>
      <td>0.127473</td>
      <td>0.727412</td>
      <td>-1.628162</td>
      <td>1.708633</td>
      <td>2.378518</td>
      <td>-1.185460</td>
    </tr>
    <tr>
      <th>3</th>
      <td>313</td>
      <td>6137</td>
      <td>96</td>
      <td>3.140000</td>
      <td>0.091784</td>
      <td>0.000000</td>
      <td>4060.184994</td>
      <td>0.226727</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>5.607381</td>
      <td>-0.857567</td>
      <td>-2.838239</td>
      <td>-5.324318</td>
      <td>-21.805404</td>
      <td>8.777023</td>
      <td>5.298749</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378</td>
      <td>8452</td>
      <td>145</td>
      <td>3.278912</td>
      <td>0.060882</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.184630</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>5.813062</td>
      <td>4.105240</td>
      <td>-2.443006</td>
      <td>-1.055450</td>
      <td>2.258324</td>
      <td>2.950110</td>
      <td>-1.207996</td>
    </tr>
    <tr>
      <th>5</th>
      <td>443</td>
      <td>1416</td>
      <td>23</td>
      <td>3.521739</td>
      <td>0.131752</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.351102</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>7.220447</td>
      <td>-0.976269</td>
      <td>3.855626</td>
      <td>-7.584883</td>
      <td>1.314609</td>
      <td>2.985858</td>
      <td>-5.452076</td>
    </tr>
    <tr>
      <th>6</th>
      <td>444</td>
      <td>1963</td>
      <td>30</td>
      <td>3.027778</td>
      <td>0.118371</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039808</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.778128</td>
      <td>-2.953580</td>
      <td>6.690823</td>
      <td>-2.168602</td>
      <td>0.573109</td>
      <td>0.846566</td>
      <td>0.330949</td>
    </tr>
    <tr>
      <th>7</th>
      <td>508</td>
      <td>10039</td>
      <td>66</td>
      <td>3.185714</td>
      <td>0.084119</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.452238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35</td>
      <td>5.160865</td>
      <td>2.633752</td>
      <td>2.603606</td>
      <td>-5.491300</td>
      <td>0.870054</td>
      <td>2.333111</td>
      <td>-2.635874</td>
    </tr>
    <tr>
      <th>8</th>
      <td>574</td>
      <td>4149</td>
      <td>9</td>
      <td>2.200000</td>
      <td>0.186813</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>13.047463</td>
      <td>-8.727022</td>
      <td>19.133299</td>
      <td>-0.429795</td>
      <td>-2.138990</td>
      <td>-3.293593</td>
      <td>7.022550</td>
    </tr>
    <tr>
      <th>9</th>
      <td>575</td>
      <td>1870</td>
      <td>42</td>
      <td>2.722222</td>
      <td>0.109391</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>10.640185</td>
      <td>-3.735674</td>
      <td>8.036214</td>
      <td>0.167269</td>
      <td>0.042020</td>
      <td>-0.175620</td>
      <td>3.443778</td>
    </tr>
    <tr>
      <th>10</th>
      <td>635</td>
      <td>1016</td>
      <td>21</td>
      <td>3.095238</td>
      <td>0.152256</td>
      <td>0.000000</td>
      <td>1626.015841</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>9.618602</td>
      <td>-4.938202</td>
      <td>6.757275</td>
      <td>-5.175873</td>
      <td>-8.770859</td>
      <td>3.003045</td>
      <td>1.565444</td>
    </tr>
    <tr>
      <th>11</th>
      <td>638</td>
      <td>1563</td>
      <td>20</td>
      <td>3.150000</td>
      <td>0.158772</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>9.987605</td>
      <td>-3.979870</td>
      <td>8.896803</td>
      <td>-4.784795</td>
      <td>0.543647</td>
      <td>0.296633</td>
      <td>-1.574255</td>
    </tr>
    <tr>
      <th>12</th>
      <td>639</td>
      <td>2623</td>
      <td>19</td>
      <td>3.315789</td>
      <td>0.141899</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027211</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
      <td>9.021302</td>
      <td>-2.448525</td>
      <td>6.530937</td>
      <td>-5.286253</td>
      <td>1.003418</td>
      <td>1.193177</td>
      <td>-3.026523</td>
    </tr>
    <tr>
      <th>13</th>
      <td>640</td>
      <td>2120</td>
      <td>12</td>
      <td>2.562500</td>
      <td>0.197619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>12.386180</td>
      <td>-7.986901</td>
      <td>16.713704</td>
      <td>-2.656744</td>
      <td>-1.253468</td>
      <td>-2.182350</td>
      <td>3.612527</td>
    </tr>
    <tr>
      <th>14</th>
      <td>700</td>
      <td>3186</td>
      <td>129</td>
      <td>3.079710</td>
      <td>0.069031</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.140474</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>3</td>
      <td>6.554672</td>
      <td>0.423184</td>
      <td>-0.469490</td>
      <td>1.318962</td>
      <td>1.428247</td>
      <td>1.704762</td>
      <td>0.110696</td>
    </tr>
    <tr>
      <th>15</th>
      <td>701</td>
      <td>5925</td>
      <td>129</td>
      <td>3.147059</td>
      <td>0.054680</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.073444</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>2</td>
      <td>6.195126</td>
      <td>1.526745</td>
      <td>-1.642301</td>
      <td>2.191782</td>
      <td>1.686763</td>
      <td>2.129217</td>
      <td>-0.032069</td>
    </tr>
    <tr>
      <th>16</th>
      <td>703</td>
      <td>2573</td>
      <td>131</td>
      <td>3.131034</td>
      <td>0.050211</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>8.067006</td>
      <td>1.778042</td>
      <td>-2.265129</td>
      <td>1.791666</td>
      <td>2.209533</td>
      <td>2.416521</td>
      <td>1.185972</td>
    </tr>
    <tr>
      <th>17</th>
      <td>704</td>
      <td>2642</td>
      <td>25</td>
      <td>3.111111</td>
      <td>0.140171</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.963691</td>
      <td>-3.449446</td>
      <td>7.893462</td>
      <td>-3.098225</td>
      <td>0.595067</td>
      <td>0.742412</td>
      <td>-0.570550</td>
    </tr>
    <tr>
      <th>18</th>
      <td>705</td>
      <td>3445</td>
      <td>20</td>
      <td>3.190476</td>
      <td>0.134336</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>9.580116</td>
      <td>-2.850207</td>
      <td>7.114061</td>
      <td>-3.396500</td>
      <td>0.764975</td>
      <td>1.104674</td>
      <td>-1.288542</td>
    </tr>
    <tr>
      <th>19</th>
      <td>766</td>
      <td>5774</td>
      <td>191</td>
      <td>3.270833</td>
      <td>0.050461</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.265891</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>4</td>
      <td>4.560264</td>
      <td>3.701022</td>
      <td>-5.003858</td>
      <td>1.474342</td>
      <td>2.357564</td>
      <td>3.171156</td>
      <td>-1.180550</td>
    </tr>
    <tr>
      <th>20</th>
      <td>768</td>
      <td>7648</td>
      <td>181</td>
      <td>3.316940</td>
      <td>0.047453</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.180476</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>5</td>
      <td>4.600562</td>
      <td>3.987507</td>
      <td>-4.984812</td>
      <td>1.823156</td>
      <td>2.439783</td>
      <td>3.211944</td>
      <td>-1.376860</td>
    </tr>
    <tr>
      <th>21</th>
      <td>829</td>
      <td>2298</td>
      <td>75</td>
      <td>3.129870</td>
      <td>0.089241</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057823</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
      <td>7.511895</td>
      <td>-1.458355</td>
      <td>2.270921</td>
      <td>-0.003671</td>
      <td>1.049518</td>
      <td>1.291518</td>
      <td>-0.762275</td>
    </tr>
    <tr>
      <th>22</th>
      <td>830</td>
      <td>9489</td>
      <td>307</td>
      <td>3.319355</td>
      <td>0.036052</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.263815</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>9</td>
      <td>3.090643</td>
      <td>7.238608</td>
      <td>-9.691996</td>
      <td>4.037817</td>
      <td>3.470729</td>
      <td>3.908363</td>
      <td>-0.369153</td>
    </tr>
    <tr>
      <th>23</th>
      <td>832</td>
      <td>6887</td>
      <td>159</td>
      <td>3.257862</td>
      <td>0.053697</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.355541</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>11</td>
      <td>4.130940</td>
      <td>3.512732</td>
      <td>-3.544550</td>
      <td>-0.191141</td>
      <td>1.906533</td>
      <td>3.033940</td>
      <td>-1.909831</td>
    </tr>
    <tr>
      <th>24</th>
      <td>833</td>
      <td>6820</td>
      <td>254</td>
      <td>3.205426</td>
      <td>0.037250</td>
      <td>798.744052</td>
      <td>0.000000</td>
      <td>0.180348</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>8</td>
      <td>4.004272</td>
      <td>5.550877</td>
      <td>-7.875336</td>
      <td>3.435550</td>
      <td>2.658737</td>
      <td>0.447846</td>
      <td>1.662655</td>
    </tr>
    <tr>
      <th>25</th>
      <td>834</td>
      <td>7112</td>
      <td>199</td>
      <td>3.170616</td>
      <td>0.044921</td>
      <td>677.380694</td>
      <td>0.000000</td>
      <td>0.078081</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>17</td>
      <td>4.797122</td>
      <td>4.123703</td>
      <td>-5.165559</td>
      <td>2.551414</td>
      <td>2.180125</td>
      <td>0.046509</td>
      <td>1.205930</td>
    </tr>
    <tr>
      <th>26</th>
      <td>835</td>
      <td>3161</td>
      <td>64</td>
      <td>3.057143</td>
      <td>0.113019</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055204</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.237356</td>
      <td>-1.793919</td>
      <td>5.133964</td>
      <td>-1.477011</td>
      <td>0.924772</td>
      <td>1.201720</td>
      <td>0.439545</td>
    </tr>
    <tr>
      <th>27</th>
      <td>840</td>
      <td>1000</td>
      <td>26</td>
      <td>3.000000</td>
      <td>0.184672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.085998</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>10.651390</td>
      <td>-5.560985</td>
      <td>11.721956</td>
      <td>-4.776007</td>
      <td>-0.031810</td>
      <td>-0.207172</td>
      <td>-0.365805</td>
    </tr>
    <tr>
      <th>28</th>
      <td>894</td>
      <td>1423</td>
      <td>32</td>
      <td>3.000000</td>
      <td>0.143672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
      <td>9.186110</td>
      <td>-4.951267</td>
      <td>8.535443</td>
      <td>-1.737543</td>
      <td>0.052699</td>
      <td>-0.278792</td>
      <td>-0.451314</td>
    </tr>
    <tr>
      <th>29</th>
      <td>897</td>
      <td>8544</td>
      <td>290</td>
      <td>3.250000</td>
      <td>0.036473</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.396344</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>10</td>
      <td>2.867305</td>
      <td>6.785169</td>
      <td>-8.743875</td>
      <td>2.954167</td>
      <td>3.045589</td>
      <td>3.815264</td>
      <td>-0.402792</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>622</th>
      <td>3653</td>
      <td>2246</td>
      <td>62</td>
      <td>3.014925</td>
      <td>0.077842</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.152148</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
      <td>7.708224</td>
      <td>-1.296204</td>
      <td>2.760071</td>
      <td>-0.242940</td>
      <td>0.749922</td>
      <td>1.362489</td>
      <td>0.236301</td>
    </tr>
    <tr>
      <th>623</th>
      <td>3654</td>
      <td>1136</td>
      <td>51</td>
      <td>2.948276</td>
      <td>0.094531</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.286316</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
      <td>7.692212</td>
      <td>-2.257399</td>
      <td>4.660716</td>
      <td>-1.740929</td>
      <td>0.266960</td>
      <td>1.071139</td>
      <td>0.080019</td>
    </tr>
    <tr>
      <th>624</th>
      <td>3656</td>
      <td>1393</td>
      <td>28</td>
      <td>3.107143</td>
      <td>0.117623</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.310952</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>8.275209</td>
      <td>-2.243143</td>
      <td>5.921565</td>
      <td>-4.488768</td>
      <td>0.475513</td>
      <td>1.631573</td>
      <td>-1.358258</td>
    </tr>
    <tr>
      <th>625</th>
      <td>3657</td>
      <td>1177</td>
      <td>80</td>
      <td>3.219512</td>
      <td>0.059982</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.048138</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>8.130549</td>
      <td>0.562405</td>
      <td>-0.786001</td>
      <td>-0.148490</td>
      <td>1.888048</td>
      <td>2.543923</td>
      <td>-0.353553</td>
    </tr>
    <tr>
      <th>626</th>
      <td>3709</td>
      <td>4097</td>
      <td>53</td>
      <td>3.254545</td>
      <td>0.082574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>5.126004</td>
      <td>-2.403826</td>
      <td>1.641583</td>
      <td>1.050660</td>
      <td>0.790019</td>
      <td>0.967430</td>
      <td>-3.092140</td>
    </tr>
    <tr>
      <th>627</th>
      <td>3710</td>
      <td>10769</td>
      <td>190</td>
      <td>3.325000</td>
      <td>0.045522</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.950692</td>
      <td>2.827299</td>
      <td>-5.115716</td>
      <td>5.166793</td>
      <td>2.214348</td>
      <td>2.384739</td>
      <td>-1.808210</td>
    </tr>
    <tr>
      <th>628</th>
      <td>3711</td>
      <td>10013</td>
      <td>258</td>
      <td>3.585271</td>
      <td>0.037401</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>2</td>
      <td>1.787539</td>
      <td>5.261248</td>
      <td>-10.048357</td>
      <td>5.371859</td>
      <td>3.509995</td>
      <td>3.600469</td>
      <td>-3.467707</td>
    </tr>
    <tr>
      <th>629</th>
      <td>3712</td>
      <td>6538</td>
      <td>222</td>
      <td>3.646288</td>
      <td>0.042664</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.313567</td>
      <td>4.006627</td>
      <td>-9.445107</td>
      <td>4.055533</td>
      <td>3.410307</td>
      <td>3.582967</td>
      <td>-4.447516</td>
    </tr>
    <tr>
      <th>630</th>
      <td>3713</td>
      <td>18333</td>
      <td>215</td>
      <td>3.305085</td>
      <td>0.047301</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.048151</td>
      <td>4.494314</td>
      <td>-4.603223</td>
      <td>6.193876</td>
      <td>2.151706</td>
      <td>2.564550</td>
      <td>-1.243486</td>
    </tr>
    <tr>
      <th>631</th>
      <td>3714</td>
      <td>9502</td>
      <td>183</td>
      <td>3.447368</td>
      <td>0.045289</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039784</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.604941</td>
      <td>3.053710</td>
      <td>-6.047884</td>
      <td>4.063645</td>
      <td>2.449006</td>
      <td>2.858231</td>
      <td>-3.139127</td>
    </tr>
    <tr>
      <th>632</th>
      <td>3715</td>
      <td>3061</td>
      <td>154</td>
      <td>3.267516</td>
      <td>0.036386</td>
      <td>0.000000</td>
      <td>1845.681177</td>
      <td>0.082943</td>
      <td>171.0</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>30970.0</td>
      <td>0.0</td>
      <td>5443.0</td>
      <td>110070.0</td>
      <td>2</td>
      <td>1.348418</td>
      <td>-0.676523</td>
      <td>-7.671264</td>
      <td>3.293536</td>
      <td>-8.973567</td>
      <td>4.379221</td>
      <td>-0.594836</td>
    </tr>
    <tr>
      <th>633</th>
      <td>3717</td>
      <td>1963</td>
      <td>69</td>
      <td>3.281690</td>
      <td>0.071513</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>5304.0</td>
      <td>0.0</td>
      <td>520.0</td>
      <td>9091.0</td>
      <td>2</td>
      <td>7.635872</td>
      <td>-0.148072</td>
      <td>-0.013822</td>
      <td>-0.308361</td>
      <td>1.686361</td>
      <td>2.132816</td>
      <td>-1.547070</td>
    </tr>
    <tr>
      <th>634</th>
      <td>3719</td>
      <td>1644</td>
      <td>88</td>
      <td>3.217391</td>
      <td>0.064075</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080264</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>3</td>
      <td>7.146136</td>
      <td>0.256668</td>
      <td>-0.740441</td>
      <td>0.060128</td>
      <td>1.655740</td>
      <td>2.163562</td>
      <td>-1.050646</td>
    </tr>
    <tr>
      <th>635</th>
      <td>3721</td>
      <td>2824</td>
      <td>32</td>
      <td>2.815789</td>
      <td>0.113008</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>1</td>
      <td>10.232179</td>
      <td>-3.556909</td>
      <td>8.014171</td>
      <td>-0.549979</td>
      <td>0.111697</td>
      <td>0.047302</td>
      <td>2.388190</td>
    </tr>
    <tr>
      <th>636</th>
      <td>3723</td>
      <td>5009</td>
      <td>12</td>
      <td>2.857143</td>
      <td>0.226190</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.127762</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>10.918062</td>
      <td>-7.114726</td>
      <td>16.848065</td>
      <td>-5.752593</td>
      <td>-0.976964</td>
      <td>-1.246141</td>
      <td>0.211838</td>
    </tr>
    <tr>
      <th>637</th>
      <td>3726</td>
      <td>1175</td>
      <td>14</td>
      <td>2.764706</td>
      <td>0.144608</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>11.095935</td>
      <td>-5.519466</td>
      <td>11.075864</td>
      <td>-1.792307</td>
      <td>-0.350198</td>
      <td>-0.696229</td>
      <td>2.316277</td>
    </tr>
    <tr>
      <th>638</th>
      <td>3775</td>
      <td>1218</td>
      <td>22</td>
      <td>2.838710</td>
      <td>0.146830</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>7.627340</td>
      <td>-7.402113</td>
      <td>10.178699</td>
      <td>0.089758</td>
      <td>-0.872388</td>
      <td>-1.695192</td>
      <td>-0.414001</td>
    </tr>
    <tr>
      <th>639</th>
      <td>3776</td>
      <td>1421</td>
      <td>89</td>
      <td>3.000000</td>
      <td>0.067887</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>5.748206</td>
      <td>-2.735252</td>
      <td>1.033739</td>
      <td>3.364322</td>
      <td>0.663051</td>
      <td>0.301562</td>
      <td>-0.319776</td>
    </tr>
    <tr>
      <th>640</th>
      <td>3779</td>
      <td>6319</td>
      <td>180</td>
      <td>3.063725</td>
      <td>0.042501</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>4.194172</td>
      <td>0.918488</td>
      <td>-3.579627</td>
      <td>6.071268</td>
      <td>1.642519</td>
      <td>1.328820</td>
      <td>0.397949</td>
    </tr>
    <tr>
      <th>641</th>
      <td>3780</td>
      <td>1442</td>
      <td>156</td>
      <td>2.994286</td>
      <td>0.044316</td>
      <td>0.000000</td>
      <td>1242.905239</td>
      <td>0.010531</td>
      <td>142.0</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>25666.0</td>
      <td>0.0</td>
      <td>4923.0</td>
      <td>100979.0</td>
      <td>0</td>
      <td>3.760160</td>
      <td>-1.734009</td>
      <td>-4.407463</td>
      <td>4.925109</td>
      <td>-5.908830</td>
      <td>2.506965</td>
      <td>1.798920</td>
    </tr>
    <tr>
      <th>642</th>
      <td>3785</td>
      <td>1731</td>
      <td>13</td>
      <td>2.800000</td>
      <td>0.139927</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>10.878596</td>
      <td>-5.130930</td>
      <td>10.567279</td>
      <td>-1.790558</td>
      <td>-0.254679</td>
      <td>-0.484708</td>
      <td>2.050301</td>
    </tr>
    <tr>
      <th>643</th>
      <td>3788</td>
      <td>2032</td>
      <td>50</td>
      <td>2.962963</td>
      <td>0.108047</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>9.767519</td>
      <td>-2.642534</td>
      <td>5.818971</td>
      <td>-0.735702</td>
      <td>0.681995</td>
      <td>0.714320</td>
      <td>1.307652</td>
    </tr>
    <tr>
      <th>644</th>
      <td>3844</td>
      <td>2363</td>
      <td>13</td>
      <td>2.687500</td>
      <td>0.152381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>8.044514</td>
      <td>-8.151473</td>
      <td>12.211521</td>
      <td>0.536849</td>
      <td>-1.389954</td>
      <td>-2.327158</td>
      <td>0.825027</td>
    </tr>
    <tr>
      <th>645</th>
      <td>3845</td>
      <td>2234</td>
      <td>154</td>
      <td>3.024096</td>
      <td>0.046908</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>4.961208</td>
      <td>-0.593878</td>
      <td>-2.668487</td>
      <td>5.328771</td>
      <td>1.407773</td>
      <td>0.928428</td>
      <td>0.363715</td>
    </tr>
    <tr>
      <th>646</th>
      <td>3907</td>
      <td>1506</td>
      <td>41</td>
      <td>3.404762</td>
      <td>0.089983</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>2</td>
      <td>8.475786</td>
      <td>-0.383027</td>
      <td>1.260831</td>
      <td>-2.691461</td>
      <td>1.844396</td>
      <td>2.549436</td>
      <td>-2.593849</td>
    </tr>
    <tr>
      <th>647</th>
      <td>4040</td>
      <td>4533</td>
      <td>72</td>
      <td>3.112500</td>
      <td>0.079286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.096302</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>0</td>
      <td>8.253958</td>
      <td>0.100771</td>
      <td>2.161373</td>
      <td>-0.639658</td>
      <td>1.292977</td>
      <td>2.059206</td>
      <td>0.286690</td>
    </tr>
    <tr>
      <th>648</th>
      <td>4041</td>
      <td>1227</td>
      <td>54</td>
      <td>3.185185</td>
      <td>0.103847</td>
      <td>0.000000</td>
      <td>1542.810577</td>
      <td>0.186673</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>7</td>
      <td>7.608644</td>
      <td>-1.693240</td>
      <td>1.516269</td>
      <td>-4.832180</td>
      <td>-7.638683</td>
      <td>4.304078</td>
      <td>0.740304</td>
    </tr>
    <tr>
      <th>649</th>
      <td>4105</td>
      <td>6060</td>
      <td>165</td>
      <td>3.078652</td>
      <td>0.052909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.064199</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>7.102299</td>
      <td>3.127608</td>
      <td>-2.364986</td>
      <td>1.662651</td>
      <td>2.190932</td>
      <td>2.228158</td>
      <td>1.426332</td>
    </tr>
    <tr>
      <th>650</th>
      <td>4106</td>
      <td>3668</td>
      <td>130</td>
      <td>3.175573</td>
      <td>0.061268</td>
      <td>0.000000</td>
      <td>3270.108195</td>
      <td>0.088558</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>3</td>
      <td>6.165648</td>
      <td>0.553022</td>
      <td>-5.743413</td>
      <td>-2.234864</td>
      <td>-16.498842</td>
      <td>7.867602</td>
      <td>4.961850</td>
    </tr>
    <tr>
      <th>651</th>
      <td>4171</td>
      <td>3753</td>
      <td>48</td>
      <td>3.244898</td>
      <td>0.116279</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055028</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8.711044</td>
      <td>-1.278561</td>
      <td>4.485927</td>
      <td>-2.970254</td>
      <td>1.204063</td>
      <td>1.737618</td>
      <td>-1.536139</td>
    </tr>
  </tbody>
</table>
<p>652 rows × 23 columns</p>
</div>



### Running K-Means clustering

Running K-Means clustering for the summarized data.


```python
kopt, ClustersData = OptClust(pca_data,
                              ['Prin'+str(i) for i in range(1,n_components+1)], minK=6)
```

    Optimal number of clusters: 7



```python
ClustersData
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>population</th>
      <th>count_intersections</th>
      <th>streets_per_node_avg</th>
      <th>betweenness_centrality_avg</th>
      <th>primary_length_total_ext</th>
      <th>highway_length_total_ext</th>
      <th>fraction_oneway_ext</th>
      <th>N_Est_Mfg</th>
      <th>N_Est_BFA</th>
      <th>N_Est_RW</th>
      <th>N_Emp_Mfg</th>
      <th>N_Emp_BFA</th>
      <th>N_Emp_RW</th>
      <th>N_Emp_Tot</th>
      <th>POI_Count</th>
      <th>Prin1</th>
      <th>Prin2</th>
      <th>Prin3</th>
      <th>Prin4</th>
      <th>Prin5</th>
      <th>Prin6</th>
      <th>Prin7</th>
      <th>kopt+0_Clusters</th>
      <th>kopt+1_Clusters</th>
      <th>kopt+2_Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>117</td>
      <td>1461</td>
      <td>114</td>
      <td>3.041322</td>
      <td>0.059484</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.164404</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>38</td>
      <td>7.114727</td>
      <td>1.648241</td>
      <td>-0.716472</td>
      <td>-1.662589</td>
      <td>1.690011</td>
      <td>1.488966</td>
      <td>0.175915</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>248</td>
      <td>3558</td>
      <td>57</td>
      <td>3.030769</td>
      <td>0.096963</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>48</td>
      <td>8.239472</td>
      <td>-0.497781</td>
      <td>4.218135</td>
      <td>-3.305417</td>
      <td>1.012448</td>
      <td>0.217245</td>
      <td>-0.444929</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>312</td>
      <td>1768</td>
      <td>63</td>
      <td>3.260870</td>
      <td>0.076863</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.061655</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>8.235277</td>
      <td>0.127473</td>
      <td>0.727412</td>
      <td>-1.628162</td>
      <td>1.708633</td>
      <td>2.378518</td>
      <td>-1.185460</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>313</td>
      <td>6137</td>
      <td>96</td>
      <td>3.140000</td>
      <td>0.091784</td>
      <td>0.000000</td>
      <td>4060.184994</td>
      <td>0.226727</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>5.607381</td>
      <td>-0.857567</td>
      <td>-2.838239</td>
      <td>-5.324318</td>
      <td>-21.805404</td>
      <td>8.777023</td>
      <td>5.298749</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378</td>
      <td>8452</td>
      <td>145</td>
      <td>3.278912</td>
      <td>0.060882</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.184630</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>5.813062</td>
      <td>4.105240</td>
      <td>-2.443006</td>
      <td>-1.055450</td>
      <td>2.258324</td>
      <td>2.950110</td>
      <td>-1.207996</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>443</td>
      <td>1416</td>
      <td>23</td>
      <td>3.521739</td>
      <td>0.131752</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.351102</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>7.220447</td>
      <td>-0.976269</td>
      <td>3.855626</td>
      <td>-7.584883</td>
      <td>1.314609</td>
      <td>2.985858</td>
      <td>-5.452076</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>444</td>
      <td>1963</td>
      <td>30</td>
      <td>3.027778</td>
      <td>0.118371</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039808</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.778128</td>
      <td>-2.953580</td>
      <td>6.690823</td>
      <td>-2.168602</td>
      <td>0.573109</td>
      <td>0.846566</td>
      <td>0.330949</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>508</td>
      <td>10039</td>
      <td>66</td>
      <td>3.185714</td>
      <td>0.084119</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.452238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35</td>
      <td>5.160865</td>
      <td>2.633752</td>
      <td>2.603606</td>
      <td>-5.491300</td>
      <td>0.870054</td>
      <td>2.333111</td>
      <td>-2.635874</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>574</td>
      <td>4149</td>
      <td>9</td>
      <td>2.200000</td>
      <td>0.186813</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>13.047463</td>
      <td>-8.727022</td>
      <td>19.133299</td>
      <td>-0.429795</td>
      <td>-2.138990</td>
      <td>-3.293593</td>
      <td>7.022550</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>575</td>
      <td>1870</td>
      <td>42</td>
      <td>2.722222</td>
      <td>0.109391</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>10.640185</td>
      <td>-3.735674</td>
      <td>8.036214</td>
      <td>0.167269</td>
      <td>0.042020</td>
      <td>-0.175620</td>
      <td>3.443778</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>635</td>
      <td>1016</td>
      <td>21</td>
      <td>3.095238</td>
      <td>0.152256</td>
      <td>0.000000</td>
      <td>1626.015841</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>9.618602</td>
      <td>-4.938202</td>
      <td>6.757275</td>
      <td>-5.175873</td>
      <td>-8.770859</td>
      <td>3.003045</td>
      <td>1.565444</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>638</td>
      <td>1563</td>
      <td>20</td>
      <td>3.150000</td>
      <td>0.158772</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>9.987605</td>
      <td>-3.979870</td>
      <td>8.896803</td>
      <td>-4.784795</td>
      <td>0.543647</td>
      <td>0.296633</td>
      <td>-1.574255</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>639</td>
      <td>2623</td>
      <td>19</td>
      <td>3.315789</td>
      <td>0.141899</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027211</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
      <td>9.021302</td>
      <td>-2.448525</td>
      <td>6.530937</td>
      <td>-5.286253</td>
      <td>1.003418</td>
      <td>1.193177</td>
      <td>-3.026523</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>640</td>
      <td>2120</td>
      <td>12</td>
      <td>2.562500</td>
      <td>0.197619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>12.386180</td>
      <td>-7.986901</td>
      <td>16.713704</td>
      <td>-2.656744</td>
      <td>-1.253468</td>
      <td>-2.182350</td>
      <td>3.612527</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>700</td>
      <td>3186</td>
      <td>129</td>
      <td>3.079710</td>
      <td>0.069031</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.140474</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>3</td>
      <td>6.554672</td>
      <td>0.423184</td>
      <td>-0.469490</td>
      <td>1.318962</td>
      <td>1.428247</td>
      <td>1.704762</td>
      <td>0.110696</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>701</td>
      <td>5925</td>
      <td>129</td>
      <td>3.147059</td>
      <td>0.054680</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.073444</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>2</td>
      <td>6.195126</td>
      <td>1.526745</td>
      <td>-1.642301</td>
      <td>2.191782</td>
      <td>1.686763</td>
      <td>2.129217</td>
      <td>-0.032069</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>703</td>
      <td>2573</td>
      <td>131</td>
      <td>3.131034</td>
      <td>0.050211</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>8.067006</td>
      <td>1.778042</td>
      <td>-2.265129</td>
      <td>1.791666</td>
      <td>2.209533</td>
      <td>2.416521</td>
      <td>1.185972</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>704</td>
      <td>2642</td>
      <td>25</td>
      <td>3.111111</td>
      <td>0.140171</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.963691</td>
      <td>-3.449446</td>
      <td>7.893462</td>
      <td>-3.098225</td>
      <td>0.595067</td>
      <td>0.742412</td>
      <td>-0.570550</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>705</td>
      <td>3445</td>
      <td>20</td>
      <td>3.190476</td>
      <td>0.134336</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>9.580116</td>
      <td>-2.850207</td>
      <td>7.114061</td>
      <td>-3.396500</td>
      <td>0.764975</td>
      <td>1.104674</td>
      <td>-1.288542</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>766</td>
      <td>5774</td>
      <td>191</td>
      <td>3.270833</td>
      <td>0.050461</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.265891</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>4</td>
      <td>4.560264</td>
      <td>3.701022</td>
      <td>-5.003858</td>
      <td>1.474342</td>
      <td>2.357564</td>
      <td>3.171156</td>
      <td>-1.180550</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>768</td>
      <td>7648</td>
      <td>181</td>
      <td>3.316940</td>
      <td>0.047453</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.180476</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>5</td>
      <td>4.600562</td>
      <td>3.987507</td>
      <td>-4.984812</td>
      <td>1.823156</td>
      <td>2.439783</td>
      <td>3.211944</td>
      <td>-1.376860</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>829</td>
      <td>2298</td>
      <td>75</td>
      <td>3.129870</td>
      <td>0.089241</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057823</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
      <td>7.511895</td>
      <td>-1.458355</td>
      <td>2.270921</td>
      <td>-0.003671</td>
      <td>1.049518</td>
      <td>1.291518</td>
      <td>-0.762275</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>830</td>
      <td>9489</td>
      <td>307</td>
      <td>3.319355</td>
      <td>0.036052</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.263815</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>9</td>
      <td>3.090643</td>
      <td>7.238608</td>
      <td>-9.691996</td>
      <td>4.037817</td>
      <td>3.470729</td>
      <td>3.908363</td>
      <td>-0.369153</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>832</td>
      <td>6887</td>
      <td>159</td>
      <td>3.257862</td>
      <td>0.053697</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.355541</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>11</td>
      <td>4.130940</td>
      <td>3.512732</td>
      <td>-3.544550</td>
      <td>-0.191141</td>
      <td>1.906533</td>
      <td>3.033940</td>
      <td>-1.909831</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>833</td>
      <td>6820</td>
      <td>254</td>
      <td>3.205426</td>
      <td>0.037250</td>
      <td>798.744052</td>
      <td>0.000000</td>
      <td>0.180348</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>8</td>
      <td>4.004272</td>
      <td>5.550877</td>
      <td>-7.875336</td>
      <td>3.435550</td>
      <td>2.658737</td>
      <td>0.447846</td>
      <td>1.662655</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>834</td>
      <td>7112</td>
      <td>199</td>
      <td>3.170616</td>
      <td>0.044921</td>
      <td>677.380694</td>
      <td>0.000000</td>
      <td>0.078081</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>17</td>
      <td>4.797122</td>
      <td>4.123703</td>
      <td>-5.165559</td>
      <td>2.551414</td>
      <td>2.180125</td>
      <td>0.046509</td>
      <td>1.205930</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>835</td>
      <td>3161</td>
      <td>64</td>
      <td>3.057143</td>
      <td>0.113019</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055204</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.237356</td>
      <td>-1.793919</td>
      <td>5.133964</td>
      <td>-1.477011</td>
      <td>0.924772</td>
      <td>1.201720</td>
      <td>0.439545</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27</th>
      <td>840</td>
      <td>1000</td>
      <td>26</td>
      <td>3.000000</td>
      <td>0.184672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.085998</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>10.651390</td>
      <td>-5.560985</td>
      <td>11.721956</td>
      <td>-4.776007</td>
      <td>-0.031810</td>
      <td>-0.207172</td>
      <td>-0.365805</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>894</td>
      <td>1423</td>
      <td>32</td>
      <td>3.000000</td>
      <td>0.143672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
      <td>9.186110</td>
      <td>-4.951267</td>
      <td>8.535443</td>
      <td>-1.737543</td>
      <td>0.052699</td>
      <td>-0.278792</td>
      <td>-0.451314</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>897</td>
      <td>8544</td>
      <td>290</td>
      <td>3.250000</td>
      <td>0.036473</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.396344</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>10</td>
      <td>2.867305</td>
      <td>6.785169</td>
      <td>-8.743875</td>
      <td>2.954167</td>
      <td>3.045589</td>
      <td>3.815264</td>
      <td>-0.402792</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>622</th>
      <td>3653</td>
      <td>2246</td>
      <td>62</td>
      <td>3.014925</td>
      <td>0.077842</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.152148</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
      <td>7.708224</td>
      <td>-1.296204</td>
      <td>2.760071</td>
      <td>-0.242940</td>
      <td>0.749922</td>
      <td>1.362489</td>
      <td>0.236301</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>623</th>
      <td>3654</td>
      <td>1136</td>
      <td>51</td>
      <td>2.948276</td>
      <td>0.094531</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.286316</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
      <td>7.692212</td>
      <td>-2.257399</td>
      <td>4.660716</td>
      <td>-1.740929</td>
      <td>0.266960</td>
      <td>1.071139</td>
      <td>0.080019</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>624</th>
      <td>3656</td>
      <td>1393</td>
      <td>28</td>
      <td>3.107143</td>
      <td>0.117623</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.310952</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>8.275209</td>
      <td>-2.243143</td>
      <td>5.921565</td>
      <td>-4.488768</td>
      <td>0.475513</td>
      <td>1.631573</td>
      <td>-1.358258</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>625</th>
      <td>3657</td>
      <td>1177</td>
      <td>80</td>
      <td>3.219512</td>
      <td>0.059982</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.048138</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>8.130549</td>
      <td>0.562405</td>
      <td>-0.786001</td>
      <td>-0.148490</td>
      <td>1.888048</td>
      <td>2.543923</td>
      <td>-0.353553</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>626</th>
      <td>3709</td>
      <td>4097</td>
      <td>53</td>
      <td>3.254545</td>
      <td>0.082574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>5.126004</td>
      <td>-2.403826</td>
      <td>1.641583</td>
      <td>1.050660</td>
      <td>0.790019</td>
      <td>0.967430</td>
      <td>-3.092140</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>627</th>
      <td>3710</td>
      <td>10769</td>
      <td>190</td>
      <td>3.325000</td>
      <td>0.045522</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.950692</td>
      <td>2.827299</td>
      <td>-5.115716</td>
      <td>5.166793</td>
      <td>2.214348</td>
      <td>2.384739</td>
      <td>-1.808210</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>628</th>
      <td>3711</td>
      <td>10013</td>
      <td>258</td>
      <td>3.585271</td>
      <td>0.037401</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>2</td>
      <td>1.787539</td>
      <td>5.261248</td>
      <td>-10.048357</td>
      <td>5.371859</td>
      <td>3.509995</td>
      <td>3.600469</td>
      <td>-3.467707</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>629</th>
      <td>3712</td>
      <td>6538</td>
      <td>222</td>
      <td>3.646288</td>
      <td>0.042664</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.313567</td>
      <td>4.006627</td>
      <td>-9.445107</td>
      <td>4.055533</td>
      <td>3.410307</td>
      <td>3.582967</td>
      <td>-4.447516</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>630</th>
      <td>3713</td>
      <td>18333</td>
      <td>215</td>
      <td>3.305085</td>
      <td>0.047301</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.048151</td>
      <td>4.494314</td>
      <td>-4.603223</td>
      <td>6.193876</td>
      <td>2.151706</td>
      <td>2.564550</td>
      <td>-1.243486</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>631</th>
      <td>3714</td>
      <td>9502</td>
      <td>183</td>
      <td>3.447368</td>
      <td>0.045289</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039784</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.604941</td>
      <td>3.053710</td>
      <td>-6.047884</td>
      <td>4.063645</td>
      <td>2.449006</td>
      <td>2.858231</td>
      <td>-3.139127</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>632</th>
      <td>3715</td>
      <td>3061</td>
      <td>154</td>
      <td>3.267516</td>
      <td>0.036386</td>
      <td>0.000000</td>
      <td>1845.681177</td>
      <td>0.082943</td>
      <td>171.0</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>30970.0</td>
      <td>0.0</td>
      <td>5443.0</td>
      <td>110070.0</td>
      <td>2</td>
      <td>1.348418</td>
      <td>-0.676523</td>
      <td>-7.671264</td>
      <td>3.293536</td>
      <td>-8.973567</td>
      <td>4.379221</td>
      <td>-0.594836</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>633</th>
      <td>3717</td>
      <td>1963</td>
      <td>69</td>
      <td>3.281690</td>
      <td>0.071513</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>5304.0</td>
      <td>0.0</td>
      <td>520.0</td>
      <td>9091.0</td>
      <td>2</td>
      <td>7.635872</td>
      <td>-0.148072</td>
      <td>-0.013822</td>
      <td>-0.308361</td>
      <td>1.686361</td>
      <td>2.132816</td>
      <td>-1.547070</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>634</th>
      <td>3719</td>
      <td>1644</td>
      <td>88</td>
      <td>3.217391</td>
      <td>0.064075</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080264</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>3</td>
      <td>7.146136</td>
      <td>0.256668</td>
      <td>-0.740441</td>
      <td>0.060128</td>
      <td>1.655740</td>
      <td>2.163562</td>
      <td>-1.050646</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>635</th>
      <td>3721</td>
      <td>2824</td>
      <td>32</td>
      <td>2.815789</td>
      <td>0.113008</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>1</td>
      <td>10.232179</td>
      <td>-3.556909</td>
      <td>8.014171</td>
      <td>-0.549979</td>
      <td>0.111697</td>
      <td>0.047302</td>
      <td>2.388190</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>636</th>
      <td>3723</td>
      <td>5009</td>
      <td>12</td>
      <td>2.857143</td>
      <td>0.226190</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.127762</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>10.918062</td>
      <td>-7.114726</td>
      <td>16.848065</td>
      <td>-5.752593</td>
      <td>-0.976964</td>
      <td>-1.246141</td>
      <td>0.211838</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>637</th>
      <td>3726</td>
      <td>1175</td>
      <td>14</td>
      <td>2.764706</td>
      <td>0.144608</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>11.095935</td>
      <td>-5.519466</td>
      <td>11.075864</td>
      <td>-1.792307</td>
      <td>-0.350198</td>
      <td>-0.696229</td>
      <td>2.316277</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>638</th>
      <td>3775</td>
      <td>1218</td>
      <td>22</td>
      <td>2.838710</td>
      <td>0.146830</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>7.627340</td>
      <td>-7.402113</td>
      <td>10.178699</td>
      <td>0.089758</td>
      <td>-0.872388</td>
      <td>-1.695192</td>
      <td>-0.414001</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>639</th>
      <td>3776</td>
      <td>1421</td>
      <td>89</td>
      <td>3.000000</td>
      <td>0.067887</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>5.748206</td>
      <td>-2.735252</td>
      <td>1.033739</td>
      <td>3.364322</td>
      <td>0.663051</td>
      <td>0.301562</td>
      <td>-0.319776</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>640</th>
      <td>3779</td>
      <td>6319</td>
      <td>180</td>
      <td>3.063725</td>
      <td>0.042501</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>4.194172</td>
      <td>0.918488</td>
      <td>-3.579627</td>
      <td>6.071268</td>
      <td>1.642519</td>
      <td>1.328820</td>
      <td>0.397949</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>641</th>
      <td>3780</td>
      <td>1442</td>
      <td>156</td>
      <td>2.994286</td>
      <td>0.044316</td>
      <td>0.000000</td>
      <td>1242.905239</td>
      <td>0.010531</td>
      <td>142.0</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>25666.0</td>
      <td>0.0</td>
      <td>4923.0</td>
      <td>100979.0</td>
      <td>0</td>
      <td>3.760160</td>
      <td>-1.734009</td>
      <td>-4.407463</td>
      <td>4.925109</td>
      <td>-5.908830</td>
      <td>2.506965</td>
      <td>1.798920</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>642</th>
      <td>3785</td>
      <td>1731</td>
      <td>13</td>
      <td>2.800000</td>
      <td>0.139927</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>10.878596</td>
      <td>-5.130930</td>
      <td>10.567279</td>
      <td>-1.790558</td>
      <td>-0.254679</td>
      <td>-0.484708</td>
      <td>2.050301</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>643</th>
      <td>3788</td>
      <td>2032</td>
      <td>50</td>
      <td>2.962963</td>
      <td>0.108047</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>9.767519</td>
      <td>-2.642534</td>
      <td>5.818971</td>
      <td>-0.735702</td>
      <td>0.681995</td>
      <td>0.714320</td>
      <td>1.307652</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>644</th>
      <td>3844</td>
      <td>2363</td>
      <td>13</td>
      <td>2.687500</td>
      <td>0.152381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>8.044514</td>
      <td>-8.151473</td>
      <td>12.211521</td>
      <td>0.536849</td>
      <td>-1.389954</td>
      <td>-2.327158</td>
      <td>0.825027</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>645</th>
      <td>3845</td>
      <td>2234</td>
      <td>154</td>
      <td>3.024096</td>
      <td>0.046908</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>4.961208</td>
      <td>-0.593878</td>
      <td>-2.668487</td>
      <td>5.328771</td>
      <td>1.407773</td>
      <td>0.928428</td>
      <td>0.363715</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>646</th>
      <td>3907</td>
      <td>1506</td>
      <td>41</td>
      <td>3.404762</td>
      <td>0.089983</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>2</td>
      <td>8.475786</td>
      <td>-0.383027</td>
      <td>1.260831</td>
      <td>-2.691461</td>
      <td>1.844396</td>
      <td>2.549436</td>
      <td>-2.593849</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>647</th>
      <td>4040</td>
      <td>4533</td>
      <td>72</td>
      <td>3.112500</td>
      <td>0.079286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.096302</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>0</td>
      <td>8.253958</td>
      <td>0.100771</td>
      <td>2.161373</td>
      <td>-0.639658</td>
      <td>1.292977</td>
      <td>2.059206</td>
      <td>0.286690</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>648</th>
      <td>4041</td>
      <td>1227</td>
      <td>54</td>
      <td>3.185185</td>
      <td>0.103847</td>
      <td>0.000000</td>
      <td>1542.810577</td>
      <td>0.186673</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>7</td>
      <td>7.608644</td>
      <td>-1.693240</td>
      <td>1.516269</td>
      <td>-4.832180</td>
      <td>-7.638683</td>
      <td>4.304078</td>
      <td>0.740304</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>649</th>
      <td>4105</td>
      <td>6060</td>
      <td>165</td>
      <td>3.078652</td>
      <td>0.052909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.064199</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>7.102299</td>
      <td>3.127608</td>
      <td>-2.364986</td>
      <td>1.662651</td>
      <td>2.190932</td>
      <td>2.228158</td>
      <td>1.426332</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>650</th>
      <td>4106</td>
      <td>3668</td>
      <td>130</td>
      <td>3.175573</td>
      <td>0.061268</td>
      <td>0.000000</td>
      <td>3270.108195</td>
      <td>0.088558</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>3</td>
      <td>6.165648</td>
      <td>0.553022</td>
      <td>-5.743413</td>
      <td>-2.234864</td>
      <td>-16.498842</td>
      <td>7.867602</td>
      <td>4.961850</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>651</th>
      <td>4171</td>
      <td>3753</td>
      <td>48</td>
      <td>3.244898</td>
      <td>0.116279</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055028</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8.711044</td>
      <td>-1.278561</td>
      <td>4.485927</td>
      <td>-2.970254</td>
      <td>1.204063</td>
      <td>1.737618</td>
      <td>-1.536139</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>652 rows × 26 columns</p>
</div>



Calculating centroids using the mean of each numeric component. Input those columns containing the number of clusters to be plotted and those columns containing data that does not want to be considered, i.e.: the PCA's columns.


```python
centroids = calculate_centroids(ClustersData,
                                ['kopt+0_Clusters'],
                                ['index']+['Prin'+str(j) for j in range(1,n_components+1)])
```


```python
centroids
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_clusters</th>
      <th>cluster_id</th>
      <th>primary_length_total_ext</th>
      <th>kopt+2_Clusters</th>
      <th>highway_length_total_ext</th>
      <th>POI_Count</th>
      <th>N_Est_RW</th>
      <th>streets_per_node_avg</th>
      <th>N_Est_Mfg</th>
      <th>N_Emp_Mfg</th>
      <th>fraction_oneway_ext</th>
      <th>N_Emp_RW</th>
      <th>betweenness_centrality_avg</th>
      <th>N_Emp_Tot</th>
      <th>N_Est_BFA</th>
      <th>population</th>
      <th>N_Emp_BFA</th>
      <th>kopt+1_Clusters</th>
      <th>count_intersections</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>0</td>
      <td>72.384413</td>
      <td>3.421965</td>
      <td>80.214265</td>
      <td>2.884393</td>
      <td>22.520231</td>
      <td>2.930038</td>
      <td>37.710983</td>
      <td>10496.184971</td>
      <td>0.079690</td>
      <td>2442.890173</td>
      <td>0.110475</td>
      <td>36039.913295</td>
      <td>1.086705</td>
      <td>6025.190751</td>
      <td>207.369942</td>
      <td>3.473988</td>
      <td>47.502890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
      <td>2379.493413</td>
      <td>1.255814</td>
      <td>50.316451</td>
      <td>42.232558</td>
      <td>52.476744</td>
      <td>3.214088</td>
      <td>68.930233</td>
      <td>15295.825581</td>
      <td>0.305096</td>
      <td>4045.313953</td>
      <td>0.053492</td>
      <td>66767.767442</td>
      <td>3.639535</td>
      <td>17861.337209</td>
      <td>456.837209</td>
      <td>0.069767</td>
      <td>170.953488</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2</td>
      <td>174.133214</td>
      <td>4.325688</td>
      <td>29.193521</td>
      <td>13.188073</td>
      <td>26.050459</td>
      <td>3.184827</td>
      <td>52.050459</td>
      <td>11768.009174</td>
      <td>0.170794</td>
      <td>2292.027523</td>
      <td>0.054641</td>
      <td>43953.059633</td>
      <td>0.903670</td>
      <td>11977.449541</td>
      <td>94.724771</td>
      <td>2.041284</td>
      <td>160.183486</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>3</td>
      <td>1478.201418</td>
      <td>6.481481</td>
      <td>656.960619</td>
      <td>56.111111</td>
      <td>538.444444</td>
      <td>3.287951</td>
      <td>309.148148</td>
      <td>104430.296296</td>
      <td>0.658304</td>
      <td>61729.407407</td>
      <td>0.062349</td>
      <td>688509.185185</td>
      <td>84.666667</td>
      <td>20480.814815</td>
      <td>25091.629630</td>
      <td>3.000000</td>
      <td>140.592593</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>4</td>
      <td>515.677294</td>
      <td>3.000000</td>
      <td>125.760847</td>
      <td>20.595506</td>
      <td>227.808989</td>
      <td>3.101048</td>
      <td>233.348315</td>
      <td>69562.876404</td>
      <td>0.269504</td>
      <td>29293.033708</td>
      <td>0.075701</td>
      <td>246162.067416</td>
      <td>15.067416</td>
      <td>13001.213483</td>
      <td>3032.089888</td>
      <td>7.000000</td>
      <td>105.370787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>5</td>
      <td>1243.984487</td>
      <td>6.000000</td>
      <td>382.815460</td>
      <td>142.750000</td>
      <td>241.500000</td>
      <td>3.518673</td>
      <td>168.375000</td>
      <td>43309.625000</td>
      <td>0.884708</td>
      <td>19906.750000</td>
      <td>0.062476</td>
      <td>273312.375000</td>
      <td>33.250000</td>
      <td>26601.583333</td>
      <td>7383.333333</td>
      <td>4.000000</td>
      <td>123.083333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>6</td>
      <td>757.766162</td>
      <td>6.942857</td>
      <td>2445.937148</td>
      <td>27.400000</td>
      <td>91.714286</td>
      <td>3.168427</td>
      <td>108.857143</td>
      <td>27719.800000</td>
      <td>0.330609</td>
      <td>9671.457143</td>
      <td>0.057888</td>
      <td>114952.628571</td>
      <td>7.942857</td>
      <td>13131.628571</td>
      <td>1410.771429</td>
      <td>5.971429</td>
      <td>135.000000</td>
    </tr>
  </tbody>
</table>
</div>



Plotting the results.


```python
radar_chart(centroids, 'number_of_clusters', 'cluster_id',
            ['population','count_intersections',
             'N_Est_BFA','N_Est_RW','N_Est_Mfg'],
            ['Population','Intersection\nDensity',
             'Food\nService\n','Retail','Manufacturing'], city_name)
```

    Cluster 0
    Cluster 1
    Cluster 2
    Cluster 3
    Cluster 4
    Cluster 5
    Cluster 6



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_64_1.png)



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_64_2.png)



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_64_3.png)



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_64_4.png)



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_64_5.png)



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_64_6.png)



![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_64_7.png)


Adding the data frame the information regarding geo-coordinates.


```python
pixels_data_filtered['index'] = pixels_data_filtered.index.tolist()
ClustersData = pd.merge(ClustersData, pixels_data_filtered[['index','lat','lon','utm_n','utm_e']],
                        how='left')
```


```python
ClustersData
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>population</th>
      <th>count_intersections</th>
      <th>streets_per_node_avg</th>
      <th>betweenness_centrality_avg</th>
      <th>primary_length_total_ext</th>
      <th>highway_length_total_ext</th>
      <th>fraction_oneway_ext</th>
      <th>N_Est_Mfg</th>
      <th>N_Est_BFA</th>
      <th>N_Est_RW</th>
      <th>N_Emp_Mfg</th>
      <th>N_Emp_BFA</th>
      <th>N_Emp_RW</th>
      <th>N_Emp_Tot</th>
      <th>POI_Count</th>
      <th>Prin1</th>
      <th>Prin2</th>
      <th>Prin3</th>
      <th>Prin4</th>
      <th>Prin5</th>
      <th>Prin6</th>
      <th>Prin7</th>
      <th>kopt+0_Clusters</th>
      <th>kopt+1_Clusters</th>
      <th>kopt+2_Clusters</th>
      <th>lat</th>
      <th>lon</th>
      <th>utm_n</th>
      <th>utm_e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>117</td>
      <td>1461</td>
      <td>114</td>
      <td>3.041322</td>
      <td>0.059484</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.164404</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>38</td>
      <td>7.114727</td>
      <td>1.648241</td>
      <td>-0.716472</td>
      <td>-1.662589</td>
      <td>1.690011</td>
      <td>1.488966</td>
      <td>0.175915</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>-12.330026</td>
      <td>-76.826524</td>
      <td>8636276.063</td>
      <td>301380.8033</td>
    </tr>
    <tr>
      <th>1</th>
      <td>248</td>
      <td>3558</td>
      <td>57</td>
      <td>3.030769</td>
      <td>0.096963</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>332.0</td>
      <td>48</td>
      <td>8.239472</td>
      <td>-0.497781</td>
      <td>4.218135</td>
      <td>-3.305417</td>
      <td>1.012448</td>
      <td>0.217245</td>
      <td>-0.444929</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-12.311887</td>
      <td>-76.835591</td>
      <td>8638276.063</td>
      <td>300380.8033</td>
    </tr>
    <tr>
      <th>2</th>
      <td>312</td>
      <td>1768</td>
      <td>63</td>
      <td>3.260870</td>
      <td>0.076863</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.061655</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>8.235277</td>
      <td>0.127473</td>
      <td>0.727412</td>
      <td>-1.628162</td>
      <td>1.708633</td>
      <td>2.378518</td>
      <td>-1.185460</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>-12.302725</td>
      <td>-76.853912</td>
      <td>8639276.063</td>
      <td>298380.8033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>313</td>
      <td>6137</td>
      <td>96</td>
      <td>3.140000</td>
      <td>0.091784</td>
      <td>0.000000</td>
      <td>4060.184994</td>
      <td>0.226727</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>5.607381</td>
      <td>-0.857567</td>
      <td>-2.838239</td>
      <td>-5.324318</td>
      <td>-21.805404</td>
      <td>8.777023</td>
      <td>5.298749</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
      <td>-12.302787</td>
      <td>-76.844720</td>
      <td>8639276.063</td>
      <td>299380.8033</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378</td>
      <td>8452</td>
      <td>145</td>
      <td>3.278912</td>
      <td>0.060882</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.184630</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>5.813062</td>
      <td>4.105240</td>
      <td>-2.443006</td>
      <td>-1.055450</td>
      <td>2.258324</td>
      <td>2.950110</td>
      <td>-1.207996</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.293686</td>
      <td>-76.853848</td>
      <td>8640276.063</td>
      <td>298380.8033</td>
    </tr>
    <tr>
      <th>5</th>
      <td>443</td>
      <td>1416</td>
      <td>23</td>
      <td>3.521739</td>
      <td>0.131752</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.351102</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>7.220447</td>
      <td>-0.976269</td>
      <td>3.855626</td>
      <td>-7.584883</td>
      <td>1.314609</td>
      <td>2.985858</td>
      <td>-5.452076</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-12.284586</td>
      <td>-76.862976</td>
      <td>8641276.063</td>
      <td>297380.8033</td>
    </tr>
    <tr>
      <th>6</th>
      <td>444</td>
      <td>1963</td>
      <td>30</td>
      <td>3.027778</td>
      <td>0.118371</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039808</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.778128</td>
      <td>-2.953580</td>
      <td>6.690823</td>
      <td>-2.168602</td>
      <td>0.573109</td>
      <td>0.846566</td>
      <td>0.330949</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.284648</td>
      <td>-76.853785</td>
      <td>8641276.063</td>
      <td>298380.8033</td>
    </tr>
    <tr>
      <th>7</th>
      <td>508</td>
      <td>10039</td>
      <td>66</td>
      <td>3.185714</td>
      <td>0.084119</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.452238</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35</td>
      <td>5.160865</td>
      <td>2.633752</td>
      <td>2.603606</td>
      <td>-5.491300</td>
      <td>0.870054</td>
      <td>2.333111</td>
      <td>-2.635874</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>-12.275485</td>
      <td>-76.872103</td>
      <td>8642276.063</td>
      <td>296380.8033</td>
    </tr>
    <tr>
      <th>8</th>
      <td>574</td>
      <td>4149</td>
      <td>9</td>
      <td>2.200000</td>
      <td>0.186813</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>13.047463</td>
      <td>-8.727022</td>
      <td>19.133299</td>
      <td>-0.429795</td>
      <td>-2.138990</td>
      <td>-3.293593</td>
      <td>7.022550</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.266446</td>
      <td>-76.872040</td>
      <td>8643276.063</td>
      <td>296380.8033</td>
    </tr>
    <tr>
      <th>9</th>
      <td>575</td>
      <td>1870</td>
      <td>42</td>
      <td>2.722222</td>
      <td>0.109391</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>10.640185</td>
      <td>-3.735674</td>
      <td>8.036214</td>
      <td>0.167269</td>
      <td>0.042020</td>
      <td>-0.175620</td>
      <td>3.443778</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.266509</td>
      <td>-76.862849</td>
      <td>8643276.063</td>
      <td>297380.8033</td>
    </tr>
    <tr>
      <th>10</th>
      <td>635</td>
      <td>1016</td>
      <td>21</td>
      <td>3.095238</td>
      <td>0.152256</td>
      <td>0.000000</td>
      <td>1626.015841</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>9.618602</td>
      <td>-4.938202</td>
      <td>6.757275</td>
      <td>-5.175873</td>
      <td>-8.770859</td>
      <td>3.003045</td>
      <td>1.565444</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.257090</td>
      <td>-76.917926</td>
      <td>8644276.063</td>
      <td>291380.8033</td>
    </tr>
    <tr>
      <th>11</th>
      <td>638</td>
      <td>1563</td>
      <td>20</td>
      <td>3.150000</td>
      <td>0.158772</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>9.987605</td>
      <td>-3.979870</td>
      <td>8.896803</td>
      <td>-4.784795</td>
      <td>0.543647</td>
      <td>0.296633</td>
      <td>-1.574255</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.257282</td>
      <td>-76.890356</td>
      <td>8644276.063</td>
      <td>294380.8033</td>
    </tr>
    <tr>
      <th>12</th>
      <td>639</td>
      <td>2623</td>
      <td>19</td>
      <td>3.315789</td>
      <td>0.141899</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027211</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
      <td>9.021302</td>
      <td>-2.448525</td>
      <td>6.530937</td>
      <td>-5.286253</td>
      <td>1.003418</td>
      <td>1.193177</td>
      <td>-3.026523</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.257345</td>
      <td>-76.881166</td>
      <td>8644276.063</td>
      <td>295380.8033</td>
    </tr>
    <tr>
      <th>13</th>
      <td>640</td>
      <td>2120</td>
      <td>12</td>
      <td>2.562500</td>
      <td>0.197619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>12.386180</td>
      <td>-7.986901</td>
      <td>16.713704</td>
      <td>-2.656744</td>
      <td>-1.253468</td>
      <td>-2.182350</td>
      <td>3.612527</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.257408</td>
      <td>-76.871976</td>
      <td>8644276.063</td>
      <td>296380.8033</td>
    </tr>
    <tr>
      <th>14</th>
      <td>700</td>
      <td>3186</td>
      <td>129</td>
      <td>3.079710</td>
      <td>0.069031</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.140474</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>3</td>
      <td>6.554672</td>
      <td>0.423184</td>
      <td>-0.469490</td>
      <td>1.318962</td>
      <td>1.428247</td>
      <td>1.704762</td>
      <td>0.110696</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>-12.247988</td>
      <td>-76.927050</td>
      <td>8645276.063</td>
      <td>290380.8033</td>
    </tr>
    <tr>
      <th>15</th>
      <td>701</td>
      <td>5925</td>
      <td>129</td>
      <td>3.147059</td>
      <td>0.054680</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.073444</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>2</td>
      <td>6.195126</td>
      <td>1.526745</td>
      <td>-1.642301</td>
      <td>2.191782</td>
      <td>1.686763</td>
      <td>2.129217</td>
      <td>-0.032069</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>-12.248052</td>
      <td>-76.917860</td>
      <td>8645276.063</td>
      <td>291380.8033</td>
    </tr>
    <tr>
      <th>16</th>
      <td>703</td>
      <td>2573</td>
      <td>131</td>
      <td>3.131034</td>
      <td>0.050211</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>8.067006</td>
      <td>1.778042</td>
      <td>-2.265129</td>
      <td>1.791666</td>
      <td>2.209533</td>
      <td>2.416521</td>
      <td>1.185972</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>-12.248180</td>
      <td>-76.899481</td>
      <td>8645276.063</td>
      <td>293380.8033</td>
    </tr>
    <tr>
      <th>17</th>
      <td>704</td>
      <td>2642</td>
      <td>25</td>
      <td>3.111111</td>
      <td>0.140171</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.963691</td>
      <td>-3.449446</td>
      <td>7.893462</td>
      <td>-3.098225</td>
      <td>0.595067</td>
      <td>0.742412</td>
      <td>-0.570550</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.248244</td>
      <td>-76.890291</td>
      <td>8645276.063</td>
      <td>294380.8033</td>
    </tr>
    <tr>
      <th>18</th>
      <td>705</td>
      <td>3445</td>
      <td>20</td>
      <td>3.190476</td>
      <td>0.134336</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>9.580116</td>
      <td>-2.850207</td>
      <td>7.114061</td>
      <td>-3.396500</td>
      <td>0.764975</td>
      <td>1.104674</td>
      <td>-1.288542</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.248307</td>
      <td>-76.881102</td>
      <td>8645276.063</td>
      <td>295380.8033</td>
    </tr>
    <tr>
      <th>19</th>
      <td>766</td>
      <td>5774</td>
      <td>191</td>
      <td>3.270833</td>
      <td>0.050461</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.265891</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>4</td>
      <td>4.560264</td>
      <td>3.701022</td>
      <td>-5.003858</td>
      <td>1.474342</td>
      <td>2.357564</td>
      <td>3.171156</td>
      <td>-1.180550</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.238950</td>
      <td>-76.926984</td>
      <td>8646276.063</td>
      <td>290380.8033</td>
    </tr>
    <tr>
      <th>20</th>
      <td>768</td>
      <td>7648</td>
      <td>181</td>
      <td>3.316940</td>
      <td>0.047453</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.180476</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>5</td>
      <td>4.600562</td>
      <td>3.987507</td>
      <td>-4.984812</td>
      <td>1.823156</td>
      <td>2.439783</td>
      <td>3.211944</td>
      <td>-1.376860</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.239078</td>
      <td>-76.908606</td>
      <td>8646276.063</td>
      <td>292380.8033</td>
    </tr>
    <tr>
      <th>21</th>
      <td>829</td>
      <td>2298</td>
      <td>75</td>
      <td>3.129870</td>
      <td>0.089241</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057823</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
      <td>7.511895</td>
      <td>-1.458355</td>
      <td>2.270921</td>
      <td>-0.003671</td>
      <td>1.049518</td>
      <td>1.291518</td>
      <td>-0.762275</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-12.229717</td>
      <td>-76.954485</td>
      <td>8647276.063</td>
      <td>287380.8033</td>
    </tr>
    <tr>
      <th>22</th>
      <td>830</td>
      <td>9489</td>
      <td>307</td>
      <td>3.319355</td>
      <td>0.036052</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.263815</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>9</td>
      <td>3.090643</td>
      <td>7.238608</td>
      <td>-9.691996</td>
      <td>4.037817</td>
      <td>3.470729</td>
      <td>3.908363</td>
      <td>-0.369153</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.229782</td>
      <td>-76.945296</td>
      <td>8647276.063</td>
      <td>288380.8033</td>
    </tr>
    <tr>
      <th>23</th>
      <td>832</td>
      <td>6887</td>
      <td>159</td>
      <td>3.257862</td>
      <td>0.053697</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.355541</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>11</td>
      <td>4.130940</td>
      <td>3.512732</td>
      <td>-3.544550</td>
      <td>-0.191141</td>
      <td>1.906533</td>
      <td>3.033940</td>
      <td>-1.909831</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.229912</td>
      <td>-76.926919</td>
      <td>8647276.063</td>
      <td>290380.8033</td>
    </tr>
    <tr>
      <th>24</th>
      <td>833</td>
      <td>6820</td>
      <td>254</td>
      <td>3.205426</td>
      <td>0.037250</td>
      <td>798.744052</td>
      <td>0.000000</td>
      <td>0.180348</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>8</td>
      <td>4.004272</td>
      <td>5.550877</td>
      <td>-7.875336</td>
      <td>3.435550</td>
      <td>2.658737</td>
      <td>0.447846</td>
      <td>1.662655</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.229976</td>
      <td>-76.917730</td>
      <td>8647276.063</td>
      <td>291380.8033</td>
    </tr>
    <tr>
      <th>25</th>
      <td>834</td>
      <td>7112</td>
      <td>199</td>
      <td>3.170616</td>
      <td>0.044921</td>
      <td>677.380694</td>
      <td>0.000000</td>
      <td>0.078081</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>17</td>
      <td>4.797122</td>
      <td>4.123703</td>
      <td>-5.165559</td>
      <td>2.551414</td>
      <td>2.180125</td>
      <td>0.046509</td>
      <td>1.205930</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.230040</td>
      <td>-76.908541</td>
      <td>8647276.063</td>
      <td>292380.8033</td>
    </tr>
    <tr>
      <th>26</th>
      <td>835</td>
      <td>3161</td>
      <td>64</td>
      <td>3.057143</td>
      <td>0.113019</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055204</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9.237356</td>
      <td>-1.793919</td>
      <td>5.133964</td>
      <td>-1.477011</td>
      <td>0.924772</td>
      <td>1.201720</td>
      <td>0.439545</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-12.230104</td>
      <td>-76.899352</td>
      <td>8647276.063</td>
      <td>293380.8033</td>
    </tr>
    <tr>
      <th>27</th>
      <td>840</td>
      <td>1000</td>
      <td>26</td>
      <td>3.000000</td>
      <td>0.184672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.085998</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>10.651390</td>
      <td>-5.560985</td>
      <td>11.721956</td>
      <td>-4.776007</td>
      <td>-0.031810</td>
      <td>-0.207172</td>
      <td>-0.365805</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.230417</td>
      <td>-76.853406</td>
      <td>8647276.063</td>
      <td>298380.8033</td>
    </tr>
    <tr>
      <th>28</th>
      <td>894</td>
      <td>1423</td>
      <td>32</td>
      <td>3.000000</td>
      <td>0.143672</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>0</td>
      <td>9.186110</td>
      <td>-4.951267</td>
      <td>8.535443</td>
      <td>-1.737543</td>
      <td>0.052699</td>
      <td>-0.278792</td>
      <td>-0.451314</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-12.220614</td>
      <td>-76.963607</td>
      <td>8648276.063</td>
      <td>286380.8033</td>
    </tr>
    <tr>
      <th>29</th>
      <td>897</td>
      <td>8544</td>
      <td>290</td>
      <td>3.250000</td>
      <td>0.036473</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.396344</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>5748.0</td>
      <td>0.0</td>
      <td>1174.0</td>
      <td>17136.0</td>
      <td>10</td>
      <td>2.867305</td>
      <td>6.785169</td>
      <td>-8.743875</td>
      <td>2.954167</td>
      <td>3.045589</td>
      <td>3.815264</td>
      <td>-0.402792</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-12.220809</td>
      <td>-76.936042</td>
      <td>8648276.063</td>
      <td>289380.8033</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>622</th>
      <td>3653</td>
      <td>2246</td>
      <td>62</td>
      <td>3.014925</td>
      <td>0.077842</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.152148</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
      <td>7.708224</td>
      <td>-1.296204</td>
      <td>2.760071</td>
      <td>-0.242940</td>
      <td>0.749922</td>
      <td>1.362489</td>
      <td>0.236301</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-11.840166</td>
      <td>-77.080129</td>
      <td>8690276.063</td>
      <td>273380.8033</td>
    </tr>
    <tr>
      <th>623</th>
      <td>3654</td>
      <td>1136</td>
      <td>51</td>
      <td>2.948276</td>
      <td>0.094531</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.286316</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>0</td>
      <td>7.692212</td>
      <td>-2.257399</td>
      <td>4.660716</td>
      <td>-1.740929</td>
      <td>0.266960</td>
      <td>1.071139</td>
      <td>0.080019</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-11.840233</td>
      <td>-77.070955</td>
      <td>8690276.063</td>
      <td>274380.8033</td>
    </tr>
    <tr>
      <th>624</th>
      <td>3656</td>
      <td>1393</td>
      <td>28</td>
      <td>3.107143</td>
      <td>0.117623</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.310952</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>8.275209</td>
      <td>-2.243143</td>
      <td>5.921565</td>
      <td>-4.488768</td>
      <td>0.475513</td>
      <td>1.631573</td>
      <td>-1.358258</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>-11.840367</td>
      <td>-77.052605</td>
      <td>8690276.063</td>
      <td>276380.8033</td>
    </tr>
    <tr>
      <th>625</th>
      <td>3657</td>
      <td>1177</td>
      <td>80</td>
      <td>3.219512</td>
      <td>0.059982</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.048138</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>8.130549</td>
      <td>0.562405</td>
      <td>-0.786001</td>
      <td>-0.148490</td>
      <td>1.888048</td>
      <td>2.543923</td>
      <td>-0.353553</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>-11.840433</td>
      <td>-77.043430</td>
      <td>8690276.063</td>
      <td>277380.8033</td>
    </tr>
    <tr>
      <th>626</th>
      <td>3709</td>
      <td>4097</td>
      <td>53</td>
      <td>3.254545</td>
      <td>0.082574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>5.126004</td>
      <td>-2.403826</td>
      <td>1.641583</td>
      <td>1.050660</td>
      <td>0.790019</td>
      <td>0.967430</td>
      <td>-3.092140</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-11.830441</td>
      <td>-77.171803</td>
      <td>8691276.063</td>
      <td>263380.8033</td>
    </tr>
    <tr>
      <th>627</th>
      <td>3710</td>
      <td>10769</td>
      <td>190</td>
      <td>3.325000</td>
      <td>0.045522</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.950692</td>
      <td>2.827299</td>
      <td>-5.115716</td>
      <td>5.166793</td>
      <td>2.214348</td>
      <td>2.384739</td>
      <td>-1.808210</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.830511</td>
      <td>-77.162629</td>
      <td>8691276.063</td>
      <td>264380.8033</td>
    </tr>
    <tr>
      <th>628</th>
      <td>3711</td>
      <td>10013</td>
      <td>258</td>
      <td>3.585271</td>
      <td>0.037401</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>2</td>
      <td>1.787539</td>
      <td>5.261248</td>
      <td>-10.048357</td>
      <td>5.371859</td>
      <td>3.509995</td>
      <td>3.600469</td>
      <td>-3.467707</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.830581</td>
      <td>-77.153455</td>
      <td>8691276.063</td>
      <td>265380.8033</td>
    </tr>
    <tr>
      <th>629</th>
      <td>3712</td>
      <td>6538</td>
      <td>222</td>
      <td>3.646288</td>
      <td>0.042664</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.313567</td>
      <td>4.006627</td>
      <td>-9.445107</td>
      <td>4.055533</td>
      <td>3.410307</td>
      <td>3.582967</td>
      <td>-4.447516</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.830650</td>
      <td>-77.144281</td>
      <td>8691276.063</td>
      <td>266380.8033</td>
    </tr>
    <tr>
      <th>630</th>
      <td>3713</td>
      <td>18333</td>
      <td>215</td>
      <td>3.305085</td>
      <td>0.047301</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.048151</td>
      <td>4.494314</td>
      <td>-4.603223</td>
      <td>6.193876</td>
      <td>2.151706</td>
      <td>2.564550</td>
      <td>-1.243486</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.830719</td>
      <td>-77.135107</td>
      <td>8691276.063</td>
      <td>267380.8033</td>
    </tr>
    <tr>
      <th>631</th>
      <td>3714</td>
      <td>9502</td>
      <td>183</td>
      <td>3.447368</td>
      <td>0.045289</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039784</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>2.604941</td>
      <td>3.053710</td>
      <td>-6.047884</td>
      <td>4.063645</td>
      <td>2.449006</td>
      <td>2.858231</td>
      <td>-3.139127</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.830788</td>
      <td>-77.125933</td>
      <td>8691276.063</td>
      <td>268380.8033</td>
    </tr>
    <tr>
      <th>632</th>
      <td>3715</td>
      <td>3061</td>
      <td>154</td>
      <td>3.267516</td>
      <td>0.036386</td>
      <td>0.000000</td>
      <td>1845.681177</td>
      <td>0.082943</td>
      <td>171.0</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>30970.0</td>
      <td>0.0</td>
      <td>5443.0</td>
      <td>110070.0</td>
      <td>2</td>
      <td>1.348418</td>
      <td>-0.676523</td>
      <td>-7.671264</td>
      <td>3.293536</td>
      <td>-8.973567</td>
      <td>4.379221</td>
      <td>-0.594836</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
      <td>-11.830857</td>
      <td>-77.116758</td>
      <td>8691276.063</td>
      <td>269380.8033</td>
    </tr>
    <tr>
      <th>633</th>
      <td>3717</td>
      <td>1963</td>
      <td>69</td>
      <td>3.281690</td>
      <td>0.071513</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>5304.0</td>
      <td>0.0</td>
      <td>520.0</td>
      <td>9091.0</td>
      <td>2</td>
      <td>7.635872</td>
      <td>-0.148072</td>
      <td>-0.013822</td>
      <td>-0.308361</td>
      <td>1.686361</td>
      <td>2.132816</td>
      <td>-1.547070</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>-11.830993</td>
      <td>-77.098410</td>
      <td>8691276.063</td>
      <td>271380.8033</td>
    </tr>
    <tr>
      <th>634</th>
      <td>3719</td>
      <td>1644</td>
      <td>88</td>
      <td>3.217391</td>
      <td>0.064075</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080264</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6294.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>10955.0</td>
      <td>3</td>
      <td>7.146136</td>
      <td>0.256668</td>
      <td>-0.740441</td>
      <td>0.060128</td>
      <td>1.655740</td>
      <td>2.163562</td>
      <td>-1.050646</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>-11.831128</td>
      <td>-77.080061</td>
      <td>8691276.063</td>
      <td>273380.8033</td>
    </tr>
    <tr>
      <th>635</th>
      <td>3721</td>
      <td>2824</td>
      <td>32</td>
      <td>2.815789</td>
      <td>0.113008</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>1</td>
      <td>10.232179</td>
      <td>-3.556909</td>
      <td>8.014171</td>
      <td>-0.549979</td>
      <td>0.111697</td>
      <td>0.047302</td>
      <td>2.388190</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-11.831262</td>
      <td>-77.061712</td>
      <td>8691276.063</td>
      <td>275380.8033</td>
    </tr>
    <tr>
      <th>636</th>
      <td>3723</td>
      <td>5009</td>
      <td>12</td>
      <td>2.857143</td>
      <td>0.226190</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.127762</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>10.918062</td>
      <td>-7.114726</td>
      <td>16.848065</td>
      <td>-5.752593</td>
      <td>-0.976964</td>
      <td>-1.246141</td>
      <td>0.211838</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-11.831395</td>
      <td>-77.043363</td>
      <td>8691276.063</td>
      <td>277380.8033</td>
    </tr>
    <tr>
      <th>637</th>
      <td>3726</td>
      <td>1175</td>
      <td>14</td>
      <td>2.764706</td>
      <td>0.144608</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>11.095935</td>
      <td>-5.519466</td>
      <td>11.075864</td>
      <td>-1.792307</td>
      <td>-0.350198</td>
      <td>-0.696229</td>
      <td>2.316277</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-11.831592</td>
      <td>-77.015839</td>
      <td>8691276.063</td>
      <td>280380.8033</td>
    </tr>
    <tr>
      <th>638</th>
      <td>3775</td>
      <td>1218</td>
      <td>22</td>
      <td>2.838710</td>
      <td>0.146830</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>7.627340</td>
      <td>-7.402113</td>
      <td>10.178699</td>
      <td>0.089758</td>
      <td>-0.872388</td>
      <td>-1.695192</td>
      <td>-0.414001</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-11.821404</td>
      <td>-77.171731</td>
      <td>8692276.063</td>
      <td>263380.8033</td>
    </tr>
    <tr>
      <th>639</th>
      <td>3776</td>
      <td>1421</td>
      <td>89</td>
      <td>3.000000</td>
      <td>0.067887</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>5.748206</td>
      <td>-2.735252</td>
      <td>1.033739</td>
      <td>3.364322</td>
      <td>0.663051</td>
      <td>0.301562</td>
      <td>-0.319776</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-11.821474</td>
      <td>-77.162558</td>
      <td>8692276.063</td>
      <td>264380.8033</td>
    </tr>
    <tr>
      <th>640</th>
      <td>3779</td>
      <td>6319</td>
      <td>180</td>
      <td>3.063725</td>
      <td>0.042501</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>4.194172</td>
      <td>0.918488</td>
      <td>-3.579627</td>
      <td>6.071268</td>
      <td>1.642519</td>
      <td>1.328820</td>
      <td>0.397949</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.821682</td>
      <td>-77.135037</td>
      <td>8692276.063</td>
      <td>267380.8033</td>
    </tr>
    <tr>
      <th>641</th>
      <td>3780</td>
      <td>1442</td>
      <td>156</td>
      <td>2.994286</td>
      <td>0.044316</td>
      <td>0.000000</td>
      <td>1242.905239</td>
      <td>0.010531</td>
      <td>142.0</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>25666.0</td>
      <td>0.0</td>
      <td>4923.0</td>
      <td>100979.0</td>
      <td>0</td>
      <td>3.760160</td>
      <td>-1.734009</td>
      <td>-4.407463</td>
      <td>4.925109</td>
      <td>-5.908830</td>
      <td>2.506965</td>
      <td>1.798920</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>-11.821751</td>
      <td>-77.125863</td>
      <td>8692276.063</td>
      <td>268380.8033</td>
    </tr>
    <tr>
      <th>642</th>
      <td>3785</td>
      <td>1731</td>
      <td>13</td>
      <td>2.800000</td>
      <td>0.139927</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>10.878596</td>
      <td>-5.130930</td>
      <td>10.567279</td>
      <td>-1.790558</td>
      <td>-0.254679</td>
      <td>-0.484708</td>
      <td>2.050301</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-11.822091</td>
      <td>-77.079993</td>
      <td>8692276.063</td>
      <td>273380.8033</td>
    </tr>
    <tr>
      <th>643</th>
      <td>3788</td>
      <td>2032</td>
      <td>50</td>
      <td>2.962963</td>
      <td>0.108047</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>990.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>1864.0</td>
      <td>0</td>
      <td>9.767519</td>
      <td>-2.642534</td>
      <td>5.818971</td>
      <td>-0.735702</td>
      <td>0.681995</td>
      <td>0.714320</td>
      <td>1.307652</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-11.822291</td>
      <td>-77.052470</td>
      <td>8692276.063</td>
      <td>276380.8033</td>
    </tr>
    <tr>
      <th>644</th>
      <td>3844</td>
      <td>2363</td>
      <td>13</td>
      <td>2.687500</td>
      <td>0.152381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91963.0</td>
      <td>0</td>
      <td>8.044514</td>
      <td>-8.151473</td>
      <td>12.211521</td>
      <td>0.536849</td>
      <td>-1.389954</td>
      <td>-2.327158</td>
      <td>0.825027</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>-11.812576</td>
      <td>-77.144140</td>
      <td>8693276.063</td>
      <td>266380.8033</td>
    </tr>
    <tr>
      <th>645</th>
      <td>3845</td>
      <td>2234</td>
      <td>154</td>
      <td>3.024096</td>
      <td>0.046908</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20362.0</td>
      <td>0.0</td>
      <td>4403.0</td>
      <td>91888.0</td>
      <td>0</td>
      <td>4.961208</td>
      <td>-0.593878</td>
      <td>-2.668487</td>
      <td>5.328771</td>
      <td>1.407773</td>
      <td>0.928428</td>
      <td>0.363715</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.812645</td>
      <td>-77.134967</td>
      <td>8693276.063</td>
      <td>267380.8033</td>
    </tr>
    <tr>
      <th>646</th>
      <td>3907</td>
      <td>1506</td>
      <td>41</td>
      <td>3.404762</td>
      <td>0.089983</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>2</td>
      <td>8.475786</td>
      <td>-0.383027</td>
      <td>1.260831</td>
      <td>-2.691461</td>
      <td>1.844396</td>
      <td>2.549436</td>
      <td>-2.593849</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-11.803330</td>
      <td>-77.171589</td>
      <td>8694276.063</td>
      <td>263380.8033</td>
    </tr>
    <tr>
      <th>647</th>
      <td>4040</td>
      <td>4533</td>
      <td>72</td>
      <td>3.112500</td>
      <td>0.079286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.096302</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>0</td>
      <td>8.253958</td>
      <td>0.100771</td>
      <td>2.161373</td>
      <td>-0.639658</td>
      <td>1.292977</td>
      <td>2.059206</td>
      <td>0.286690</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-11.785326</td>
      <td>-77.162274</td>
      <td>8696276.063</td>
      <td>264380.8033</td>
    </tr>
    <tr>
      <th>648</th>
      <td>4041</td>
      <td>1227</td>
      <td>54</td>
      <td>3.185185</td>
      <td>0.103847</td>
      <td>0.000000</td>
      <td>1542.810577</td>
      <td>0.186673</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>7</td>
      <td>7.608644</td>
      <td>-1.693240</td>
      <td>1.516269</td>
      <td>-4.832180</td>
      <td>-7.638683</td>
      <td>4.304078</td>
      <td>0.740304</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>-11.785395</td>
      <td>-77.153102</td>
      <td>8696276.063</td>
      <td>265380.8033</td>
    </tr>
    <tr>
      <th>649</th>
      <td>4105</td>
      <td>6060</td>
      <td>165</td>
      <td>3.078652</td>
      <td>0.052909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.064199</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>7.102299</td>
      <td>3.127608</td>
      <td>-2.364986</td>
      <td>1.662651</td>
      <td>2.190932</td>
      <td>2.228158</td>
      <td>1.426332</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>-11.776219</td>
      <td>-77.171376</td>
      <td>8697276.063</td>
      <td>263380.8033</td>
    </tr>
    <tr>
      <th>650</th>
      <td>4106</td>
      <td>3668</td>
      <td>130</td>
      <td>3.175573</td>
      <td>0.061268</td>
      <td>0.000000</td>
      <td>3270.108195</td>
      <td>0.088558</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.0</td>
      <td>3</td>
      <td>6.165648</td>
      <td>0.553022</td>
      <td>-5.743413</td>
      <td>-2.234864</td>
      <td>-16.498842</td>
      <td>7.867602</td>
      <td>4.961850</td>
      <td>6</td>
      <td>6</td>
      <td>8</td>
      <td>-11.776289</td>
      <td>-77.162204</td>
      <td>8697276.063</td>
      <td>264380.8033</td>
    </tr>
    <tr>
      <th>651</th>
      <td>4171</td>
      <td>3753</td>
      <td>48</td>
      <td>3.244898</td>
      <td>0.116279</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055028</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8.711044</td>
      <td>-1.278561</td>
      <td>4.485927</td>
      <td>-2.970254</td>
      <td>1.204063</td>
      <td>1.737618</td>
      <td>-1.536139</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>-11.767182</td>
      <td>-77.171305</td>
      <td>8698276.063</td>
      <td>263380.8033</td>
    </tr>
  </tbody>
</table>
<p>652 rows × 30 columns</p>
</div>



Plotting results


```python
plot_map(ClustersData, 'kopt+0_Clusters')
```


![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_69_0.png)



```python
plot_map(ClustersData, 'kopt+1_Clusters')
```


![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_70_0.png)



```python
plot_map(ClustersData, 'kopt+2_Clusters')
```


![png](World_Bank_Clutering_Script_v1.1_LIM_files/World_Bank_Clutering_Script_v1.1_LIM_71_0.png)


Saving file


```python
save_results(ClustersData, file_name='Output/'+city_name+'/'+city_name+'_ClustersData_withPOI.xlsx')
save_results(centroids, file_name='Output/'+city_name+'/'+city_name+'_Centroids_withPOI.xlsx')
```

End of code.
