
# Road Network Analysis

## Network Analysis - v0.2


*MIT Megacity Logistics Lab*(c): Daniel Merchan <dmerchan@mit.edu>, Esteban Mascarino <estmasca@mit.edu>, Matthias Winkenbach <mwinkenb@mit.edu>

**Summary**: This script reads a list of pixels and obtains relevant network-based metrics. Uses OSMnx library. 

Details about OSMnx can be found here: https://github.com/gboeing/osmnx


```python
# Basic numeric libraries
import numpy as np
import math as m
import scipy as sp
import pandas as pd
pd.options.mode.chained_assignment = None
#from scipy import stats
from __future__ import division

# Library to handle geometric objects:
#from shapely.geometry import Point, MultiPoint, box, Polygon

# Libraries for data visualization
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for statistical analysis
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression as Lin_Reg
import random

#Libraries for handling temporal data or monitoring processing time
import datetime as dt
import time
from time import sleep
#import multiprocessing as mp

#Specify code isualization settings
sns.set(color_codes=True)
pd.options.display.max_columns = 40
%matplotlib inline

#Libraries to work with geospatial data
import zipcode
import googlemaps
import utm

#System libraries
import sys
import os
import itertools


#Libraries for optimization
import networkx as nx
import osmnx as ox
from IPython.display import Image
%matplotlib inline
ox.config(log_file=True, log_console=True, use_cache=True)
#from gurobipy import *
```

## Network statistics for a list of pixels

## Functions


```python
'''
Input:
- file_csv: file name 
Output:
- pixs: dataframe of pixels
'''
def read_file(file_csv):
    pixs = pd.read_csv(file_csv, index_col=["pixel_ID"])
    return pixs
```

To get basic and advanced network statistics as defined by OSMnx. For futhre details see: https://github.com/gboeing/osmnx/blob/master/examples/06-example-osmnx-networkx.ipynb


```python
'''
Input:
- network: OSMnx network extraction
- relevant_stats_names: list of selected statistics
Output:
- relevant_basic_stats_values: values for the relevant basic statistics 
'''
def get_basic_nextwork_stats(network, relevant_stats_names):
    
    basic_stats = ox.basic_stats(network, area = 1000000)
    
    relevant_basic_stats_values = []
    for stat in relevant_stats_names:
        relevant_basic_stats_values.append(basic_stats[stat])
        
    return relevant_basic_stats_values
```


```python
'''
Input:
- network: OSMnx network extraction
- relevant_stats_names: list of selected statistics
Output:
- relevant_extended_stats_values: values for the relevant extended statistics 
'''
def get_extended_nextwork_stats(network, relevant_stats_names):
    
    extended_stats = ox.extended_stats(G, connectivity = False, anc = True, ecc = True, bc=True, cc = True)
    
    relevant_extended_stats_values = []
    for stat in relevant_stats_names:
        relevant_extended_stats_values.append(extended_stats[stat])
        
    return relevant_extended_stats_values
```

## Run script

Load the file


```python
city =  str(input("Enter the name of the city enclosed by'': "))
```


```python
# The folder separator for iOS and Linux is "/" whereas for Windows it is "\"
if city == 'Lima':
    file_name = 'Input/LIM_pixels_population.csv'
elif city == 'Bogota':
    file_name = 'Input/BOG_pixels_population.csv'
elif city == 'Quito':
    file_name = 'Input/UIO_pixels_population.csv'
```


```python
pixels = read_file(file_name)
```


```python
pixels.head()
```

Define relevant statistics


```python
relevant_basic_stats_names = ['count_intersections',
                              'street_length_avg','street_length_total',
                              'streets_per_node_avg']
relevant_extended_stats_names = ['betweenness_centrality_avg','closeness_centrality_avg',
                                 'clustering_coefficient_avg']
```

Extract pixels index & lat_lons 


```python
pixels_index_list = pixels.index.values
pixels_latlon_matrix = pixels[['lat', 'lon']].as_matrix()
```

Get statistics for each pixel 


```python
print 'Start time:', time.ctime()

#Matrix for all network statistics
network_stats = []
#to keep track of pixels in which an error was reported
pixels_errors = []
counter = 0
#iterate over the list of pixels
for pixel_centroid in pixels_latlon_matrix:
    counter +=1
    print 'Pixel', counter
    pixel_basic_stats = []
    pixel_extended_stats = []
    try: 
        G = ox.graph_from_point((pixel_centroid[0], pixel_centroid[1]), distance=500, distance_type = 'bbox', network_type='drive', simplify = True,  clean_periphery = True)
        print 'Network extracted'
        pixel_basic_stats = get_basic_nextwork_stats(G, relevant_basic_stats_names)
        #print 'Basic stats processed'
        pixel_extended_stats = get_extended_nextwork_stats(G, relevant_extended_stats_names)
        #print 'Extended stats processed'
    except:
        print 'Network could not be extracted'
        #print np.indices[np.where(pixels_latlon_matrix==pixel_centroid)]
        pixels_errors.append(pixels_index_list[counter-1])
        pixel_basic_stats = [0 for stat in xrange(0,len(relevant_basic_stats_names))]
        pixel_extended_stats = [0 for stat in xrange(0,len(relevant_extended_stats_names))]

    network_stats.append(pixel_basic_stats + pixel_extended_stats)
    
print 'End time:', time.ctime()
```

Post-processing: Create a dataframe of results and concatenate it to the original pixels dataframe


```python
network_stats_df = pd.DataFrame(network_stats, columns = relevant_basic_stats_names+relevant_extended_stats_names, index = pixels_index_list )
```


```python
network_stats_df.head()
```


```python
pixels_results = pd.concat([pixels, network_stats_df], axis=1)
```

## Export results


```python
pixels_results.head()
```


```python
# The folder separator for iOS and Linux is "/" whereas for Windows it is "\"
if city == 'Lima':
    file_name = 'Output/LIM_pixels_population_roads.csv'
elif city == 'Bogota':
    file_name = 'Output/BOG_pixels_population_roads.csv'
elif city == 'Quito':
    file_name = 'Output/UIO_pixels_population_roads.csv'
```


```python
pixels_results.to_csv(file_name, index = False)
```
