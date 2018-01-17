
# Mapping Population to Pixels


*MIT Megacity Logistics Lab*(c): Daniel Merchan <dmerchan@mit.edu>, Andre Snoeck <asnoeck@mit.edu>, Esteban Mascarino <estmasca@mit.edu>



**Summary**: This script can be used to map LandScan population cells to a list of pixels

Input data requirements:
- Origin point coordinates (lower left corner of the pixel grid)
- Number of kilometers in the horizontal axis (longitude)
- Number of kilometers in the veritcal axis (latitute)
- Populations files

**Case Study**: Sao Paulo

This script processes and analyses one set of data:
1. GeoSpatial data - Maps population data (LandScan) to create a rectangular grid over a city or ***pixels***. 

The total size of the grid must be defined by the user based upon each case study. For instance, for **Sao Paulo**, the size to cover the entire metropolitan area is approximately 75 km (vertical axis) x 100 km (horizontal axis). 

Furthremore, the size/area of each pixel might vary based on visualization/modeling considerations. The minimum pixel size for this code is 1 sq.km. 

** Run time **: ~ 2 minutes


## References

- LandsScan:
This product was made utilizing the LandScan (insert dataset year)™ High Resolution global Population Data Set copyrighted by UT-Battelle, LLC, operator of Oak Ridge National Laboratory under Contract No. DE-AC05-00OR22725 with the United States Department of Energy.  The United States Government has certain rights in this Data Set.  Neither UT-BATTELLE, LLC NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE DATA SET


Import Python libraries and functions from other scripts


```python
# Basic numeric libraries
import numpy as np
import math as m
import pandas as pd
pd.options.mode.chained_assignment = None
#pd.options.display.max_columns = 20
from scipy import stats
from __future__ import division

# Library to handle geometric objects: 
from shapely.geometry import Point, MultiPoint, box, Polygon 

# Libraries for data visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)

#Libraries for handling temporal data or monitoring processing time
import datetime as dt
import time

# Libraries for statistical analysis
import statsmodels.api as sm
import statsmodels.formula.api as smf

#System libraries
import utm
import sys
import random
from __future__ import division
import os
import csv
```


Key definitions:
- **urbgrid**: Grid of datacells or pixels to wich data (orders, customer locations, population, etc) will be mapped.
+ **pixel**: Each cell within the grid. Default size: 1 sq.km.


### 1. City Settings: Select city and specify grid origin point

#### 1.1 User inputs the city name and coordinates origin (lower bound for latitude and longitude) for the grid generation


```python
# The user must enter three inputs in this cell:
#  - a string with the name of the city in the "city" variable
#  - a float representing the lower bound in latitude for the origin of the pixel grid in the "latitude" variable
#  - a float representing the lower bound in longitude for the origin of the pixel grid in the "longitude" variable
city =  str(input("Enter the name of the city enclosed by '': "))
```

    Enter the name of the city enclosed by '': 'Lima'


The code now **automatically** creates a data frame with the input city and the lower-left corner's coordinates of the city grid. No input from the user is required at this stage.


```python
#Dictionaries below allow for handling several cities at the same time
origins = pd.DataFrame({'lat': [-12.340000, 4.420000, -0.619080],
                        'lon': [-77.300000, -74.390000, -78.857006]},
                       index = ['Lima', 'Bogota', 'Quito'])


# Conver Lat-lon coordinates to UTM coordinates. This is conversion is necesary to project the lat-lon system in a 
# 2 dimensional system.

for index, row in origins.iterrows():
    [east, north, zone_n, zone_l] = utm.from_latlon(row['lat'], row['lon'])
    origins.ix[index,'utm_n'] = north
    origins.ix[index, 'utm_e'] = east
    origins.ix[index, 'utm_z_n'] = zone_n
    origins.ix[index, 'utm_z_l'] = zone_l

print origins
```

                 lat        lon         utm_n          utm_e  utm_z_n utm_z_l
    Lima   -12.34000 -77.300000  8.634776e+06  249880.803307     18.0       L
    Bogota   4.42000 -74.390000  4.885803e+05  567678.470377     18.0       N
    Quito   -0.61908 -78.857006  9.931525e+06  738503.728230     17.0       M


The code **automatically** creates a bounding box around the city. The dictionary xy_dim specifies the vertical and horizontal extension of the box, previously entered by the user. No input from the user is required at this stage.


```python
xy_dim = pd.DataFrame({'x_dim': [66, 70, 70],
                       'y_dim': [65, 50, 66]},
                      index = ['Lima', 'Bogota', 'Quito'])

origin_city = Point(origins.loc[city]['utm_e'], origins.loc[city]['utm_n'])

citybox = box(origin_city.x, origin_city.y, 
              origin_city.x + xy_dim.loc[city]['x_dim']*1000, origin_city.y + xy_dim.loc[city]['y_dim']*1000)
print 'Bounding box created'
```

    Bounding box created


A function is defined to generate the required grid.


```python
'''
To create a grid of (x_dim)*(y_dim) pixels/cells
Input
- x_dim: horizontal pixel dimension
- y_dim: vertical pixel dimension
- city_box: city's bounding box
- margin: Default = 0. useful for small areas only. It helps accommodate factions of population data in the borders of the cell.
    
Output 
- grid: 
'''

def getgrid(dim_x, dim_y, citybox, margin = 0):
    
    # Exterior coordinates of citybox are given starting at the bottom right corner 
    # of the box (x1, y0), countercklecwise: (x1, y0), (x1,y1), (x0, y1), (x0,y0) 
    #& again (x1, y0) 
    
  
    
    #Start grid from bottom-left corner 
    corner = Point(citybox.exterior.coords[3])
    origin = Point(citybox.exterior.coords[3])
    grid = [[]]
    j=0 #
    #i=0 # 
    while (corner.y+ dim_y <= citybox.bounds[3] + margin):
        while (corner.x+ dim_x<= citybox.bounds[2]+ margin):
            grid[j].append(Polygon([(corner.x, corner.y),
                                    (corner.x+dim_x, corner.y),
                                    (corner.x+dim_x, corner.y+dim_y),
                                    (corner.x, corner.y+dim_y)]))
            corner = Point(corner.x+dim_x,corner.y)
        j+=1
        origin = Point(citybox.exterior.coords[3])
        corner = Point(origin.x, origin.y + j*dim_y)
        if corner.y+dim_y <= (citybox.bounds[3]+margin):
            grid.append([])
    print 'Grid created'
    return grid
```

The code **automatically** generates the urbgrid. No input from the user is required at this stage.


```python
# By default and for all our analysis these two variables should always be assigned to 1000
pix_dim_x = 1000
pix_dim_y = 1000

# Calling the function to create urbgrid
urbgrid = getgrid(pix_dim_x, pix_dim_y, citybox)
```

    Grid created


## 2. Read and process population data from LandScan
Population counts will be maped to each pixel/datacell. This will be performed in two separate operations:
1. Read population data from LandScan
+ Create **lsgrid** to standarize LandScan cell sizes (see explanation below)

The input file name and directory is **automatically** detected. No input from the user is required at this stage.


```python
# The folder separator for iOS and Linux is "/" whereas for Windows it is "\"
if city == 'Lima':
    file_path_name = 'Input/Lima.txt'
elif city == 'Bogota':
    file_path_name = 'Input/Bogota.txt'
elif city == 'Quito':
    file_path_name = 'Input/Quito.txt'
```

The code is **automatically** reading LandScan population for the analyzed city. No input from the user is required at this stage.


```python
city_pop = pd.read_csv(file_path_name)

for index, row in city_pop.iterrows():
    [east, north, zone_n, zone_l] = utm.from_latlon(row['lat'], row['lon'])
    city_pop.ix[index,'utm_n'] = north
    city_pop.ix[index,'utm_e'] = east
    city_pop.ix[index,'utm_z_n'] = zone_n
    city_pop.ix[index,'utm_z_l'] = zone_l
#print the first few row to verify
city_pop.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FID</th>
      <th>POINTID</th>
      <th>population</th>
      <th>lat</th>
      <th>lon</th>
      <th>utm_n</th>
      <th>utm_e</th>
      <th>utm_z_n</th>
      <th>utm_z_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>5.070833</td>
      <td>-74.012500</td>
      <td>560577.495891</td>
      <td>609462.193528</td>
      <td>18.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>134</td>
      <td>5.062500</td>
      <td>-74.020833</td>
      <td>559654.805186</td>
      <td>608539.763048</td>
      <td>18.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5.062500</td>
      <td>-74.012500</td>
      <td>559656.204426</td>
      <td>609463.595983</td>
      <td>18.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5.062500</td>
      <td>-74.004167</td>
      <td>559657.615530</td>
      <td>610387.431213</td>
      <td>18.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>376</td>
      <td>5.062500</td>
      <td>-73.995833</td>
      <td>559659.038498</td>
      <td>611311.268769</td>
      <td>18.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>



Create grid to standarize LandScan cells. The need for standardized cells arises form the fact that the exact landscan cell size varies by lattitude and longitude( even within the same city, the size of each cell varies). Then we need to define an "average" landscan cell size to create our grid. 

The size of the "standard" LandScan cell size varies by city and our code automatically calculates that "average" cell's size.

Refer to LandScan's site for more information: http://web.ornl.gov/sci/landscan/landscan_faq.shtml#04

The code is now **automatically** normalizing the cell's size. No input from the user is required at this stage.


```python
# Approximation to the LandScan individual cell's size

# Initialize lists with distance to average:
x_dist_list = []
y_dist_list = []

# Iterate over all the LandScan cells
for i in city_pop.index.tolist():
    # Get the coordinates of the cell being analyzed
    utm_n_cell = city_pop.get_value(i, 'utm_n')
    utm_e_cell = city_pop.get_value(i, 'utm_e')
    
    # Iterate over the remaining cells to get the next x distance
    for j in range(i, len(city_pop)):
        utm_e_compare = city_pop.get_value(j, 'utm_e')
        # Limits arbitrarily defined based on Landscan's maximum feasible distance between points and logical minimum distance
        if ((abs(utm_e_cell - utm_e_compare) <= 1000) and (abs(utm_e_cell - utm_e_compare) > 500)):
            x_dist_list.append(abs(utm_e_cell - utm_e_compare))
            break
    
    # Iterate over the remaining cells to get the next y distance
    for k in range(i, len(city_pop)):
        utm_n_compare = city_pop.get_value(k, 'utm_n')
        # Limits arbitrarily defined based on Landscan's maximum feasible distance between points and logical minimum distance
        if ((abs(utm_n_cell - utm_n_compare) <= 1000) and (abs(utm_n_cell - utm_n_compare) > 500)):
            y_dist_list.append(abs(utm_n_cell - utm_n_compare))
            break

cell_dim_x_avg = sum(x_dist_list)/len(x_dist_list)
cell_dim_y_avg = sum(y_dist_list)/len(y_dist_list)

#The location of Sao Paulo results in lat-long cells of aporx. 850*915. For San Jose - SFO those are # 738 * 927

ls_dim = pd.DataFrame({'cel_dim_x': (cell_dim_x_avg),
                       'cel_dim_y': (cell_dim_y_avg)},
                      index = [city])

# Buffer to capture those cell that fall <500 m of the border 
# box_margin = 500

lsgrid = getgrid(ls_dim.loc[city,'cel_dim_x'], ls_dim.loc[city,'cel_dim_y'], citybox)
```

    Grid created


Map landscan centroids to **lsgrid**. Since **lsgrid** is a set of poligons that cannot contain data, we generate a parallel matrix pop_losgrid with the mapped population values. We should not expect a 100% mapping since the area selected from Landscan can't be specified with precision and might differ in size from the area used to create the city bounding box. We expect these differences to be irrelevant.

The code is now **automatically** performing that conversion. No input from the user is required at this stage.


```python
#Extract Population, utem_n and utm_e to a 2-dim array
city_pop_asg = city_pop[['population','utm_e', 'utm_n']].as_matrix()

pop_lsgrid = [[]]
for j in xrange(len(lsgrid)): # to iterate over all rows of the grid
    # filter orders within row j using the utm_n colum (x[2])
    filtered_cell = filter(
        lambda x: x[2]>=lsgrid[j][0].bounds[1] and x[2]<lsgrid[j][0].bounds[3], city_pop_asg)
    #filtered points are filtered again and assigned to a cell    
    for i in xrange(len(lsgrid[0])): #to iterate over all pixels (columns) within each row
        # filter orders within row j using the utm_n colum (x[2]) 
        assigned_cell = filter(
            lambda x: x[1]>=lsgrid[0][i].bounds[0] and x[1]<lsgrid[0][i].bounds[2], filtered_cell)
        # create the orders-grid containing the number of orders in each pixel
        pop_lsgrid[j].append(sum(assigned_cell[i][0] for i in xrange (len(assigned_cell))))
    if j < len(lsgrid)-1:
        pop_lsgrid.append([])
```


```python
print "Total population imported from LandScan:", city_pop['population'].sum()
print "Total population mapped to lsgrid:", sum(map(sum, pop_lsgrid))
print "% Population mapped", (sum(map(sum, pop_lsgrid))/city_pop['population'].sum())*100, "%"
```

    Total population imported from LandScan: 9089423
    Total population mapped to lsgrid: 8771315.0
    % Population mapped 96.5002398942 %


## 3. Mapping LandScan population to pixels


Once LandScan population counts have been mapped to a standard (**lsgrid**) grid, we can now map them to the pixels (**urbgrid**). Similarly, we generate a new matrix pop_urbgrid (same size as urbgrid). The function that performs that task is now defined. No input from the user is required at this stage.


```python
'''
To map data from a landscan grid to a grid of pixels. Since the size of the landscan grid cell is equal or smaller than 1 sq.km. (pixel size),
we need to iterate over each ls cell and assign the population proportion accordingly. 
Input:
- small_pix_grid (polygons): This is generally the landscan polygons grid, aka 'lsgrid'
- big_pix_grid (polygons): This is the grid of polygons to which population data will be mapped, aka 'urbgrid'. 
- population_grid (data): population informatio to be mapped, aka pop_lsgrid

Output:
- pop_mapped_grid(data): grid of pixels containing population data. Size should be equal as big_pix_grid.  

'''


def MapPopulationtoGrid(small_pix_grid, big_pix_grid, population_grid):
    
    print 'Start time:', time.ctime()
    

    #Initialize indexes for big_pix_grid j, i
    i=0
    j=0
    # matrix to store each pixel's population count
    pop_mapped_grid = [[0 for x in xrange (len(big_pix_grid[0]))] for y in xrange (len(big_pix_grid))]
    
    
    
    #iterate over the landscan grid
    for y in xrange (len(small_pix_grid)):
        #print 'y', y
        if (i < (len(big_pix_grid)-1)):
            j=0
            for x in xrange (len(small_pix_grid[y])):
                #print 'x', x
                if (small_pix_grid[y][x].intersects(big_pix_grid[i+1][j]) == False):
                    #print 'Case l'
                    #case 1: lscell only within one urbgrid row
                    if ((small_pix_grid[y][x].within(big_pix_grid[i][j]) == True) or (j == (len(big_pix_grid[i])-1))):
                        #print 'Case 1.1'
                        #case 1.1: data cell within landscan cell"
                        #Assign the entire population of this lscell to the corresponding urbgrid cell
                        pop_mapped_grid[i][j] += population_grid[y][x]

                    elif (small_pix_grid[y][x].intersects(big_pix_grid[i][j]) == True) and (small_pix_grid[y][x].intersects(big_pix_grid[i][j+1]) == True):
                        #print 'Case 1.2'
                        #"case 1.2: lscell intersects 2 landscan cells horizontally"
                        area_inters = small_pix_grid[y][x].intersection(big_pix_grid[i][j])
                        area_inters_next = small_pix_grid[y][x].intersection(big_pix_grid[i][j+1])
                        #Assign population by the corresponding fraction
                        pop_mapped_grid[i][j] += (((area_inters.area)/(small_pix_grid[y][x].area))*(population_grid[y][x]))
                        pop_mapped_grid[i][j+1] += (((area_inters_next.area)/(small_pix_grid[y][x].area))*(population_grid[y][x]))
                        
                        #to move one column in the big_pix_grid, after reaching an intersection
                        if j <= (len(big_pix_grid[i]) - 2):
                            j+=1
                        else:
                            #Finish iteration over the row
                            x = len(small_pix_grid[y])
                else:
                    #print 'Case 2'
                    #case 2: lscell intersect two urbgird cell rows
                    if (j < (len(big_pix_grid[i])-1)):
                        if (small_pix_grid[y][x].intersects(big_pix_grid[i][j+1]) == True):
                            #case 2.2: lscell intersects upper row and 2 urbgrid cells horizontally"
                            area_inters = small_pix_grid[y][x].intersection(big_pix_grid[i][j])
                            area_inters_next = small_pix_grid[y][x].intersection(big_pix_grid[i][j+1])
                            area_inters_up = small_pix_grid[y][x].intersection(big_pix_grid[i+1][j])
                            area_inters_up_next = small_pix_grid[y][x].intersection(big_pix_grid[i+1][j+1])
                            
                            pop_mapped_grid[i][j] += (((area_inters.area)/(small_pix_grid[y][x].area))*(population_grid[y][x]))
                            
                            pop_mapped_grid[i][j+1] += ((area_inters_next.area)/(small_pix_grid[y][x].area))*(population_grid[y][x])
                            
                            pop_mapped_grid[i+1][j] += ((area_inters_up.area)/(small_pix_grid[y][x].area))*(population_grid[y][x]) 
                            
                            pop_mapped_grid[i+1][j+1] += ((area_inters_up_next.area)/(small_pix_grid[y][x].area))*(population_grid[y][x])
                            
                            
                            #to move one column in the big_pix_grid, after reaching an intersection
                            if j <= (len(big_pix_grid[i]) - 2):
                                j+=1
                            else:
                                #Finish iteration over the row
                                x = len(small_pix_grid[y])
                                
                        else:
                            #print "case 2.1: ls cell overlaps with upper row urbgrid cells but there is no horizontal intersection"
                            area_inters = small_pix_grid[y][x].intersection(big_pix_grid[i][j])
                            area_inters_up = small_pix_grid[y][x].intersection(big_pix_grid[i+1][j])
                            pop_mapped_grid[i][j] += ((area_inters.area)/(small_pix_grid[y][x].area))*(population_grid[y][x])
                            pop_mapped_grid[i+1][j] += ((area_inters_up.area)/(small_pix_grid[y][x].area))*(population_grid[y][x])
                            
                    
                    elif (j == (len(big_pix_grid[i])-1)):
                            #print "case 2.3 when reaching the horizontal border"
                            area_inters = small_pix_grid[y][x].intersection(big_pix_grid[i][j])
                            area_inters_up = small_pix_grid[y][x].intersection(big_pix_grid[i+1][j])
                            pop_mapped_grid[i][j] += ((area_inters.area)/(small_pix_grid[y][x].area))*(population_grid[y][x])
                            pop_mapped_grid[i+1][j] += ((area_inters_up.area)/(small_pix_grid[y][x].area))*(population_grid[y][x])
                            
                            
                    # To move one row up in the big_pix_grid only ofter after the loop signals an intersection across two row
                    if (x == (len(small_pix_grid[y])-1)):
                        i+=1
                        #print 'i',i
                        
                        
            pop_mapped_grid.append([])

        #For the topmost row in the big_pix_grid, there can't be intersections with another upper row, then only a subset 
        #of the cases (i.e. 1.1. and 1.2 apply)
        elif (i == (len(big_pix_grid)-1)):
            #print 'top row'
            j=0
            #"Special case: top row of big_pix_grid
            for x in xrange (len(small_pix_grid[y])):
                #case 1: datcells only within one landscan cell rows
                if ((small_pix_grid[y][x].within(big_pix_grid[i][j]) == True) or (j == (len(big_pix_grid[i])-1))):
                    pop_mapped_grid[i][j] += population_grid[y][x]

                elif (small_pix_grid[y][x].intersects(big_pix_grid[i][j]) == True) and (small_pix_grid[y][x].intersects(big_pix_grid[i][j+1]) == True):
                    #print "case 1.2X: data cell intersects 2 landscan cells horizontally"
                    area_inters = small_pix_grid[y][x].intersection(big_pix_grid[i][j])
                    area_inters_next = small_pix_grid[y][x].intersection(big_pix_grid[i][j+1])
                    pop_mapped_grid[i][j] += (((area_inters.area)/(small_pix_grid[y][x].area))*(population_grid[y][x]))
                    pop_mapped_grid[i][j+1] += (((area_inters_next.area)/(small_pix_grid[y][x].area))*(population_grid[y][x]))
    
                    if j <= (len(big_pix_grid[i]) - 2):
                        j+=1
                else:
                    pop_mapped_grid[y].append(0)

            if (y < (len(big_pix_grid))-1):
                pop_mapped_grid.append([])
                
    print 'End time:', time.ctime()
                
    return pop_mapped_grid
```

The previously defined function is now **automatically** called in order to map the population grid. Moreover, some preliminary results are shown. No input from the user is required at this stage.


```python
pop_urbgrid = MapPopulationtoGrid(lsgrid, urbgrid, pop_lsgrid)
```

    Start time: Wed May  3 00:08:42 2017
    End time: Wed May  3 00:08:47 2017



```python
sum_pop_lsgrid = sum(map(sum, pop_lsgrid))
sum_pop_urbgrid = sum(map(sum, pop_urbgrid))

print "Total population mapped to lsgrid:", sum_pop_lsgrid
print "Total population mapped to urbgrid:", sum_pop_urbgrid
print "% Population mapped", sum_pop_urbgrid/ sum_pop_lsgrid*100, "%"
```

    Total population mapped to lsgrid: 8771315.0
    Total population mapped to urbgrid: 8771315.0
    % Population mapped 100.0 %


## 4. Process and export results

The code **automatically** generates a dataframe of pixels. No input from the user is required at this stage.


```python
# to convert both grids to a list of sq.kms. (ie. pixels) with lat-lon data 
def get_pixel_list(urbgrid, pop_urbgrid):
    print 'Start time:', time.ctime()
    pixel_list = [[0 for j in range(0,6)] for k in xrange(len(urbgrid)*len(urbgrid[0]))]
    k = 0 
    for i in xrange(len(urbgrid)):
        for j in xrange(len(urbgrid[0])): 
            (Lat, Lon) = utm.to_latlon(urbgrid[i][j].centroid.x, urbgrid[i][j].centroid.y,
                                     origins.loc[city]['utm_z_n'], origins.loc[city]['utm_z_l'] )
            pixel_list[k][1] = Lat                         #Latiude in decimal degrees   
            pixel_list[k][2] = Lon                         #Longitude in decimal degrees
            pixel_list[k][3] = urbgrid[i][j].centroid.y    #Northing   
            pixel_list[k][4] = urbgrid[i][j].centroid.x    #Easting
            pixel_list[k][5] = round(pop_urbgrid[i][j])    # population
            k+=1
    pixel_list_df = pd.DataFrame(pixel_list)
    pixel_list_df.columns = ['pixel_ID','lat', 'lon', 'utm_n', 'utm_e','population']
    pixel_list_df['pixel_ID'] = pixel_list_df.index.values
    
    pixel_list_df[['pixel_ID','population']] = pixel_list_df[['pixel_ID','population']].astype(int)
    print 'End time:', time.ctime()
    return pixel_list_df

###
# The polygon of each pixel could also be printed by adding:
#datacell_list[k][5] = urbgrid[i][j]         
```


```python
pixel_list = get_pixel_list(urbgrid, pop_urbgrid)
```

    Start time: Wed May  3 00:08:59 2017
    End time: Wed May  3 00:09:00 2017


The output file name and directory is **automatically** defined. No input from the user is required at this stage.


```python
# The folder separator for iOS and Linux is "/" whereas for Windows it is "\"
if city == 'Lima':
    destination_file_path_name = 'Output/LIM_pixels_population.csv'
elif city == 'Bogota':
    destination_file_path_name = 'Output/BOG_pixels_population.csv'
elif city == 'Quito':
    destination_file_path_name = 'Output/UIO_pixels_population.csv'
```

The code **automatically** saves the results in the specified file. No input from the user is required at this stage.


```python
#Export results
pixel_list.to_csv(destination_file_path_name, index = False)
```


```python

```
