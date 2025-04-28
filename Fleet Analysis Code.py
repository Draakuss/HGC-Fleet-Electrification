import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import DBSCAN
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

#reading CSV data file into a dataframe
file = Path('./fleetdatasheet/file.csv')
if not file.exists():
    raise FileNotFoundError(f"Data file not found: {file}")
df = pd.read_csv(file)

#sorting and grouping data
df = df.sort_values(by=['reg_number', 'timestamp'])

###################################
#Data Cleaning Functions
###################################

#convert timestamped data column to datetime format for time-based operations 
df['timestamp'] = pd.to_datetime(df['timestamp'])

#function to remove GPS outliers using DBSCAN clustering
def remove_gps_outliers_dbscan(df, eps_meters=10000, min_samples=4): #default arguments set at 10km distance looking for gorups of 4
    """Uses DBSCAN clustering to remove spatial outliers based on lon/lat"""
    coords = df[['longitude', 'latitude']].to_numpy()
    
    # Convert meters to radians (approximating for haversine with lat/lon for more accurate modelling)
    kms_per_radian = 6371.0088
    epsilon = eps_meters / 1000 / kms_per_radian

    #running DBscan
    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine')
    radians = np.radians(coords)
    df['cluster'] = db.fit_predict(radians)
    
    # Keep only clustered points (cluster >= 0)
    return df[df['cluster'] != -1].drop(columns='cluster')

#function for cleaning data
def clean_data(df, max_speed=60):
    """
    Cleans data by:
    - Dropping NaNs
    - Removing rows where speed > 60
    - Removing duplicates
    - Removing outliers with DBSCAN
    """
    df = df.dropna()
    df = df[df['speed_tacho'] <= max_speed]
    df = df.drop_duplicates()

    # Apply DBSCAN outlier removal per vehicle
    df = df.groupby('reg_number', group_keys=False).apply(remove_gps_outliers_dbscan)
    
    return df

#clean and sort the dataframe by calling on clean_data function
df_cleaned = clean_data(df)

#sorting dataframe by both datetime and vehicle registration number (not required due to earlier sorting)
df_sorted = df_cleaned.sort_values(by=['reg_number', 'timestamp'])



###################################
#Data Splitting by REG
###################################


#selects data specific to a specified vehicle and returns as dataframe
def HGV_data_return(numberplate):
    """Returns dataframe for a specific vehicle"""
    return df_sorted.loc[df_sorted['reg_number'] == numberplate]



###################################
#Basic Data Analysis Functions
###################################


#average moving speed of vehicle (does not factor rest/idling (0mph))
def speed_avg(dataframe):
    '''returns the average speed of a given HGV dataframe
    Function ignores speeds when the vehicle is stationary
    
    eg. for a df where speed = 10, 0, 6, 0
    avg_speed = 8'''
    avg_spd = dataframe.loc[dataframe.speed_tacho > 0, 'speed_tacho'].mean()
    return avg_spd


#function to calculte distance travelled
def travelled_dist(dataframe):
    '''returns total distance travelled by vehicle
    assumed tachometer uses cumulative distance'''
    distance_travelled = dataframe.distance_tacho.max() #may need to subtract distance.min() if initial distance =/= 0
    return distance_travelled


#Function to return daily travel distance
def daily_travel(dataframe):
    '''
    Returns:
    - Daily distance travelled as a list
    - Average daily distance
    '''
    dataframe['date'] = dataframe['timestamp'].dt.date  #adds new 'date' col to df date without the timestamps
    daily_dist = dataframe.groupby('date')['distance_tacho'].agg(lambda x: x.max() - x.min()) #first groups by date, then selects just the data from 'distance_Tacho
    #agg to apply the function to each group
    return daily_dist, daily_dist.mean()


#function to analsyse vehicle downtime
def downtime(dataframe):
    '''1) Adds a date column to the datframe to group data by the date it was taken
    2)takes the time difference between each timestamp reading
    3)records the sum of all of the time that the HGV was stationary (ie. speed ~ 0)
    4)converts time from seconds into hours 
    '''
    dataframe['date'] = dataframe['timestamp'].dt.date  #adds new 'date' col to df date without the timestamps
    dataframe['time_diff'] = dataframe['timestamp'].diff()
    downtime_data = dataframe.loc[dataframe.speed_tacho < 0.1]
    daily_downtime = downtime_data.groupby('date')['time_diff'].sum().dt.total_seconds()/3600
    average_downtime = daily_downtime.mean()
    return daily_downtime, average_downtime


#Function to cover SOC analysis
def soc_summary(dataframe):
    """
    Returns average, min, and max SOC for a vehicle and states if battery was drained (SOC <= 0)
    """
    is_positive = (dataframe['soc'] > 0).all() #returns TRUE if all soc values are above 0
    return {
        'avg_soc': dataframe['soc'].mean(),
        'min_soc': dataframe['soc'].min(),
        'max_soc': dataframe['soc'].max(),
        'SOC did not reach 0': is_positive
    }


###################################
#Fuel and Energy Cost Comparisons
###################################

#function for finding fuel costs
def ICE_fuel_costs(dataframe, mpg=9, fuel_price_per_litre=1.42, co2_per_litre=2.54):
     '''estimates fuel cost for the ICE HGV
   
       Parameters:
     - mpg: average miles per gallon for a 27t HGV (default ~9 mpg)
     - fuel_price_per_litre: average fuel cost in GBP (default ~£1.42/litre)

        Returns:
     - fuel_cost_mile: estimated fuel cost per mile
     - fuel_cost_sum: total fuel cost for the vehicle in the given dataframe'''
   
   
     miles_per_litre = mpg/4.54609 #conversion using 1 UK gallon (imperial) = 4.54609l
     tot_miles = dataframe.distance_tacho.max()

     #cost per mile calculation
     fuel_cost_mile = fuel_price_per_litre / miles_per_litre
     
     #fuel consumption
     fuel_consumed = tot_miles/miles_per_litre

     #total cost
     tot_fuel_cost = fuel_cost_mile*tot_miles

     #Total litres of diesel used
     litres_used = tot_miles / miles_per_litre

     #total CO2 emissions (kg)
     total_co2_emissions = litres_used * co2_per_litre


     return fuel_cost_mile, tot_fuel_cost, total_co2_emissions, fuel_consumed


#function for calculating energy costs

def energy_per_km(dataframe, energy_price_per_kWh=0.2703): 
    """
    Estimates energy cost based on simulated EV HGV data using only transition points.

    Parameters:
    - energy_price_per_kWh: The price of energy in GBP per kWh (default is 0.2703 for UK 2025).

    Returns:
    - total_energy_used_kWh: Total energy used in kWh.
    - total_energy_recovered_kWh: Total energy recovered in kWh.
    - net_energy_used: Net energy used in kWh (sum of used and recovered energy).
    - energy_cost_mile: Estimated energy cost per mile.
    - tot_energy_cost: Total energy cost for the simulated vehicle in the given dataframe.
    """
    dataframe = dataframe.copy()

    # Total miles driven (tacho distance)
    tot_miles = dataframe.distance_tacho.max()

    # Convert energy from joules to kWh
    joules_to_kWh = 1 / 3_600_000

    # Find rows where battery_energy changes
    transition_mask = dataframe['battery_energy'] != dataframe['battery_energy'].shift()
    transitions = dataframe[transition_mask]

    # Separate used vs recovered
    energy_used = transitions[transitions['battery_energy'] < 0]['battery_energy']
    energy_recovered = transitions[transitions['battery_energy'] > 0]['battery_energy']

    # Sum and convert to kWh
    total_energy_used_kWh = abs(energy_used.sum()) * joules_to_kWh
    total_energy_recovered_kWh = energy_recovered.sum() * joules_to_kWh

    # Net energy (used + recovered)
    net_energy_used = total_energy_used_kWh + total_energy_recovered_kWh

    # Costs
    energy_use_cost = total_energy_used_kWh * energy_price_per_kWh
    charging_cost = total_energy_recovered_kWh * energy_price_per_kWh
    tot_energy_cost = energy_use_cost + charging_cost

    # Cost per mile
    energy_cost_mile = energy_use_cost / tot_miles #doesn't account for charging cost to keep direct comparisons of energy use vs fuel use costs
    kwh_mile = total_energy_used_kWh / tot_miles

    print(f'Usage is {kwh_mile:.2f} kWh/mile')

    return total_energy_used_kWh, total_energy_recovered_kWh, net_energy_used, energy_cost_mile, tot_energy_cost



#function for directly comparing fuel and energy costs
def cost_comparison(EV_cost_mile, EV_cost_sum, fuel_cost_mile, fuel_cost_sum): 
    ''' compares EV and ICE vehicle running costs.
    Returns:
    - Difference per mile
    - Overall cost difference
    - A message indicating which is more cost-effective
    '''
    diff_per_mile = EV_cost_mile - fuel_cost_mile #EV hopefully cheaper than fuel, no. should be -ve
    overall_diff = EV_cost_sum - fuel_cost_sum

    if overall_diff > 0:
        EV_better_check = 'EV HGV would be more expensive to run'
    elif overall_diff == 0:
        EV_better_check = 'EV and ICE have equal running costs'
    else:
        EV_better_check = f'EV HGV is more cost effective and will reduce costs by ~£{diff_per_mile} per mile'

    return diff_per_mile, overall_diff, EV_better_check

    

###################################
#Geolocation Analysis
###################################

#Single Vehicle routes

#function to map the route of specified vehicle based on telematics data
def map_vehicle_route(dataframe):
    '''maps the vehcle route for the timewindow provided in the telematics data '''

     #converts dataframe to GeoDataFrame
    dataframe['geometry'] = dataframe.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(dataframe, geometry='geometry', crs='EPSG:4326')

    #converts CRS to Web Mercator for plotting with contextily
    gdf = gdf.to_crs(epsg=3857)

    # Plot the data with a basemap
    ax = gdf.plot(figsize=(12, 12), markersize=2, alpha=0.7)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik) #adds a map underlay for data to be plotted onto for contextualisation
    plt.title(f'Vehicle Location Map {dataframe['reg_number'].iloc[0]}', fontsize=15)
    plt.show() 

#################
#Fleet hotspot data

#function to find clusters for the full fleet dataset
def find_fleet_hotspots(dataframe, eps_meters=100, min_samples=5): #deafulat search for 100m with 5 neighbours
    """
    Identifies geographic hotspots across the entire fleet using DBSCAN clustering on GPS coordinates.

    Parameters:
    - eps_meters: float, clustering radius in meters
    - min_samples: int, minimum samples to form a cluster

    Returns:
    - hotspots_gdf: GeoDataFrame of clustered points
    - cluster_summary: pd.Series of cluster sizes
    """
    coords = dataframe[['longitude', 'latitude']].to_numpy()
    coords_rad = np.radians(coords)

    # Convert meters to radians (approximating for haversine with lat/lon for more accurate modelling
    kms_per_radian = 6371.0088 
    epsilon = eps_meters / 1000 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine')
    dataframe['cluster'] = db.fit_predict(coords_rad)

    # Keep only points assigned to a cluster (ignore noise: cluster == -1)
    clustered = dataframe[dataframe['cluster'] != -1].copy()

    # Convert to a GeoDataFrame
    clustered['geometry'] = clustered.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    hotspots_gdf = gpd.GeoDataFrame(clustered, geometry='geometry', crs='EPSG:4326')

    # Summarises the number of points in each cluster
    cluster_summary = hotspots_gdf['cluster'].value_counts()

    return hotspots_gdf, cluster_summary

#plots top hotspots as a psuedo heatmap
def plot_top_hotspots(hotspots_gdf, cluster_summary, top_n=20, title='Top Fleet Hotspots'):
    """
    Plots the top N hotspots on a basemap.

    Parameters:
    - hotspots_gdf: GeoDataFrame from find_fleet_hotspots
    - cluster_summary: Series from find_fleet_hotspots
    - top_n: int, number of top clusters to plot
    - title: str, plot title
    """
    if hotspots_gdf.empty:
        print("No hotspots to plot.")
        return

    # Get the IDs of the top N clusters
    top_clusters = cluster_summary.nlargest(top_n).index

    # Filter only points in top clusters
    top_hotspots_gdf = hotspots_gdf[hotspots_gdf['cluster'].isin(top_clusters)]

    # Project to Web Mercator for plotting
    top_hotspots_gdf = top_hotspots_gdf.to_crs(epsg=3857)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    top_hotspots_gdf.plot(ax=ax, column='cluster', categorical=True, legend=True,
                          markersize=10, cmap='tab20')

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title(title, fontsize=18)
    ax.set_axis_off()
    plt.show()




############################
# FULL ANALYTICS FUNCTION
############################

#Function to return the analytics for a specified vehjicle
def run_analytics(numberplate):
    """
    Runs and returns key analytics for a given HGV:
    - Average speed
    - Total distance
    - Daily distance stats
    - Downtime analysis
    - Fuel costs
    - Energy costs
    - Fuel vs energy cost comparison
    - CO2 emissions
    - SOC stats
    """
    df_hgv = HGV_data_return(numberplate)

    # Run analysis functions
    avg_speed = speed_avg(df_hgv)
    total_distance = travelled_dist(df_hgv)
    daily_distances, avg_daily_distance = daily_travel(df_hgv)
    daily_downtime, mean_downtime = downtime(df_hgv)
    soc_stats = soc_summary(df_hgv)

    fuel_cost_mile, fuel_cost_sum, co2_emissions, tot_fuel_consumed = ICE_fuel_costs(df_hgv)
    total_energy_used, total_energy_recovered, net_energy_used, energy_cost_mile, energy_cost_sum = energy_per_km(df_hgv)
    cost_diff_mile, overall_cost_diff, cost_msg = cost_comparison(
        energy_cost_mile, energy_cost_sum, fuel_cost_mile, fuel_cost_sum
    )

    Route_track = map_vehicle_route(df_hgv)

    # Compile results into a dictionary
    summary = {
        'Vehicle': numberplate,
        'Average Speed (moving only)': avg_speed,
        'tot fuel consumed': tot_fuel_consumed,
        'Total Distance Travelled (miles)': total_distance,
        'Average Daily Distance (miles)': avg_daily_distance,
        'Daily Distances': daily_distances.to_dict(), #converted from pandas series to python dictionary
        'Daily Downtime (hours)': daily_downtime.to_dict(), #converted from pandas series to python dictionary
        'Average Daily Downtime': mean_downtime,
        'SOC Summary': soc_stats,
        'total CO2 emissions (kg)': co2_emissions,
        'Fuel Cost Per Mile (GBP)': fuel_cost_mile,
        'Fuel Cost Total (GBP)': fuel_cost_sum,
        'EV Energy Used (kWh)': total_energy_used,
        'EV Energy Recovered (kWh)': total_energy_recovered,
        'EV Net Energy Used (kWh)': net_energy_used,
        'EV Energy Cost Per Mile (GBP)': energy_cost_mile,
        'EV Energy Cost Total (GBP)': energy_cost_sum,
        'Cost Comparison (EV vs ICE)': {
            'Cost Difference Per Mile (GBP)': cost_diff_mile,
            'Overall Cost Difference (GBP)': overall_cost_diff,
            'Cost Effectiveness Message': cost_msg
        }
    }

    return summary


#Test lines for running analytics with a specific singular vehicle (unhash to run)
#HGVA = HGV_data_return('xnumberplatex')
#Route_track = map_vehicle_route(HGVA)



#return list of all unique vehicles in fleet data
unique_reg_numbers = df_sorted['reg_number'].unique()
print(unique_reg_numbers)

#run analysis for each vehicle in fleet
for x in unique_reg_numbers:
    vehicle_summary = run_analytics(str(x))
    print(f'Analysis summary of vehicle {str(x)}: {vehicle_summary}')

#display the hotspots for the entire fleet (runs over full data set not per vehicle)
hotspot_gdf, hotspot_summary = find_fleet_hotspots(df_sorted, eps_meters=100, min_samples=5)
print(hotspot_summary)
plot_top_hotspots(hotspot_gdf, hotspot_summary, top_n=20)

