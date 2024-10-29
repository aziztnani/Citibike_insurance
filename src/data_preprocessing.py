import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import month, count
import pyspark.sql.functions as F
import zipfile
import os
from sodapy import Socrata


# Preprocesses each CSV's main data and returns cleaned data and extracted station info
def preprocess_citibike_data(df):
    df = df.drop_duplicates()
    df = df.drop(columns=['ride_id'], errors='ignore')
    df['rideable_type'] = df['rideable_type'].fillna("unknown")
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce').fillna('1970-01-01').astype('datetime64[s]')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce').fillna('1970-01-01').astype('datetime64[s]')
    df['start_station_id'] = df['start_station_id'].fillna('unknown').astype(str)
    df['end_station_id'] = df['end_station_id'].fillna('unknown').astype(str)
    df['member_casual'] = df['member_casual'].fillna("unknown")
    
    # Extract station information
    start_stations = df[['start_station_id', 'start_station_name', 'start_lat', 'start_lng']].drop_duplicates()
    start_stations.columns = ['station_id', 'station_name', 'lat', 'lng']
    end_stations = df[['end_station_id', 'end_station_name', 'end_lat', 'end_lng']].drop_duplicates()
    end_stations.columns = ['station_id', 'station_name', 'lat', 'lng']
    
    # Combine and group to remove duplicates
    stations = pd.concat([start_stations, end_stations])
    stations = stations.groupby('station_id').agg({
        'station_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        'lat': 'median',
        'lng': 'median'
    }).reset_index()
    
    # Drop redundant columns
    df = df.drop(columns=['start_station_name', 'start_lat', 'start_lng', 
                          'end_station_name', 'end_lat', 'end_lng'], errors='ignore')
    
    return df, stations

# Iterates through each CSV file in nested zips and yields DataFrames for processing
def load_and_extract_csvs(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as main_zip:
        for inner_zip_name in main_zip.namelist():
            if inner_zip_name.endswith('.zip'):
                with main_zip.open(inner_zip_name) as inner_zip_file:
                    with zipfile.ZipFile(inner_zip_file) as inner_zip:
                        for csv_filename in inner_zip.namelist():
                            if csv_filename.endswith('.csv'):
                                with inner_zip.open(csv_filename) as csv_file:
                                    yield pd.read_csv(csv_file), csv_filename

# Updates the consolidated station DataFrame to keep unique station IDs
def update_stations(all_stations, new_stations):
    all_stations = pd.concat([all_stations, new_stations])
    return all_stations.groupby('station_id').agg({
        'station_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        'lat': 'median',
        'lng': 'median'
    }).reset_index()


# Main function to control the overall workflow
def convert_citibike_zip_to_parquet(zip_path, parquet_output_path):
    os.makedirs(parquet_output_path, exist_ok=True)
    all_stations = pd.DataFrame(columns=['station_id', 'station_name', 'lat', 'lng'])
    
    # Iterate over each extracted CSV
    for csv_file, csv_filename in load_and_extract_csvs(zip_path):
        df, new_stations = preprocess_citibike_data(csv_file)
        
        # Save each trip file as Parquet
        trips_path = os.path.join(parquet_output_path, 'trips')
        os.makedirs(trips_path, exist_ok=True)
        parquet_file_path = os.path.join(trips_path, f"{csv_filename.replace('.csv', '.parquet')}")
        df.to_parquet(parquet_file_path, index=False)
        
        # Update the consolidated stations DataFrame
        all_stations = update_stations(all_stations, new_stations)
    
    # Save the consolidated stations information
    stations_path = os.path.join(parquet_output_path, 'stations')
    os.makedirs(stations_path, exist_ok=True)
    stations_output_path = os.path.join(stations_path, 'stations.parquet')
    all_stations.to_parquet(stations_output_path, index=False)

    print('Data conversion to parquet was successful')

def load_parquet_in_spark(directory_path):
    # Initialize Spark session
    spark = SparkSession.builder.appName("CitiBikeAnalysis").getOrCreate()

    # List to hold individual DataFrames
    dataframes = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            # Load each Parquet file individually and add to the list
            df = spark.read.parquet(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    combined_df = dataframes[0]
    for df in dataframes[1:]:
        combined_df = combined_df.union(df)


    # Return the combined DataFrame
    return combined_df

def fetch_data(data_id, app_token, sql_query=None, batch_size=10000, nb_datapoints=50000):
    """
    Fetches data from a Socrata API using an SQL-like query and returns it as a DataFrame.

    Parameters:
    - data_id (str): The dataset ID to query.
    - app_token (str): The application token for the Socrata API.
    - sql_query (str, optional): The full SQL-like query to run against the dataset.
    - batch_size (int, optional): The number of records to retrieve in each batch (default is 10000).
    - nb_datapoints (int, optional): The total number of records to retrieve across all batches (default is 50000).

    Returns:
    - pd.DataFrame: A DataFrame containing the concatenated records from all batches.
    """
    # Initialize Socrata client
    client = Socrata("data.cityofnewyork.us", app_token)

    # Calculate the number of iterations based on the batch size and total data points
    num_batches = max(1, (nb_datapoints + batch_size - 1) // batch_size)

    # Initialize an empty DataFrame
    df_all = pd.DataFrame()

    # Retrieve data in batches with the specified SQL query
    for i in range(num_batches):
        offset = i * batch_size
        limit = min(batch_size, nb_datapoints - offset)
        
        # Add limit and offset to the SQL query if they're not present
        query_with_limit_offset = f"{sql_query} LIMIT {limit} OFFSET {offset}"
        
        # Fetch data with the SQL query
        response = client.get(data_id, query=query_with_limit_offset)

        # Convert batch to DataFrame and append
        df = pd.DataFrame.from_records(response)
        df_all = pd.concat([df_all, df])

    # Close the client
    client.close()

    return df_all
