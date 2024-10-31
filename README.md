Citybike Insurance Case Study

The following repository concerns the analysis of Citibike Data, NYPD Data and Bicycle count data for the propose of developing insurance products suitable for Citibike.

We will first start by processing the citibike data and save it to parquet files for faster analysis.
Second we will analyse Citybike Data, to etract useful information and patterns for insurance products.
Third we analyse the NYPD Data using an API to fetch data.
Fourth We Analyze Risk, Accident Risk.
Fifth we use ARIMA to prove that future demand of Citibike is predictable, which is useful for pricing purposes.


In order to execute the code:
1) Download the required data (2023-citibike-tripdata.zip, 2022-citibike-tripdata.zip, 2021-citibike-tripdata.zip) from https://s3.amazonaws.com/tripdata/index.html
2) Save it in the folder data/raw/
3) run the notebook data_preprocessing_onetime
4) run the notebook data_analysis
5) run the notebook citibike_forecast 
