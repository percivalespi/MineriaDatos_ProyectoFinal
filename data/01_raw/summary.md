---
dataset_info:
  features:
  - name: ID
    dtype: string
  - name: Delivery_person_ID
    dtype: string
  - name: Delivery_person_Age
    dtype: float64
  - name: Delivery_person_Ratings
    dtype: float64
  - name: Restaurant_latitude
    dtype: float64
  - name: Restaurant_longitude
    dtype: float64
  - name: Delivery_location_latitude
    dtype: float64
  - name: Delivery_location_longitude
    dtype: float64
  - name: Order_Date
    dtype: string
  - name: Time_Orderd
    dtype: string
  - name: Time_Order_picked
    dtype: string
  - name: Weather_conditions
    dtype: string
  - name: Road_traffic_density
    dtype: string
  - name: Vehicle_condition
    dtype: int64
  - name: Type_of_order
    dtype: string
  - name: Type_of_vehicle
    dtype: string
  - name: multiple_deliveries
    dtype: float64
  - name: Festival
    dtype: string
  - name: City
    dtype: string
  - name: Time_taken (min)
    dtype: int64
  splits:
  - name: train
    num_bytes: 7081046.7646323275
    num_examples: 36467
  - name: test
    num_bytes: 1770310.235367673
    num_examples: 9117
  download_size: 2209801
  dataset_size: 8851357.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
