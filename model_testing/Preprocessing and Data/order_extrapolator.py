import pandas as pd
from datetime import timedelta
from math import radians, sin, cos, sqrt, asin

def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

# Calculate closest order in the last 15 minutes for the same store
def closest_order_in_last_15_min(row, df):
    current_time = row['business_timestamp_order_taken']
    current_store = row['store_number']  # Assuming 'store_number' is the column name
    lat1, lon1 = row['delivery_latitude'], row['delivery_longitude']

    # Filter orders in the last 15 minutes at the same store, excluding the current order
    recent_same_store_orders = df[
        (df['store_number'] == current_store) &
        (df['business_timestamp_order_taken'] >= current_time - timedelta(minutes=15)) &
        (df['business_timestamp_order_taken'] < current_time)
    ]
    
    if not recent_same_store_orders.empty:
        # Calculate distance for each recent order at the same store
        recent_same_store_orders['distance'] = recent_same_store_orders.apply(
            lambda x: haversine(lon1, lat1, x['delivery_longitude'], x['delivery_latitude']), axis=1
        )
        # Find the minimum distance
        closest_order_distance = recent_same_store_orders['distance'].min()
        return closest_order_distance
    else:
        return -1  # Default value if no orders in last 15 minutes at the same store

# Calculate closest order in the last 15 minutes for the same store
def closest_chain_orders_in_last_15_min(row, df):
    current_time = row['business_timestamp_order_taken']
    current_store = row['store_number']  # Assuming 'store_number' is the column name
    lat1, lon1 = row['delivery_latitude'], row['delivery_longitude']

    # Filter orders in the last 15 minutes at the same store, excluding the current order
    recent_same_store_orders = df[
        (df['store_number'] == current_store) &
        (df['business_timestamp_order_taken'] >= current_time - timedelta(minutes=15)) &
        (df['business_timestamp_order_taken'] < current_time)
    ]
    
    if not recent_same_store_orders.empty:
        # Calculate distance for each recent order at the same store
        recent_same_store_orders['distance'] = recent_same_store_orders.apply(
            lambda x: haversine(lon1, lat1, x['delivery_longitude'], x['delivery_latitude']) + x['closest_order_distance'], axis=1
        )
        # Find the minimum distance
        closest_order_distance = recent_same_store_orders['distance'].min()
        return closest_order_distance
    else:
        return -1  # Default value if no orders in last 15 minutes at the same store

def combine_and_process_datasets(input_excel_file):
    try:

        # Get the total number of rows in the Excel file
        total_rows = pd.read_excel(input_excel_file).shape[0]

        # Load the Excel file into a DataFrame and parse relevant columns as datetime
        dataset = pd.read_excel(input_excel_file,
                                parse_dates=['business_timestamp_order_taken', 'initial_estimated_delivery_timestamp', 'estimated_delivery_timestamp'],)
        print(f"Initial rows: {len(dataset)}")

        # Convert timestamp columns to datetime objects for order rank calculation
        dataset = dataset.dropna(subset=['out_the_door_timestamp']).dropna(subset=['to_the_door_timestamp'])
        dataset['out_the_door_timestamp'] = pd.to_datetime(dataset['out_the_door_timestamp'], format='mixed').dt.tz_localize(None)
        dataset['to_the_door_timestamp'] = pd.to_datetime(dataset['to_the_door_timestamp'], format='mixed').dt.tz_localize(None)

        # Group by 'out_the_door_timestamp' and assign order ranks within each group
        grouped = dataset.groupby('out_the_door_timestamp')['to_the_door_timestamp']
        dataset['order_rank'] = grouped.rank(method='min', ascending=True).astype(int)

        # Assign rank 0 to orders taken individually
        dataset.loc[grouped.transform('count') == 1, 'order_rank'] = 0

        # Convert timestamp columns to datetime objects for sector column and recent orders taken column
        dataset = dataset.dropna(subset=['business_timestamp_order_taken'])
        dataset['business_timestamp_order_taken'] = pd.to_datetime(dataset['business_timestamp_order_taken'], format='mixed').dt.tz_localize(None)

        dataset['num_orders_taken_last_ten_minutes_same_sector'] = 0
        dataset['num_orders_taken_last_ten_minutes'] = 0

        dataset['store_number'] = dataset['order_id'].apply(lambda x: x.split('|')[0])

        # Iterate through each row and count orders within ten minutes on the same day and hour from the same sector
        for index, row in dataset.iterrows():
            same_sector_rows = dataset[
                (dataset['business_timestamp_order_taken'].dt.date == row['business_timestamp_order_taken'].date()) &
                (dataset['business_timestamp_order_taken'].dt.hour == row['business_timestamp_order_taken'].hour) &
                (dataset['delivery_sector'] == row['delivery_sector']) & (dataset['store_number'] == row['store_number']) &
                (abs(dataset['business_timestamp_order_taken'] - row['business_timestamp_order_taken']) <= timedelta(minutes=10))
            ]
            num_orders_same_sector = len(same_sector_rows) - 1  # Exclude the current row itself
            if num_orders_same_sector < 0:
                num_orders_same_sector = 0
            dataset.at[index, 'num_orders_taken_last_ten_minutes_same_sector'] = num_orders_same_sector

            # Count orders within ten minutes on the same day and hour, regardless of sector
            same_hour_rows = dataset[
                (dataset['business_timestamp_order_taken'].dt.date == row['business_timestamp_order_taken'].date()) &
                (dataset['business_timestamp_order_taken'].dt.hour == row['business_timestamp_order_taken'].hour) & (dataset['store_number'] == row['store_number']) &
                (abs(dataset['business_timestamp_order_taken'] - row['business_timestamp_order_taken']) <= timedelta(minutes=10))
            ]
            num_orders_last_ten_minutes = len(same_hour_rows) - 1  # Exclude the current row itself
            if num_orders_last_ten_minutes < 0:
                num_orders_last_ten_minutes = 0
            dataset.at[index, 'num_orders_taken_last_ten_minutes'] = num_orders_last_ten_minutes

        # Sort DataFrame by 'business_timestamp_order_taken'
        dataset_sorted = dataset.sort_values(by='business_timestamp_order_taken')

        # Convert 'customer_quote_time' to Timedelta
        dataset_sorted['customer_quote_time_delta'] = pd.to_timedelta(dataset_sorted['customer_quote_time'], unit='m')

        # Convert to datetime and remove timezone information if present
        try:
            # Strip 'UTC' and remove fractional seconds
            dataset_sorted['initial_estimated_delivery_timestamp'] = dataset_sorted['initial_estimated_delivery_timestamp'].str.replace(" UTC", "").str.split(".").str[0]
            
            # Convert to datetime
            dataset_sorted['initial_estimated_delivery_timestamp'] = pd.to_datetime(dataset_sorted['initial_estimated_delivery_timestamp'], errors='coerce')
            
            # Convert timezone if needed (not required here)
            # dataset_sorted['initial_estimated_delivery_timestamp'] = dataset_sorted['initial_estimated_delivery_timestamp'].dt.tz_convert(None)
        except Exception as e:
            print(f"An error occurred while converting to datetime: {e}")

        # Perform the subtraction
        try:
            dataset_sorted['estimated_order_time'] = dataset_sorted['initial_estimated_delivery_timestamp'] - dataset_sorted['customer_quote_time_delta']
        except Exception as e:
            print(f"An error occurred: {e}")

        # Initialize a list to store the number of orders in progress
        orders_in_progress = []

        for idx, row in dataset_sorted.iterrows():
            start_time = row['estimated_order_time']
            end_time = row['initial_estimated_delivery_timestamp']

            # Count orders in progress
            count = dataset_sorted[(dataset_sorted['store_number'] == row['store_number']) & (((dataset_sorted['estimated_order_time'] <= end_time) & (dataset_sorted['estimated_order_time'] >= start_time)) |
                              ((dataset_sorted['initial_estimated_delivery_timestamp'] >= start_time) & (dataset_sorted['initial_estimated_delivery_timestamp'] <= end_time)) |
                              ((dataset_sorted['initial_estimated_delivery_timestamp'] >= end_time) & (dataset_sorted['estimated_order_time'] <= start_time))) 
                              ].shape[0]
            orders_in_progress.append(count)

        # Add a new column for the estimated orders in progress
        dataset_sorted['estimated_orders_in_progress'] = orders_in_progress

        # Replace NaN values with -1
        dataset_sorted['estimated_orders_in_progress'].fillna(-1, inplace=True)

        # Convert 'estimated_orders_in_progress' column to numeric to handle non-finite values
        dataset_sorted['estimated_orders_in_progress'] = pd.to_numeric(dataset_sorted['estimated_orders_in_progress'], errors='coerce')

        # Calculate the mean value of the 'estimated_orders_in_progress' column
        mean_orders_in_progress = dataset_sorted['estimated_orders_in_progress'].mean()

        # Fill in -1 values with the mean value
        dataset_sorted['estimated_orders_in_progress'].replace(-1, mean_orders_in_progress, inplace=True)


        # Add a new column for the closest order distance
        dataset_sorted['closest_order_distance'] = dataset_sorted.apply(lambda row: closest_order_in_last_15_min(row, dataset_sorted), axis=1)


        dataset_sorted['closest_order_distance'].replace([-1, ''], 10, inplace=True)

        dataset_sorted['closest_order_chain'] = dataset_sorted.apply(lambda row: closest_chain_orders_in_last_15_min(row, dataset_sorted), axis=1)

        max_closest_order_distance = dataset_sorted[dataset_sorted['closest_order_chain'] != -1]['closest_order_chain'].max()
        dataset_sorted['closest_order_chain'].replace([-1, ''], max_closest_order_distance, inplace=True)

        cols_to_keep = ['order_rank', 'num_orders_taken_last_ten_minutes', 'estimated_orders_in_progress', 'closest_order_distance', 'closest_order_chain', 'manager_on_duty', 'store_number']
        dataset_sorted = dataset_sorted[cols_to_keep]


        # Save the modified DataFrame to a new Excel file
        output_excel_file = input_excel_file.replace('.xlsx', '-processed.xlsx')
        dataset_sorted.to_excel(output_excel_file, index=False)

        return f"Data combined and processed. Output saved to {output_excel_file}."

    except Exception as e:
        return f"An error occurred: {e}"




input_excel_file = "order.xlsx"
print(combine_and_process_datasets(input_excel_file))
