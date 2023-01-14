import requests
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from datetime import datetime

# Connect to Azure Table
table_service = TableService(account_name='your_account_name', account_key='your_account_key')

# Define the table name
table_name = 'api_calls'

# Create the table if it doesn't exist
if not table_service.exists(table_name):
    table_service.create_table(table_name)

# List of API URLs
api_urls = ['https://api1.example.com/data', 'https://api2.example.com/data', 'https://api3.example.com/data']

for url in api_urls:
    # Make the API request
    start_time = datetime.now()
    response = requests.get(url)
    end_time = datetime.now()

    # Calculate the response time
    response_time = (end_time - start_time).microseconds

    # Prepare the data for insertion into Azure Table
    data = {'PartitionKey': 'api_calls', 'RowKey': url, 'response_time': response_time}

    # Insert the data into Azure Table
    table_service.insert_or_replace_entity(table_name, entities.Entity.from_dict(data))

    print(f'Response time of API call to {url} recorded into Azure Table {table_name}')
