from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity

# Connect to the Azure Table Storage service
import json
table_service = TableService(account_name='demotestsuneel', account_key='')

#sample json

sampleobj={}

# Define the table name


table_name = 'test'

# Create the table if it does not exist
table_service.create_table(table_name)

json_str = json.dumps(DexIncidents)
item = json.loads(json_str, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

# Upload the entity to the table
table_service.insert_entity(table_name, item)
