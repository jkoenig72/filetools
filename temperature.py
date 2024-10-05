#!/usr/bin/python3

import os
import requests
import json

# Set up the Home Assistant server details
HASS_API_BASE_URL = 'http://192.168.10.22:8123'
HASS_API_TOKEN = 'token'

# Define the sensor entity ID
SENSOR_ENTITY_ID = 'sensor.esphome_web_c09641_temperature'

# Get the JSON file name from the environment variable
json_file = os.environ.get('TEMP_JSON')

if not json_file:
    print('Error: TEMP_JSON environment variable is not set.')
    exit(1)
# Make a GET request to Home Assistant API to fetch the sensor state
headers = {
    'Authorization': f'Bearer {HASS_API_TOKEN}',
    'Content-Type': 'application/json',
}
url = f'{HASS_API_BASE_URL}/api/states/{SENSOR_ENTITY_ID}'
response = requests.get(url, headers=headers)
sensor_data = response.json()

# Extract the sensor value from the response
sensor_value = sensor_data['state']

# Create a dictionary with the sensor value
data = {
    'temp': sensor_value
}

# Write the data to the JSON file
with open(json_file, 'w') as f:
    json.dump(data, f)

print(f'Sensor value written to {json_file}.')
