#!/usr/bin/python3

import os
import requests
import json

# Set up the Home Assistant server details
HASS_API_BASE_URL = 'http://192.168.10.22:8123'  # Base URL for Home Assistant API
HASS_API_TOKEN = 'token'  # Authentication token for API access

# Define the sensor entity ID
SENSOR_ENTITY_ID = 'sensor.esphome_web_c09641_temperature'  # Entity ID of the temperature sensor

# Get the JSON file name from the environment variable
json_file = os.environ.get('TEMP_JSON')

if not json_file:
    print('Error: TEMP_JSON environment variable is not set.')
    exit(1)

# Make a GET request to Home Assistant API to fetch the sensor state
headers = {
    'Authorization': f'Bearer {HASS_API_TOKEN}',  # Include the authentication token in the headers
    'Content-Type': 'application/json',  # Specify that the content type is JSON
}
url = f'{HASS_API_BASE_URL}/api/states/{SENSOR_ENTITY_ID}'  # Construct the full URL for the API endpoint
response = requests.get(url, headers=headers)  # Send the GET request to the API
sensor_data = response.json()  # Parse the JSON response to extract sensor data

# Extract the sensor value from the response
sensor_value = sensor_data['state']  # Get the current state (value) of the sensor

# Create a dictionary with the sensor value
data = {
    'temp': sensor_value  # Store the sensor value in a dictionary under the key 'temp'
}

# Write the data to the JSON file
with open(json_file, 'w') as f:
    json.dump(data, f)  # Serialize the dictionary and write it to the specified JSON file

print(f'Sensor value written to {json_file}.')  # Output confirmation