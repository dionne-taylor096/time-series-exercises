#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import os
import pandas as pd


# In[ ]:


def fetch_data(url):
    data = []

    while url:
        response = requests.get(url)
        json_data = response.json()
        data.extend(json_data["results"])
        url = json_data["next"]

    return data

def grab_csv_data(api_url, output_file):
    if not os.path.exists(output_file):
        response = requests.get(api_url)

        if response.status_code == 200:
            csv_data = response.text
            with open(output_file, 'w') as f:
                f.write(csv_data)
            print(f"CSV data saved to {output_file}")
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    return pd.read_csv(output_file)

def create_starships_dataframe():
    base_url = "https://swapi.dev/api/"
    starships_url = f"{base_url}starships/"

    starships_data = fetch_data(starships_url)
    starships_df = pd.DataFrame(starships_data)

    return starships_df

# Call the function to create the people dataframe
# starships_df = create_starships_dataframe()
# Print the first few rows of the dataframe

def create_people_dataframe():
    base_url = "https://swapi.dev/api/"
    people_url = f"{base_url}people/"

    people_data = fetch_data(people_url)
    people_df = pd.DataFrame(people_data)
    people_df.to_csv('people.csv', index=False)
    return people_df

# Call the function to create the people dataframe
#people_df = create_people_dataframe()
# Print the first few rows of the dataframe

def create_planets_dataframe():
    base_url = "https://swapi.dev/api/"
    planets_url = f"{base_url}planets/"

    planets_data = fetch_data(planets_url)
    planets_df = pd.DataFrame(planets_data)
    planets_df.to_csv('planets.csv', index=False)
    return planets_df

# Call the function to create the people dataframe
#planets_df = create_planets_dataframe()


