import numpy as np
import json
from itertools import islice
import re
import os
import ast
import random
from utils.LLM import LLM_api

model_name = "deepseek-ai/DeepSeek-V3"
filepath2 = 'benchmark/knapsack/kp.json'
Sector = ['Resource Allocation in Cloud Computing', 'Inventory Management for Retail', 'Military Equipment Deployment',
          'Energy Grid Optimization', 'Software Feature Prioritization', 'Wildlife Conservation Planning',
          'Conference Scheduling', 'Sports Team Formation', 'Academic Course Selection',
          'Medical Treatment Prioritization', 'Ship Cargo Loading', 'Digital Storage Allocation',
          'Adventure Backpack Packing', 'Music Festival Scheduling', 'Home Renovation Budgeting', 'Game Loot Selection',
          'Movie Production Budgeting', 'Vaccine Distribution Planning', 'Crop Planting Strategy',
          'Space Mission Payload Selection', 'Charitable Donation Allocation', 'Movie Theater Snack Menu Design',
          'Product Line Optimization', 'Cybersecurity Patch Deployment', 'Academic Research Grant Allocation',
          'Travel Itinerary Planning', 'Industrial Equipment Maintenance', 'Data Center Server Procurement',
          'Construction Material Selection', 'Movie Merchandise Production', 'Emergency Relief Supplies Packing',
          'Fashion Retail Store Stocking', 'Film Festival Programming', 'Logistics Truck Loading',
          'AI Model Feature Inclusion', 'Custom Jewelry Design', 'Athlete Nutrition Planning',
          'Vehicle Fleet Optimization', 'Gift Wrapping Efficiency', 'Job Interview Scheduling',
          'Warehouse Rack Space Allocation', 'Chemical Compound Selection', 'Smartphone Component Assembly',
          'E-commerce Order Shipping', 'Tourist Attraction Planning', 'Luxury Car Customization',
          'Music Album Track Listing', 'VIP Event Guest Selection', 'Ancient Artifact Acquisition',
          'Athlete Sponsorship Allocation', 'Startup Hiring Strategy', 'Catering Menu Planning',
          'Real Estate Investment', 'Pharmaceutical Drug Development', 'Movie Script Selection',
          'Startup Office Layout Design', 'Wedding Catering Menu Design', 'Digital Art Collection Curation',
          'R&D Investment Allocation', 'Disaster Relief Packing', 'Luxury Watch Collection Curation',
          'AI Training Data Selection', 'Urban Park Amenity Planning', 'Athlete Meal Prep Planning',
          'Jewelry Thief Heist', 'Spacecraft Payload Optimization', 'Luxury Hotel Menu Design',
          'Auction House Art Acquisition', 'Formula 1 Pit Stop Tool Selection', 'Supply Chain Container Loading',
          'Hiking Gear Selection', 'Film Equipment Rental', 'Pharmacy Shelf Stocking', 'Personal Library Curation',
          'Restaurant Menu Design', 'Luxury Yacht Interior Design', 'Fashion Week Model Selection',
          'Cargo Plane Loading', 'Museum Exhibit Transport', 'Backcountry Research Kit Assembly',
          'Smartphone Feature Integration', 'Emergency Shelter Supply Packing']

with open("benchmark/knapsack/prompt.json", "r", encoding="utf-8") as f:
    data_prompt = json.load(f)
with open("benchmark/knapsack/seed.json", "r", encoding="utf-8") as f:
    data_seed_prompt = json.load(f)


def generate_uniform_numbers(n):
    return [round(random.uniform(0, 1), 4) for _ in range(n)]


# Initialize LLM instance
llm = LLM_api(
    model=model_name,
    max_tokens=8000,
    temperature=1.3
)

for key in ["knapsack"]:
    problem_type = key

    num_requests = 1
    for i in range(num_requests - 1, num_requests + 5):
        Sector_process = ','.join(Sector)
        user_input = data_prompt[key] + data_seed_prompt[
            key] + "The scenario title for each generated example must not duplicate any in the following list, meaning each application scenario must be distinct and cannot be the same as or similar to the ones in the list below:" + Sector_process + " ,Note that the language style of different examples must vary, and it should not be apparent that they were asked by the same person."
        number = 5
        user_input = user_input.format(number=number)

        # Use LLM class to get response
        result = llm.get_text(content=user_input)

        if result:  # Check if response is not empty
            print(f"Total tokens used: {llm.get_token()}")
            resulttt = result.split('&&&')
            for result in resulttt:
                match = re.search(r"##(.*?)##", result)
                title = match.group(1) if match else None
                if title == None:
                    print("null_exist;")
                else:
                    Sector.append(title)
                    text_without_title = re.sub(r"##.*?##", "", result).strip()
                    responses = []
                    responses.append(text_without_title)
                    responses = [response.replace("\n", "") for response in responses]
                    responses = responses[0]
                    n_item = random.randint(50, 100)
                    item_weight = generate_uniform_numbers(n_item)
                    item_value = generate_uniform_numbers(n_item)
                    knapsack_capacity = random.randint(5, 20)
                    desc_merge = responses.replace('<item_weight>', str(item_weight))
                    desc_merge = desc_merge.replace('<item_value>', str(item_value))
                    desc_merge = desc_merge.replace('<knapsack_capacity>', str(knapsack_capacity))
                    add_data = {
                        "title": title,
                        "desc_split": responses,
                        "desc_merge": desc_merge,
                        "data_template": {
                            "item_weight": item_weight,
                            "item_value": item_value,
                            "knapsack_capacity": knapsack_capacity
                        },
                        "label": problem_type
                    }
                    try:
                        with open(filepath2, "r") as file:
                            data_list = json.load(file)
                    except (FileNotFoundError, json.JSONDecodeError):
                        data_list = []
                    data_list.append(add_data)

                    with open(filepath2, "w") as file:
                        json.dump(data_list, file, indent=4)

            print(f"API call {i + 1} successful, results written to file")
        else:
            print(f"API call {i + 1} failed, empty response returned")
