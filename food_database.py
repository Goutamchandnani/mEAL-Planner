# Student: [Your Name/ID]
# Course: [Your Course Name]
# Date: [Current Date]

"""
food_database.py

This file sets up our food data. It defines what a 'Food' is and how to load
all our food items from a CSV file into a usable database.
"""

import csv
import re
from typing import Dict

class Food:
    """
    Represents a single food item.
    Stores its name, calories, macronutrients (carbs, protein, fats),
    micronutrients (vitamins, minerals), and fiber.
    All values are per 100 grams.
    """
    def __init__(self, name: str, calories: float, macronutrients: dict, micronutrients: dict, fiber: float):
        self.name = name
        self.calories = calories  # kcal per 100g
        self.macronutrients = macronutrients  # dict: total_carbs, protein, saturated_fat, unsaturated_fat (g/100g)
        self.micronutrients = micronutrients  # dict: various vitamins and minerals (mg/100g)
        self.fiber = fiber  # grams per 100g

def parse_value(value_str) -> float:
    """
    Extracts a numeric value from a string, handling units like '581 kcal'.
    Returns 0.0 if no number is found.
    """
    if isinstance(value_str, (int, float)):
        return float(value_str)
    match = re.match(r"([0-9.]+)", str(value_str))
    if match:
        return float(match.group(1))
    return 0.0

def load_food_database_from_csv(filepath: str) -> Dict[str, Food]:
    """
    Loads food data from a CSV file into a dictionary of Food objects.
    The CSV should have metrics in the first column and food names in the first row.
    """
    foods = {}
    with open(filepath, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)[1:]  # Get food names from the first row, skip empty cell
        
        data_rows = {}
        for row in reader:
            if row:
                # Clean metric name (e.g., "Calories (kcal)" -> "calories")
                metric_name = row[0].split(' (')[0].lower().replace(' ', '_')
                data_rows[metric_name] = [parse_value(v) for v in row[1:]]

    # Process each food column to create Food objects
    for i, food_name in enumerate(headers):
        # Create a clean key for the food dictionary
        food_key = food_name.lower().replace(', ', '_').replace(',', '_')
        
        # Gather macronutrient data
        macronutrients = {
            "total_carbs": data_rows.get("total_carbs", [])[i],
            "protein": data_rows.get("protein", [])[i],
            "saturated_fat": data_rows.get("saturated_fat", [])[i],
            "unsaturated_fat": data_rows.get("unsaturated_fat", [])[i],
        }
        # Gather micronutrient data
        micronutrients = {
            "vitamin_a": data_rows.get("vitamin_a", [])[i],
            "vitamin_b": data_rows.get("vitamin_b", [])[i],
            "vitamin_c": data_rows.get("vitamin_c", [])[i],
            "iron": data_rows.get("iron", [])[i],
            "magnesium": data_rows.get("magnesium", [])[i],
            "phosphorus": data_rows.get("phosphorus", [])[i],
            "potassium": data_rows.get("potassium", [])[i],
            "sodium": data_rows.get("sodium", [])[i],
            "zinc": data_rows.get("zinc", [])[i],
            "manganese": data_rows.get("manganese", [])[i],
            "selenium": data_rows.get("selenium", [])[i],
        }
        
        # Create and store the Food object
        food = Food(
            name=food_name,
            calories=data_rows.get("calories", [])[i],
            macronutrients=macronutrients,
            micronutrients=micronutrients,
            fiber=data_rows.get("fiber", [])[i]
        )
        foods[food_key] = food
        
    return foods

# Our main food database, loaded from 'training_data.csv'.
# This is used by all other parts of the meal optimization system.
FOOD_DATABASE = load_food_database_from_csv('training_data.csv')