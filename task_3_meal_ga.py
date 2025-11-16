# Student: [Your Name/ID]
# Course: [Your Course Name]
# Date: [Current Date]

"""
Task 3: Core Genetic Algorithm (mEAl)

This script implements the main genetic algorithm for meal optimization.
It brings together the data loading, meal evaluation, and the evolutionary
process to find a meal with the highest possible fitness score.
"""

import random
from typing import List, Tuple
from deap import base, creator, tools, algorithms
import numpy as np
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Data Loading and Meal Classes (from food_database.py and meal.py) ---
# Note: For submission, I've combined the necessary classes into this one file
# to make it self-contained and easy to run.

class Food:
    """Represents a single food item with its nutritional info."""
    def __init__(self, name, calories, macronutrients, micronutrients, fiber):
        self.name = name
        self.calories = calories
        self.macronutrients = macronutrients
        self.micronutrients = micronutrients
        self.fiber = fiber

def parse_value(value_str):
    """Helper to get numbers from strings like '581 kcal'."""
    if isinstance(value_str, (int, float)):
        return float(value_str)
    match = re.match(r"([0-9.]+)", str(value_str))
    return float(match.group(1)) if match else 0.0

def load_food_database_from_csv(filepath: str):
    """Loads all food data from our CSV file."""
    foods = {}
    with open(filepath, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)[1:]
        data_rows = {}
        for row in reader:
            if row:
                metric_name = row[0].split(' (')[0].lower().replace(' ', '_')
                data_rows[metric_name] = [parse_value(v) for v in row[1:]]
    for i, food_name in enumerate(headers):
        food_key = food_name.lower().replace(', ', '_').replace(',', '_')
        macronutrients = {k: data_rows.get(k, [])[i] for k in ["total_carbs", "protein", "saturated_fat", "unsaturated_fat"]}
        micronutrients = {k: data_rows.get(k, [])[i] for k in ["vitamin_a", "vitamin_b", "vitamin_c", "iron", "magnesium", "phosphorus", "potassium", "sodium", "zinc", "manganese", "selenium"]}
        foods[food_key] = Food(name=food_name, calories=data_rows.get("calories", [])[i], macronutrients=macronutrients, micronutrients=micronutrients, fiber=data_rows.get("fiber", [])[i])
    return foods

THRESHOLDS = {
    'calories_min': 500, 'calories_max': 800,
    'water_soluble_vitamins_min': 100,
    'fat_soluble_vitamins_min': 50, 'fat_soluble_vitamins_max': 200,
    'safe_minerals_min': 1000,
    'unsafe_minerals_min': 10, 'unsafe_minerals_max': 500
}

class Meal:
    """Represents a meal, calculates its nutrients and fitness score."""
    def __init__(self, foods: List[Tuple[Food, float]]):
        self.foods = foods
        self.calories = 0.0
        self.water_soluble_vitamins = 0.0
        self.fat_soluble_vitamins = 0.0
        self.safe_minerals = 0.0
        self.unsafe_minerals = 0.0
        self._calculate_totals()

    def _calculate_totals(self):
        """Sums up all nutrients for the meal."""
        for food, weight in self.foods:
            weight_100g = weight * 10
            self.calories += food.calories * weight_100g
            for vit, amount in food.micronutrients.items():
                if vit in ['vitamin_c', 'vitamin_b']: self.water_soluble_vitamins += amount * weight_100g
                elif vit in ['vitamin_a', 'vitamin_d', 'vitamin_e', 'vitamin_k']: self.fat_soluble_vitamins += amount * weight_100g
            for mineral, amount in food.micronutrients.items():
                if mineral in ['sodium']: self.unsafe_minerals += amount * weight_100g
                elif mineral in ['iron', 'magnesium', 'zinc', 'selenium']: self.safe_minerals += amount * weight_100g

    def calculate_fitness(self) -> float:
        """Calculates the meal's score based on nutritional targets."""
        fitness = 0.0
        # Score calories
        cal = self.calories
        if THRESHOLDS['calories_min'] <= cal <= THRESHOLDS['calories_max']: fitness += 1.0
        elif cal < THRESHOLDS['calories_min']: fitness -= ((THRESHOLDS['calories_min'] - cal) / THRESHOLDS['calories_min']) * 2.0
        else: fitness -= ((cal - THRESHOLDS['calories_max']) / THRESHOLDS['calories_max']) * 5.0
        # Score vitamins and minerals
        fitness += min(self.water_soluble_vitamins / THRESHOLDS['water_soluble_vitamins_min'], 1.0)
        fsv, fsv_min, fsv_max = self.fat_soluble_vitamins, THRESHOLDS['fat_soluble_vitamins_min'], THRESHOLDS['fat_soluble_vitamins_max']
        if fsv_min <= fsv <= fsv_max: fitness += 1.0
        elif fsv < fsv_min: fitness += fsv / fsv_min
        else: fitness += max(0.0, 1.0 - (fsv - fsv_max) / (fsv_max * 2))
        fitness += min(self.safe_minerals / THRESHOLDS['safe_minerals_min'], 1.0)
        um, um_min, um_max = self.unsafe_minerals, THRESHOLDS['unsafe_minerals_min'], THRESHOLDS['unsafe_minerals_max']
        if um_min <= um <= um_max: fitness += 1.0
        elif um < um_min: fitness += um / um_min
        else: fitness += max(0.0, 1.0 - (um - um_max) / (um_max * 2))
        # Penalize heavy meals and reward variety
        if self.get_total_weight() > 1.2: fitness -= (self.get_total_weight() - 1.2) * 5
        fitness += 0.1 * sum(1 for _, w in self.foods if w >= 0.05)
        return fitness

    def get_total_weight(self) -> float: return sum(w for _, w in self.foods)
    def get_calories(self) -> float: return self.calories

# --- Genetic Algorithm Implementation ---

# Load the food database to be used by the GA
FOOD_DATABASE = load_food_database_from_csv('training_data.csv')

def create_random_meal() -> List[Tuple[Food, float]]:
    """Creates a random meal to initialize the population."""
    num_foods = random.randint(1, 4)
    foods = random.sample(list(FOOD_DATABASE.values()), num_foods)
    return [(food, random.uniform(0.1, 0.5)) for food in foods]

def evaluate_meal(individual: List[Tuple[Food, float]]) -> Tuple[float,]:
    """Fitness function: evaluates a meal and returns its score."""
    return (Meal(individual).calculate_fitness(),)

def mutate_meal(individual: List[Tuple[Food, float]]) -> Tuple[List[Tuple[Food, float]],]:
    """Mutation operator: randomly changes a food or its quantity."""
    if random.random() < 0.5 and individual:
        # Change a food item
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = (random.choice(list(FOOD_DATABASE.values())), individual[idx][1])
    elif individual:
        # Change a food's quantity
        idx = random.randint(0, len(individual) - 1)
        new_quantity = random.uniform(0.1, 0.5)
        individual[idx] = (individual[idx][0], new_quantity)
    return (individual,)

def crossover_meals(ind1, ind2):
    """Crossover operator: swaps parts of two meals to create offspring."""
    if len(ind1) > 0 and len(ind2) > 0:
        point = random.randint(1, min(len(ind1), len(ind2)))
        ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    return ind1, ind2

def evolve_meal(population_size=100, generations=50, cxpb=0.7, mutpb=0.2, min_fitness_threshold=5.0) -> Meal:
    """Main function to run the genetic algorithm."""
    # Setup DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_random_meal)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_meal)
    toolbox.register("mate", crossover_meals)
    toolbox.register("mutate", mutate_meal)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population and statistics
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the evolution
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        
        record = stats.compile(population)
        print(f"Generation {gen}: Avg Fitness={record['avg']:.2f}, Max Fitness={record['max']:.2f}")
        
        # Stop if a good enough solution is found
        if record['max'] >= min_fitness_threshold:
            print(f"Termination condition met at generation {gen}.")
            break

    # Return the best meal found
    best_individual = tools.selBest(population, k=1)[0]
    return Meal(best_individual)

# --- Main Execution ---
if __name__ == "__main__":
    # This function checks if the data is split, and if not, it runs the split.
    def split_data_if_needed(file_path='foods.csv', train_path='training_data.csv', test_path='testing_data.csv'):
        try:
            open(train_path)
        except FileNotFoundError:
            print("Training data not found. Splitting data...")
            df = pd.read_csv(file_path, index_col=0)
            train_foods, test_foods = train_test_split(df.columns.tolist(), test_size=0.3, random_state=42)
            df[train_foods].to_csv(train_path)
            df[test_foods].to_csv(test_path)
    
    split_data_if_needed()

    print("\n--- Running Core mEAl Algorithm (Task 3) ---")
    # Using parameters that were found to be effective in finetuning (Task 5)
    best_meal_found = evolve_meal(
        population_size=250,
        generations=200,
        mutpb=0.2,
        cxpb=0.6
    )
    
    # Print the details of the best meal found
    print("\n--- Best Meal Found by Core mEAl ---")
    for food, quantity in best_meal_found.foods:
        print(f"- {food.name}: {quantity:.2f} kg")
    print(f"Fitness Score: {best_meal_found.calculate_fitness():.4f}")
    print(f"Total Calories: {best_meal_found.get_calories():.2f} kcal")