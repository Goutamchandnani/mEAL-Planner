# Student: [Your Name/ID]
# Course: [Your Course Name]
# Date: [Current Date]

"""
Task 5: Finetuning Genetic Algorithm Parameters

This script systematically tests different parameters for our genetic algorithm
to find the combination that produces the best results. This helps us
optimize the algorithm's performance.
"""

import random
from typing import List, Tuple
from deap import base, creator, tools, algorithms
import numpy as np
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Combined Data Loading and Meal Classes ---
# For submission, all necessary classes are included in this file.

class Food:
    """Represents a single food item with its nutritional info."""
    def __init__(self, name, calories, macronutrients, micronutrients, fiber):
        self.name, self.calories, self.macronutrients, self.micronutrients, self.fiber = name, calories, macronutrients, micronutrients, fiber

def parse_value(value_str):
    """Helper to get numbers from strings like '581 kcal'."""
    if isinstance(value_str, (int, float)): return float(value_str)
    match = re.match(r"([0-9.]+)", str(value_str))
    return float(match.group(1)) if match else 0.0

def load_food_database_from_csv(filepath: str):
    """Loads all food data from our CSV file."""
    foods = {}
    with open(filepath, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)[1:]
        data_rows = {row[0].split(' (')[0].lower().replace(' ', '_'): [parse_value(v) for v in row[1:]] for row in reader if row}
    for i, food_name in enumerate(headers):
        food_key = food_name.lower().replace(', ', '_').replace(',', '_')
        macronutrients = {k: data_rows.get(k, [0]*len(headers))[i] for k in ["total_carbs", "protein", "saturated_fat", "unsaturated_fat"]}
        micronutrients = {k: data_rows.get(k, [0]*len(headers))[i] for k in ["vitamin_a", "vitamin_b", "vitamin_c", "iron", "magnesium", "phosphorus", "potassium", "sodium", "zinc", "manganese", "selenium"]}
        foods[food_key] = Food(name=food_name, calories=data_rows.get("calories", [0]*len(headers))[i], macronutrients=macronutrients, micronutrients=micronutrients, fiber=data_rows.get("fiber", [0]*len(headers))[i])
    return foods

THRESHOLDS = {'calories_min': 500, 'calories_max': 800, 'water_soluble_vitamins_min': 100, 'fat_soluble_vitamins_min': 50, 'fat_soluble_vitamins_max': 200, 'safe_minerals_min': 1000, 'unsafe_minerals_min': 10, 'unsafe_minerals_max': 500}

class Meal:
    """Represents a meal, calculates its nutrients and fitness score."""
    def __init__(self, foods: List[Tuple[Food, float]]):
        self.foods = foods
        self.calories = self.water_soluble_vitamins = self.fat_soluble_vitamins = self.safe_minerals = self.unsafe_minerals = 0.0
        self._calculate_totals()

    def _calculate_totals(self):
        """Sums up all nutrients for the meal."""
        for food, weight in self.foods:
            weight_100g = weight * 10
            self.calories += food.calories * weight_100g
            for k, v in food.micronutrients.items():
                if k in ['vitamin_c', 'vitamin_b']: self.water_soluble_vitamins += v * weight_100g
                elif k in ['vitamin_a', 'vitamin_d', 'vitamin_e', 'vitamin_k']: self.fat_soluble_vitamins += v * weight_100g
                elif k in ['sodium']: self.unsafe_minerals += v * weight_100g
                elif k in ['iron', 'magnesium', 'zinc', 'selenium']: self.safe_minerals += v * weight_100g

    def calculate_fitness(self) -> float:
        """Calculates the meal's score based on nutritional targets."""
        fitness = 0.0
        cal = self.calories
        if THRESHOLDS['calories_min'] <= cal <= THRESHOLDS['calories_max']: fitness += 1.0
        elif cal < THRESHOLDS['calories_min']: fitness -= ((THRESHOLDS['calories_min'] - cal) / THRESHOLDS['calories_min']) * 2.0
        else: fitness -= ((cal - THRESHOLDS['calories_max']) / THRESHOLDS['calories_max']) * 5.0
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
        if self.get_total_weight() > 1.2: fitness -= (self.get_total_weight() - 1.2) * 5
        fitness += 0.1 * sum(1 for _, w in self.foods if w >= 0.05)
        return fitness

    def get_total_weight(self) -> float: return sum(w for _, w in self.foods)

# --- Genetic Algorithm and Finetuning ---
FOOD_DATABASE = load_food_database_from_csv('training_data.csv')

def create_random_meal(): return [(food, random.uniform(0.1, 0.5)) for food in random.sample(list(FOOD_DATABASE.values()), random.randint(1, 4))]
def evaluate_meal(individual): return (Meal(individual).calculate_fitness(),)
def mutate_meal(individual):
    if random.random() < 0.5 and individual: individual[random.randint(0, len(individual) - 1)] = (random.choice(list(FOOD_DATABASE.values())), individual[random.randint(0, len(individual) - 1)][1])
    elif individual: individual[random.randint(0, len(individual) - 1)] = (individual[random.randint(0, len(individual) - 1)][0], random.uniform(0.1, 0.5))
    return (individual,)
def crossover_meals(ind1, ind2):
    if ind1 and ind2:
        point = random.randint(1, min(len(ind1), len(ind2)))
        ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    return ind1, ind2

def evolve_meal(population_size=100, generations=50, cxpb=0.7, mutpb=0.2):
    """Runs the genetic algorithm with a given set of parameters."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_random_meal)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_meal)
    toolbox.register("mate", crossover_meals)
    toolbox.register("mutate", mutate_meal)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=population_size)
    # Run the GA without printing stats for each generation to speed up finetuning
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, verbose=False)
    return Meal(tools.selBest(pop, k=1)[0])

def run_experiment(pop_size, generations, mut_prob, co_prob, num_trials=5):
    """Runs the GA multiple times for a set of parameters and returns the average fitness."""
    scores = [evolve_meal(pop_size, generations, co_prob, mut_prob).calculate_fitness() for _ in range(num_trials)]
    return np.mean(scores)

def finetune_parameters():
    """Systematically tests different parameter values to find the best combination."""
    # Starting parameters
    best_params = {'pop_size': 150, 'generations': 200, 'mut_prob': 0.2, 'co_prob': 0.7}
    best_fitness = -float('inf')

    print("--- Starting Parameter Finetuning for mEAl ---")

    # 1. Tune Population Size
    print("\n1. Tuning Population Size...")
    for ps in [50, 100, 150, 200, 250]:
        avg_fitness = run_experiment(ps, best_params['generations'], best_params['mut_prob'], best_params['co_prob'])
        print(f"  Testing pop_size={ps}: Avg Fitness = {avg_fitness:.4f}")
        if avg_fitness > best_fitness:
            best_fitness, best_params['pop_size'] = avg_fitness, ps
    print(f"  Best Population Size: {best_params['pop_size']}")

    # 2. Tune Number of Generations
    print("\n2. Tuning Number of Generations...")
    for gen in [50, 100, 200, 300, 400]:
        avg_fitness = run_experiment(best_params['pop_size'], gen, best_params['mut_prob'], best_params['co_prob'])
        print(f"  Testing generations={gen}: Avg Fitness = {avg_fitness:.4f}")
        if avg_fitness > best_fitness:
            best_fitness, best_params['generations'] = avg_fitness, gen
    print(f"  Best Number of Generations: {best_params['generations']}")

    # 3. Tune Mutation Probability
    print("\n3. Tuning Mutation Probability...")
    for mp in [0.05, 0.1, 0.2, 0.3, 0.4]:
        avg_fitness = run_experiment(best_params['pop_size'], best_params['generations'], mp, best_params['co_prob'])
        print(f"  Testing mut_prob={mp}: Avg Fitness = {avg_fitness:.4f}")
        if avg_fitness > best_fitness:
            best_fitness, best_params['mut_prob'] = avg_fitness, mp
    print(f"  Best Mutation Probability: {best_params['mut_prob']}")

    # 4. Tune Crossover Probability
    print("\n4. Tuning Crossover Probability...")
    for cp in [0.6, 0.7, 0.8, 0.9]:
        avg_fitness = run_experiment(best_params['pop_size'], best_params['generations'], best_params['mut_prob'], cp)
        print(f"  Testing co_prob={cp}: Avg Fitness = {avg_fitness:.4f}")
        if avg_fitness > best_fitness:
            best_fitness, best_params['co_prob'] = avg_fitness, cp
    print(f"  Best Crossover Probability: {best_params['co_prob']}")

    print("\n--- Finetuning Complete ---")
    print(f"Optimal Parameters Found: {best_params}")
    print(f"Best Average Fitness Achieved: {best_fitness:.4f}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure data is split before running
    try:
        open('training_data.csv')
    except FileNotFoundError:
        print("Data not split. Please run prepare_data.py first.")
    else:
        finetune_parameters()