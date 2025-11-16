# Student: [Your Name/ID]
# Course: [Your Course Name]
# Date: [Current Date]

"""
Task 4: Artificial Bee Colony (ABC) vs. Genetic Algorithm (GA)

This script implements the Artificial Bee Colony (ABC) algorithm for meal optimization
and compares its performance against our baseline Genetic Algorithm (mEAl).
The goal is to see if this nature-inspired approach can find better meals.
"""

import random
from typing import List, Tuple
from deap import base, creator, tools, algorithms
import numpy as np
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

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
        # Scoring for calories, vitamins, and minerals
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

# --- Baseline Genetic Algorithm (for comparison) ---
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

def evolve_meal(population_size=100, generations=50):
    """Our baseline Genetic Algorithm."""
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
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, verbose=False)
    return Meal(tools.selBest(pop, k=1)[0])

# --- Artificial Bee Colony (ABC) Implementation ---

class FoodSource:
    """Represents a food source (a potential meal) for the bees."""
    def __init__(self, meal_composition):
        self.meal = meal_composition
        self.fitness = Meal(meal_composition).calculate_fitness()
        self.trials = 0  # How many times this source has been tried without improvement

def generate_neighbor_meal(meal):
    """Creates a new meal by slightly modifying an existing one."""
    new_meal = [list(item) for item in meal]
    if not new_meal: return create_random_meal()
    
    idx = random.randint(0, len(new_meal) - 1)
    if random.random() < 0.7: # Modify quantity
        new_meal[idx][1] = max(0.05, min(new_meal[idx][1] + random.uniform(-0.1, 0.1), 0.5))
    else: # Modify food item
        new_meal[idx][0] = random.choice(list(FOOD_DATABASE.values()))
    return [tuple(item) for item in new_meal]

def optimize_meal_abc(num_sources=50, max_trials=5, iterations=100):
    """Main function to run the Artificial Bee Colony algorithm."""
    # 1. Initialize food sources
    sources = [FoodSource(create_random_meal()) for _ in range(num_sources)]
    best_source = max(sources, key=lambda s: s.fitness)

    for it in range(iterations):
        # 2. Employed Bee Phase: Explore around existing food sources
        for i in range(num_sources):
            new_meal = generate_neighbor_meal(sources[i].meal)
            new_fitness = Meal(new_meal).calculate_fitness()
            if new_fitness > sources[i].fitness:
                sources[i] = FoodSource(new_meal)
            else:
                sources[i].trials += 1

        # 3. Onlooker Bee Phase: Bees choose sources based on fitness
        total_fitness = sum(s.fitness for s in sources if s.fitness > 0)
        probs = [s.fitness / total_fitness if total_fitness > 0 else 1/len(sources) for s in sources]
        for i in range(num_sources):
            chosen_idx = np.random.choice(len(sources), p=probs)
            new_meal = generate_neighbor_meal(sources[chosen_idx].meal)
            new_fitness = Meal(new_meal).calculate_fitness()
            if new_fitness > sources[chosen_idx].fitness:
                sources[chosen_idx] = FoodSource(new_meal)
            else:
                sources[chosen_idx].trials += 1

        # 4. Scout Bee Phase: Abandon poor sources and find new ones
        for i in range(num_sources):
            if sources[i].trials > max_trials:
                sources[i] = FoodSource(create_random_meal())
        
        # Update the best solution found so far
        current_best = max(sources, key=lambda s: s.fitness)
        if current_best.fitness > best_source.fitness:
            best_source = current_best
        
        if (it + 1) % 10 == 0:
            print(f"ABC Iteration {it+1}: Best Fitness = {best_source.fitness:.2f}")

    return Meal(best_source.meal)

# --- Main Execution and Comparison ---

def compare_algorithms(num_trials=10):
    """Runs both GA and ABC algorithms multiple times and compares their performance."""
    print("\n--- Running Genetic Algorithm (mEAl) for comparison ---")
    ga_scores = [evolve_meal(population_size=150, generations=200).calculate_fitness() for i in range(num_trials)]
    
    print(f"\n--- Running Artificial Bee Colony (ABC) Algorithm ---")
    abc_scores = [optimize_meal_abc(num_sources=50, max_trials=5, iterations=200).calculate_fitness() for i in range(num_trials)]

    print("\n--- Statistical Comparison (Welch's t-test) ---")
    print(f"GA Avg Fitness:    {np.mean(ga_scores):.4f} ± {np.std(ga_scores):.4f}")
    print(f"ABC Avg Fitness:   {np.mean(abc_scores):.4f} ± {np.std(abc_scores):.4f}")

    t_stat, p_val = stats.ttest_ind(ga_scores, abc_scores, equal_var=False)
    print(f"\nT-test Results: T-statistic = {t_stat:.4f}, P-value = {p_val:.4f}")

    if p_val < 0.05:
        winner = "Artificial Bee Colony (ABC)" if np.mean(abc_scores) > np.mean(ga_scores) else "Genetic Algorithm (mEAl)"
        print(f"\nConclusion: The difference is statistically significant. {winner} performs better.")
    else:
        print("\nConclusion: No statistically significant difference between the algorithms.")

if __name__ == "__main__":
    # Ensure data is split before running
    try:
        open('training_data.csv')
    except FileNotFoundError:
        print("Data not split. Please run prepare_data.py first.")
    else:
        compare_algorithms()