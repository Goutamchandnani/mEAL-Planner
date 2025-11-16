"""
Task 6: Diversity analysis - Fitness Sharing.
This file consolidates all necessary code for implementing Fitness Sharing and comparing it with the core mEAl algorithm.
"""

import random
from typing import List, Tuple, Dict
from deap import base, creator, tools, algorithms
import numpy as np
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

# --- Content from food_database.py ---
class Food:
    def __init__(self, name, calories, macronutrients, micronutrients, fiber):
        # Initialize each food item with nutritional details
        self.name = name
        self.calories = calories  # per 100g
        self.macronutrients = macronutrients  # dict for carbs, protein, fats
        self.micronutrients = micronutrients  # dict for vitamins and minerals
        self.fiber = fiber  # grams per 100g

# Function to extract numerical value from a string (e.g., "20 mg")
def parse_value(value_str):
    if isinstance(value_str, (int, float)):
        return value_str
    match = re.match(r"([0-9.]+)", str(value_str))
    if match:
        return float(match.group(1))
    return 0.0

# Loads and processes the food database from a CSV file
def load_food_database_from_csv(filepath: str):
    foods = {}
    with open(filepath, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)[1:]
        
        data_rows = {}
        # Store each nutrient type with its corresponding values
        for row in reader:
            if row:
                metric_name = row[0].split(' (')[0].lower().replace(' ', '_')
                data_rows[metric_name] = [parse_value(v) for v in row[1:]]

    # Create Food objects for each item in the database
    for i, food_name in enumerate(headers):
        food_key = food_name.lower().replace(', ', '_').replace(',', '_')
        
        # Extract macronutrients and micronutrients
        macronutrients = {
            "total_carbs": data_rows.get("total_carbs", [])[i],
            "protein": data_rows.get("protein", [])[i],
            "saturated_fat": data_rows.get("saturated_fat", [])[i],
            "unsaturated_fat": data_rows.get("unsaturated_fat", [])[i],
        }
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

# --- Content from meal.py ---
# Thresholds for calculating the nutritional fitness of a meal
THRESHOLDS = {
    'calories_min': 500,
    'calories_max': 800,
    'water_soluble_vitamins_min': 100,
    'fat_soluble_vitamins_min': 50,
    'fat_soluble_vitamins_max': 200,
    'safe_minerals_min': 1000,
    'unsafe_minerals_min': 10,
    'unsafe_minerals_max': 500
}

class Meal:
    """
    Represents a meal composed of multiple food items.
    Responsible for computing nutrition totals and evaluating fitness.
    """
    def __init__(self, foods: List[Tuple[Food, float]]):
        self.foods = foods
        self.calories = 0.0
        self.water_soluble_vitamins = 0.0
        self.fat_soluble_vitamins = 0.0
        self.safe_minerals = 0.0
        self.unsafe_minerals = 0.0
        self._calculate_totals()
    
    # Calculates all nutrient totals for the meal
    def _calculate_totals(self):
        for food, weight in self.foods:
            # Convert kg to 100g units for consistency
            weight_in_100g = weight * 10
            self.calories += food.calories * weight_in_100g
            
            # Add vitamin contributions based on solubility
            for vitamin, amount in food.micronutrients.items():
                if amount == "trace":
                    amount = 0.01
                    
                if vitamin in ['vitamin_c', 'vitamin_b']:
                    self.water_soluble_vitamins += amount * weight_in_100g
                elif vitamin in ['vitamin_a', 'vitamin_d', 'vitamin_e', 'vitamin_k']:
                    self.fat_soluble_vitamins += amount * weight_in_100g
            
            # Separate safe and unsafe minerals
            unsafe_minerals = ['sodium']
            for mineral, amount in food.micronutrients.items():
                if amount == "trace":
                    amount = 0.01
                    
                if mineral in unsafe_minerals:
                    self.unsafe_minerals += amount * weight_in_100g
                elif mineral in ['iron', 'magnesium', 'zinc', 'selenium']:
                    self.safe_minerals += amount * weight_in_100g
    
    # Calculates the meal's overall fitness score
    def calculate_fitness(self) -> float:
        fitness = 0.0

        # Check calorie range (ideal: 500-800 kcal)
        cal = self.calories
        cal_min = THRESHOLDS['calories_min']
        cal_max = THRESHOLDS['calories_max']
        if cal_min <= cal <= cal_max:
            fitness += 1.0
        elif cal < cal_min:
            penalty = (cal_min - cal) / cal_min * 2.0
            fitness -= penalty
        else:
            penalty = (cal - cal_max) / cal_max * 5.0
            fitness -= penalty

        # Add fitness for water-soluble vitamins
        wsv = self.water_soluble_vitamins
        wsv_min = THRESHOLDS['water_soluble_vitamins_min']
        fitness += min(wsv / wsv_min, 1.0)

        # Add fitness for fat-soluble vitamins (bounded range)
        fsv = self.fat_soluble_vitamins
        fsv_min = THRESHOLDS['fat_soluble_vitamins_min']
        fsv_max = THRESHOLDS['fat_soluble_vitamins_max']
        if fsv_min <= fsv <= fsv_max:
            fitness += 1.0
        elif fsv < fsv_min:
            fitness += fsv / fsv_min
        else:
            fitness += max(0.0, 1.0 - (fsv - fsv_max) / (fsv_max * 2))

        # Add contribution for safe minerals
        sm = self.safe_minerals
        sm_min = THRESHOLDS['safe_minerals_min']
        fitness += min(sm / sm_min, 1.0)

        # Evaluate unsafe minerals (penalize excess)
        um = self.unsafe_minerals
        um_min = THRESHOLDS['unsafe_minerals_min']
        um_max = THRESHOLDS['unsafe_minerals_max']
        if um_min <= um <= um_max:
            fitness += 1.0
        elif um < um_min:
            fitness += um / um_min
        else:
            fitness += max(0.0, 1.0 - (um - um_max) / (um_max * 2))

        # Penalize meals that are too heavy (>1.2 kg)
        total_weight = self.get_total_weight()
        max_weight = 1.2
        if total_weight > max_weight:
            fitness -= (total_weight - max_weight) * 5

        # Reward diversity (more unique foods ≥ 50g)
        min_food_weight = 0.05
        num_foods = sum(1 for _, w in self.foods if w >= min_food_weight)
        fitness += 0.1 * num_foods

        return fitness

    # Helper functions to get nutritional totals
    def get_calories(self) -> float:
        return self.calories
    
    def get_water_soluble_vitamins(self) -> float:
        return self.water_soluble_vitamins
    
    def get_fat_soluble_vitamins(self) -> float:
        return self.fat_soluble_vitamins
    
    def get_safe_minerals(self) -> float:
        return self.safe_minerals
    
    def get_unsafe_minerals(self) -> float:
        return self.unsafe_minerals

    def get_total_weight(self) -> float:
        return sum(weight for _, weight in self.foods)

# --- Content from meal_evolution_task_3.py ---
# Load the global food database
FOOD_DATABASE = load_food_database_from_csv('training_data.csv')

# Create a random initial meal (individual)
def create_random_meal_base() -> List[Tuple[Food, float]]:
    num_foods = random.randint(1, 4)
    foods = random.sample(list(FOOD_DATABASE.values()), num_foods)
    return [(food, random.uniform(0.1, 0.5)) for food in foods]

# Evaluate an individual meal’s fitness
def evaluate_meal_base(individual: List[Tuple[Food, float]]) -> Tuple[float,]:
    meal = Meal(individual)
    fitness = meal.calculate_fitness()
    return (fitness,)

# Mutation: randomly change a food or quantity
def mutate_meal_base(individual: List[Tuple[Food, float]]) -> List[Tuple[Food, float]]:
    if random.random() < 0.5:
        if individual:
            idx = random.randint(0, len(individual) - 1)
            new_food = random.choice(list(FOOD_DATABASE.values()))
            individual[idx] = (new_food, individual[idx][1])
    else:
        if individual:
            idx = random.randint(0, len(individual) - 1)
            new_quantity = random.uniform(0.1, 0.5)
            individual[idx] = (individual[idx][0], new_quantity)
    return (individual,)

# Crossover: exchange parts of two meals
def crossover_meals_base(ind1, ind2):
    if len(ind1) > 0 and len(ind2) > 0:
        point = random.randint(1, min(len(ind1), len(ind2)))
        ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    return ind1, ind2

# Main genetic algorithm (base version without diversity)
def evolve_meal_base(population_size=100, generations=50, min_fitness_threshold=5.0, cxpb=0.7, mutpb=0.2) -> Meal:
    # Set up DEAP toolbox and classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_random_meal_base)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_meal_base)
    toolbox.register("mate", crossover_meals_base)
    toolbox.register("mutate", mutate_meal_base)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Track statistics for analysis
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)
    stats.register("max", max)

    # Evolve over generations
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        
        # Evaluate offspring fitness
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # Select next generation
        population = toolbox.select(offspring, k=len(population))
        
        record = stats.compile(population)
        print(f"Generation {gen}: {record}")
        
        # Stop early if fitness threshold reached
        if record['max'] >= min_fitness_threshold:
            print(f"Termination condition met: Max fitness of {record['max']} reached.")
            break

    # Return best meal
    best_individual = tools.selBest(population, k=1)[0]
    return Meal(best_individual)

# --- Content from meal_evolution_div_task_6.py ---
# Parameters for Fitness Sharing
SHARING_RADIUS = 0.3
ALPHA = 1.0

# Calculate the distance (diversity) between two meals
def euclidean_distance(ind1, ind2):
    dict1 = {food.name: quantity for food, quantity in ind1}
    dict2 = {food.name: quantity for food, quantity in ind2}
    all_food_names = set(dict1.keys()).union(set(dict2.keys()))
    distance_sq = 0.0
    for food_name in all_food_names:
        q1 = dict1.get(food_name, 0.0)
        q2 = dict2.get(food_name, 0.0)
        distance_sq += (q1 - q2) ** 2
    return np.sqrt(distance_sq)

# Sharing function that reduces fitness for similar individuals
def sharing_function(distance, sigma_share, alpha):
    if distance < sigma_share:
        return 1.0 - (distance / sigma_share) ** alpha
    else:
        return 0.0

# Adjusts fitness values using fitness sharing to promote diversity
def apply_fitness_sharing(population, sigma_share, alpha):
    n = len(population)
    niche_counts = []
    for i in range(n):
        niche_count = 0.0
        for j in range(n):
            distance = euclidean_distance(population[i], population[j])
            niche_count += sharing_function(distance, sigma_share, alpha)
        niche_counts.append(niche_count)
    
    # Normalize fitness based on niche density
    for i, ind in enumerate(population):
        if niche_counts[i] > 0:
            original_fitness = ind.fitness.values[0]
            shared_fitness = original_fitness / niche_counts[i]
            ind.fitness.values = (shared_fitness,)

# Genetic algorithm with fitness sharing
def evolve_meal_div(population_size=100, generations=50, min_fitness_threshold=5.0, cxpb=0.7, mutpb=0.2, sigma_share=SHARING_RADIUS, alpha=ALPHA) -> Meal:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_random_meal_base)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_meal_base)
    toolbox.register("mate", crossover_meals_base)
    toolbox.register("mutate", mutate_meal_base)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)
    stats.register("max", max)

    hof = tools.HallOfFame(1)

    # Evolution with diversity-aware fitness
    for gen in range(generations):
        fits = toolbox.map(toolbox.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
        
        hof.update(population)
        apply_fitness_sharing(population, sigma_share, alpha)
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        population = toolbox.select(offspring, k=len(population))
        
        record = stats.compile(population)
        print(f"Generation {gen}: Avg Shared Fitness={record['avg']:.2f}, Max True Fitness={hof[0].fitness.values[0]:.2f}")
        
        if hof[0].fitness.values[0] >= min_fitness_threshold:
            print(f"Termination condition met: Max true fitness of {hof[0].fitness.values[0]:.2f} reached.")
            break

    return Meal(hof[0])

# --- Main execution for Task 6 ---
if __name__ == "__main__":
    # Function to split dataset into training/testing if needed
    def split_data_if_needed(file_path='foods.csv', train_path='training_data.csv', test_path='testing_data.csv', test_size=0.3, random_state=42):
        try:
            with open(train_path, 'r') as f:
                pass
            print(f"Using existing {train_path} and {test_path}")
        except FileNotFoundError:
            print(f"Preparing data: Splitting {file_path} into {train_path} and {test_path}")
            df = pd.read_csv(file_path, index_col=0)
            food_columns = df.columns.tolist()
            train_foods, test_foods = train_test_split(food_columns, test_size=test_size, random_state=random_state)
            train_df = df[train_foods]
            test_df = df[test_foods]
            train_df.to_csv(train_path)
            test_df.to_csv(test_path)
            print(f"Data successfully split into {train_path} and {test_path}")
            print(f"Training set contains {len(train_foods)} foods.")
            print(f"Testing set contains {len(test_foods)} foods.")
        except Exception as e:
            print(f"An error occurred during data preparation: {e}")

    # Prepare training/testing data if missing
    split_data_if_needed()

    print("\n--- Running Original Genetic Algorithm (mEAl) for comparison (Task 6) ---")
    original_ga_fitness = []
    # Run multiple trials for statistical significance
    for i in range(10):
        print(f"  Trial {i+1}/10...", end=" ", flush=True)
        best_meal = evolve_meal_base(population_size=250, generations=200, mutpb=0.2, cxpb=0.6)
        original_ga_fitness.append(best_meal.calculate_fitness())
        print(f"Fitness: {original_ga_fitness[-1]:.4f}")

    print("\n--- Running Genetic Algorithm with Fitness Sharing (mEAl_Div) for comparison (Task 6) ---")
    sharing_ga_fitness = []
    for i in range(10):
        print(f"  Trial {i+1}/10...", end=" ", flush=True)
        best_meal = evolve_meal_div(population_size=250, generations=200, mutpb=0.2, cxpb=0.6, sigma_share=SHARING_RADIUS, alpha=ALPHA)
        sharing_ga_fitness.append(best_meal.calculate_fitness())
        print(f"Fitness: {sharing_ga_fitness[-1]:.4f}")

    # Compare results statistically
    print("\n--- Statistical Comparison (Welch's t-test) ---")
    print(f"Original GA Average Fitness: {np.mean(original_ga_fitness):.4f} ± {np.std(original_ga_fitness):.4f}")
    print(f"Sharing GA Average Fitness: {np.mean(sharing_ga_fitness):.4f} ± {np.std(sharing_ga_fitness):.4f}")

    t_statistic_fit, p_value_fit = stats.ttest_ind(original_ga_fitness, sharing_ga_fitness, equal_var=False)

    print(f"\nWelch's t-test Results (Fitness):")
    print(f"  T-statistic: {t_statistic_fit:.4f}")
    print(f"  P-value: {p_value_fit:.4f}")

    # Draw conclusion from hypothesis test
    alpha = 0.05
    if p_value_fit < alpha:
        print(f"\nConclusion: With p-value {p_value_fit:.4f} < {alpha}, we reject the null hypothesis.")
        if np.mean(sharing_ga_fitness) > np.mean(original_ga_fitness):
            print("  Fitness Sharing GA performs significantly better in terms of fitness.")
        else:
            print("  Original GA performs significantly better in terms of fitness.")
    else:
        print(f"\nConclusion: With p-value {p_value_fit:.4f} >= {alpha}, we fail to reject the null hypothesis.")
        print("  There is no statistically significant difference in fitness performance.")
    print("----------------------------------------------------")
