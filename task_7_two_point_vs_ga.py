"""
Task 7: Improve the mEAl algorithm by enhancing one of its genetic operators (Two-Point Crossover).
This file consolidates all necessary code for implementing the enhanced crossover and comparing it with the core mEAl algorithm.
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
        self.name = name
        self.calories = calories  # per 100g
        self.macronutrients = macronutrients  # dict containing total_carbs, protein, saturated_fat, unsaturated_fat
        self.micronutrients = micronutrients  # dict containing vitamins and minerals
        self.fiber = fiber  # grams per 100g

def parse_value(value_str):
    if isinstance(value_str, (int, float)):
        return value_str
    match = re.match(r"([0-9.]+)", str(value_str))
    if match:
        return float(match.group(1))
    return 0.0

def load_food_database_from_csv(filepath: str):
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
    Represents a meal composed of various food items with their respective quantities.
    Calculates nutritional totals and a fitness score based on predefined thresholds.
    """
    def __init__(self, foods: List[Tuple[Food, float]]):
        """
        Initializes a Meal object.

        Args:
            foods: A list of (Food, weight) tuples, where weight is in kilograms.
        """
        self.foods = foods
        self.calories = 0.0
        self.water_soluble_vitamins = 0.0
        self.fat_soluble_vitamins = 0.0
        self.safe_minerals = 0.0
        self.unsafe_minerals = 0.0
        self._calculate_totals()
    
    def _calculate_totals(self):
        """
        Calculates all nutritional totals for the meal based on the food items and their weights.
        Converts weights from kg to 100g units for calculation.
        """
        for food, weight in self.foods:
            # Convert kg to 100g units (1 kg = 10 * 100g units)
            weight_in_100g = weight * 10
            
            self.calories += food.calories * weight_in_100g
            
            for vitamin, amount in food.micronutrients.items():
                if amount == "trace":
                    amount = 0.01
                    
                if vitamin in ['vitamin_c', 'vitamin_b']:
                    self.water_soluble_vitamins += amount * weight_in_100g
                elif vitamin in ['vitamin_a', 'vitamin_d', 'vitamin_e', 'vitamin_k']:
                    self.fat_soluble_vitamins += amount * weight_in_100g
            
            unsafe_minerals = ['sodium']
            for mineral, amount in food.micronutrients.items():
                if amount == "trace":
                    amount = 0.01
                    
                if mineral in unsafe_minerals:
                    self.unsafe_minerals += amount * weight_in_100g
                elif mineral in ['iron', 'magnesium', 'zinc', 'selenium']:
                    self.safe_minerals += amount * weight_in_100g
    
    def calculate_fitness(self) -> float:
        """
        Calculates the fitness value for this meal based on its nutritional content,
        using smooth penalties for deviations from target ranges and bonuses for diversity.

        Returns:
            A float representing the fitness value, where higher is better.
        """
        fitness = 0.0

        # Calories (target range 500-800 kcal)
        cal = self.calories
        cal_min = THRESHOLDS['calories_min']
        cal_max = THRESHOLDS['calories_max']
        if cal_min <= cal <= cal_max:
            fitness += 1.0
        elif cal < cal_min:
            # Penalize more heavily for being too low
            penalty = (cal_min - cal) / cal_min * 2.0
            fitness -= penalty
        else: # cal > cal_max
            # Penalize much more heavily for being too high
            penalty = (cal - cal_max) / cal_max * 5.0
            fitness -= penalty

        # Water-soluble vitamins (minimum threshold)
        wsv = self.water_soluble_vitamins
        wsv_min = THRESHOLDS['water_soluble_vitamins_min']
        fitness += min(wsv / wsv_min, 1.0)

        # Fat-soluble vitamins (min-max range)
        fsv = self.fat_soluble_vitamins
        fsv_min = THRESHOLDS['fat_soluble_vitamins_min']
        fsv_max = THRESHOLDS['fat_soluble_vitamins_max']
        if fsv_min <= fsv <= fsv_max:
            fitness += 1.0
        elif fsv < fsv_min:
            fitness += fsv / fsv_min
        else:
            fitness += max(0.0, 1.0 - (fsv - fsv_max) / (fsv_max * 2))

        # Safe minerals (minimum threshold)
        sm = self.safe_minerals
        sm_min = THRESHOLDS['safe_minerals_min']
        fitness += min(sm / sm_min, 1.0)

        # Unsafe minerals (min-max range)
        um = self.unsafe_minerals
        um_min = THRESHOLDS['unsafe_minerals_min']
        um_max = THRESHOLDS['unsafe_minerals_max']
        if um_min <= um <= um_max:
            fitness += 1.0
        elif um < um_min:
            fitness += um / um_min
        else:
            fitness += max(0.0, 1.0 - (um - um_max) / (um_max * 2))

        # Penalize excessive total weight (max 1.2 kg)
        total_weight = self.get_total_weight()
        max_weight = 1.2
        if total_weight > max_weight:
            fitness -= (total_weight - max_weight) * 5

        # Diversity bonus (0.1 point for each food >= 50g)
        min_food_weight = 0.05
        num_foods = sum(1 for _, w in self.foods if w >= min_food_weight)
        fitness += 0.1 * num_foods

        return fitness
    
    def get_calories(self) -> float:
        """Returns the total calories of the meal."""
        return self.calories
    
    def get_water_soluble_vitamins(self) -> float:
        """Returns the total water-soluble vitamins of the meal."""
        return self.water_soluble_vitamins
    
    def get_fat_soluble_vitamins(self) -> float:
        """Returns the total fat-soluble vitamins of the meal."""
        return self.fat_soluble_vitamins
    
    def get_safe_minerals(self) -> float:
        """Returns the total safe minerals of the meal."""
        return self.safe_minerals
    
    def get_unsafe_minerals(self) -> float:
        """Returns the total unsafe minerals of the meal."""
        return self.unsafe_minerals

    def get_total_weight(self) -> float:
        """Returns the total weight of the meal in kilograms."""
        return sum(weight for _, weight in self.foods)

# --- Content from meal_evolution_task_3.py (evolve_meal function) ---
# Global FOOD_DATABASE needs to be loaded before use
FOOD_DATABASE = load_food_database_from_csv('training_data.csv')

def create_random_meal_base() -> List[Tuple[Food, float]]:
    num_foods = random.randint(1, 4)
    foods = random.sample(list(FOOD_DATABASE.values()), num_foods)
    return [(food, random.uniform(0.1, 0.5)) for food in foods]

def evaluate_meal_base(individual: List[Tuple[Food, float]]) -> Tuple[float,]:
    meal = Meal(individual)
    fitness = meal.calculate_fitness()
    return (fitness,)

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

def crossover_meals_base(ind1: List[Tuple[Food, float]], 
                   ind2: List[Tuple[Food, float]]) -> Tuple[List[Tuple[Food, float]], List[Tuple[Food, float]]]:
    if len(ind1) > 0 and len(ind2) > 0:
        point = random.randint(1, min(len(ind1), len(ind2)))
        ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
    
    return ind1, ind2

def evolve_meal_base(
    population_size: int = 100,
    generations: int = 50,
    min_fitness_threshold: float = 5.0,
    cxpb: float = 0.7,
    mutpb: float = 0.2
) -> Meal:
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

    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        population = toolbox.select(offspring, k=len(population))
        
        record = stats.compile(population)
        print(f"Generation {gen}: {record}")
        
        if record['max'] >= min_fitness_threshold:
            print(f"Termination condition met: Max fitness of {record['max']} reached.")
            break

    best_individual = tools.selBest(population, k=1)[0]
    return Meal(best_individual)

# --- Content from meal_evolution_enhanced_task_7.py ---
def crossover_meals_two_point_task_7(ind1: List[Tuple[Food, float]], 
                              ind2: List[Tuple[Food, float]]) -> Tuple[List[Tuple[Food, float]], List[Tuple[Food, float]]]:
    size = min(len(ind1), len(ind2))
    if size < 2:
        return crossover_meals_base(ind1, ind2) # Fallback to one-point if not enough items

    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    
    return ind1, ind2

def evolve_meal_enhanced_task_7(
    population_size: int = 100,
    generations: int = 50,
    min_fitness_threshold: float = 5.0,
    cxpb: float = 0.7,
    mutpb: float = 0.2
) -> Meal:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_random_meal_base)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_meal_base)
    toolbox.register("mate", crossover_meals_two_point_task_7) # Use the enhanced crossover
    toolbox.register("mutate", mutate_meal_base)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)
    stats.register("max", max)

    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        population = toolbox.select(offspring, k=len(population))
        
        record = stats.compile(population)
        print(f"Generation {gen}: {record}")
        
        if record['max'] >= min_fitness_threshold:
            print(f"Termination condition met: Max fitness of {record['max']} reached.")
            break

    best_individual = tools.selBest(population, k=1)[0]
    return Meal(best_individual)

# --- Main execution for Task 7 ---
if __name__ == "__main__":
    # Ensure data is prepared (training_data.csv exists)
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

    split_data_if_needed()

    print("--- Running Original Genetic Algorithm (mEAl) for comparison (Task 7) ---")
    original_ga_fitness = []
    for i in range(10): # Reduced trials for quicker execution in a single file
        print(f"  Trial {i+1}/10...", end=" ", flush=True)
        best_meal = evolve_meal_base(
            population_size=250,
            generations=200,
            mutpb=0.2,
            cxpb=0.6
        )
        original_ga_fitness.append(best_meal.calculate_fitness())
        print(f"Fitness: {original_ga_fitness[-1]:.4f}")

    print("\n--- Running Enhanced Genetic Algorithm (mEAl with Two-Point Crossover) for comparison (Task 7) ---")
    enhanced_ga_fitness = []
    for i in range(10): # Reduced trials for quicker execution in a single file
        print(f"  Trial {i+1}/10...", end=" ", flush=True)
        best_meal = evolve_meal_enhanced_task_7(
            population_size=250,
            generations=200,
            mutpb=0.2,
            cxpb=0.6
        )
        enhanced_ga_fitness.append(best_meal.calculate_fitness())
        print(f"Fitness: {enhanced_ga_fitness[-1]:.4f}")

    print("\n--- Statistical Comparison (Welch's t-test) ---")
    print(f"Original GA Average Fitness: {np.mean(original_ga_fitness):.4f} ± {np.std(original_ga_fitness):.4f}")
    print(f"Enhanced GA Average Fitness: {np.mean(enhanced_ga_fitness):.4f} ± {np.std(enhanced_ga_fitness):.4f}")

    t_statistic_fit, p_value_fit = stats.ttest_ind(original_ga_fitness, enhanced_ga_fitness, equal_var=False)

    print(f"\nWelch's t-test Results (Fitness):")
    print(f"  T-statistic: {t_statistic_fit:.4f}")
    print(f"  P-value: {p_value_fit:.4f}")

    alpha = 0.05
    if p_value_fit < alpha:
        print(f"\nConclusion: With p-value {p_value_fit:.4f} < {alpha}, we reject the null hypothesis.")
        if np.mean(enhanced_ga_fitness) > np.mean(original_ga_fitness):
            print("  Enhanced GA performs significantly better in terms of fitness.")
        else:
            print("  Original GA performs significantly better in terms of fitness.")
    else:
        print(f"\nConclusion: With p-value {p_value_fit:.4f} >= {alpha}, we fail to reject the null hypothesis.")
        print("  There is no statistically significant difference in fitness performance.")
    print("----------------------------------------------------")
