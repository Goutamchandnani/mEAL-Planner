# Student: [Your Name/ID]
# Course: [Your Course Name]
# Date: [Current Date]

"""
meal.py

This file defines the Meal class, which is the core of our evaluation.
It takes a combination of foods and calculates its total nutritional content
and a 'fitness' score based on how well it meets our dietary goals.
"""

from typing import List, Tuple
from food_database import Food

# These are the nutritional targets for a good meal.
# We use these values in the fitness calculation to score each meal.
THRESHOLDS = {
    'calories_min': 500,
    'calories_max': 800,
    'water_soluble_vitamins_min': 100,  # e.g., Vitamin C, B
    'fat_soluble_vitamins_min': 50,     # e.g., Vitamin A, D
    'fat_soluble_vitamins_max': 200,
    'safe_minerals_min': 1000,          # e.g., Iron, Magnesium
    'unsafe_minerals_min': 10,          # e.g., Sodium
    'unsafe_minerals_max': 500
}

class Meal:
    """
    Represents a meal, which is a list of foods and their weights.
    It calculates the meal's total nutrients and its fitness score.
    """
    def __init__(self, foods: List[Tuple[Food, float]]):
        """
        Initializes a Meal.
        'foods' is a list like [(food_object, weight_in_kg), ...].
        """
        self.foods = foods
        self.calories = 0.0
        self.water_soluble_vitamins = 0.0
        self.fat_soluble_vitamins = 0.0
        self.safe_minerals = 0.0
        self.unsafe_minerals = 0.0
        self._calculate_totals()  # Calculate nutrients as soon as the meal is created.
    
    def _calculate_totals(self):
        """
        Sums up all the nutritional values from the foods in the meal.
        """
        for food, weight in self.foods:
            # Food data is per 100g, so we convert the weight from kg.
            weight_in_100g = weight * 10
            
            self.calories += food.calories * weight_in_100g
            
            # Sum up vitamins
            for vitamin, amount in food.micronutrients.items():
                if amount == "trace":
                    amount = 0.01  # Use a small value for trace amounts
                
                if vitamin in ['vitamin_c', 'vitamin_b']:
                    self.water_soluble_vitamins += amount * weight_in_100g
                elif vitamin in ['vitamin_a', 'vitamin_d', 'vitamin_e', 'vitamin_k']:
                    self.fat_soluble_vitamins += amount * weight_in_100g
            
            # Sum up minerals, separating them into 'safe' and 'unsafe' categories
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
        Calculates the fitness score of the meal.
        A higher score means a better meal. The score is based on how well
        the meal meets the nutritional thresholds.
        """
        fitness = 0.0

        # 1. Calories: Score based on being within the min/max range.
        cal = self.calories
        cal_min, cal_max = THRESHOLDS['calories_min'], THRESHOLDS['calories_max']
        if cal_min <= cal <= cal_max:
            fitness += 1.0  # Perfect score for calories
        elif cal < cal_min:
            fitness -= ((cal_min - cal) / cal_min) * 2.0  # Penalty for too few calories
        else: # cal > cal_max
            fitness -= ((cal - cal_max) / cal_max) * 5.0  # Heavier penalty for too many calories

        # 2. Water-soluble vitamins: Score based on meeting the minimum.
        wsv = self.water_soluble_vitamins
        wsv_min = THRESHOLDS['water_soluble_vitamins_min']
        fitness += min(wsv / wsv_min, 1.0)  # Score is proportional to how much of the minimum is met, capped at 1.0

        # 3. Fat-soluble vitamins: Score based on being within the min/max range.
        fsv = self.fat_soluble_vitamins
        fsv_min, fsv_max = THRESHOLDS['fat_soluble_vitamins_min'], THRESHOLDS['fat_soluble_vitamins_max']
        if fsv_min <= fsv <= fsv_max:
            fitness += 1.0
        elif fsv < fsv_min:
            fitness += fsv / fsv_min  # Proportional score if below minimum
        else:
            fitness += max(0.0, 1.0 - (fsv - fsv_max) / (fsv_max * 2)) # Penalty if above maximum

        # 4. Safe minerals: Score based on meeting the minimum.
        sm = self.safe_minerals
        sm_min = THRESHOLDS['safe_minerals_min']
        fitness += min(sm / sm_min, 1.0)

        # 5. Unsafe minerals: Score based on being within the min/max range.
        um = self.unsafe_minerals
        um_min, um_max = THRESHOLDS['unsafe_minerals_min'], THRESHOLDS['unsafe_minerals_max']
        if um_min <= um <= um_max:
            fitness += 1.0
        elif um < um_min:
            fitness += um / um_min
        else:
            fitness += max(0.0, 1.0 - (um - um_max) / (um_max * 2))

        # 6. Total Weight Penalty: Penalize meals that are too heavy (e.g., > 1.2 kg).
        total_weight = self.get_total_weight()
        if total_weight > 1.2:
            fitness -= (total_weight - 1.2) * 5

        # 7. Diversity Bonus: Reward meals with more variety.
        # Give a small bonus for each food item that has a significant weight (>= 50g).
        num_diverse_foods = sum(1 for _, w in self.foods if w >= 0.05)
        fitness += 0.1 * num_diverse_foods

        return fitness
    
    # Getter methods to easily access the meal's nutritional totals.
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
        """Returns the total weight of the meal in kilograms."""
        return sum(weight for _, weight in self.foods)