# Student: [Your Name/ID]
# Course: [Your Course Name]
# Date: [Current Date]

"""
main.py

This is the main entry point for running the meal optimization project.
You can uncomment different function calls to run various experiments.
"""

# Import the functions from our different task files
from task_3_meal_ga import evolve_meal as run_task_3
from task_4_abc_vs_ga import compare_algorithms as run_task_4
from task_5_finetune import finetune_parameters as run_task_5
from task_6_fitness_sharing import compare_fitness_sharing as run_task_6
from task_7_two_point_vs_ga import compare_crossover_performance as run_task_7
from task_11_dup_vs_share import compare_diversity_techniques as run_task_11

if __name__ == "__main__":
    # --- INSTRUCTIONS ---
    # To run a specific task, uncomment the corresponding line below.
    # Make sure to only run one at a time.

    print("--- Starting Meal Optimization Project ---")

    # Task 3: Run the core Genetic Algorithm to find an optimal meal.
    # run_task_3()

    # Task 4: Compare the Genetic Algorithm with the Artificial Bee Colony algorithm.
    # run_task_4()

    # Task 5: Finetune the parameters of the Genetic Algorithm.
    # run_task_5()

    # Task 6: Compare the standard GA with a GA using Fitness Sharing.
    # run_task_6()

    # Task 7: Compare the standard GA with a GA using an enhanced Two-Point Crossover.
    run_task_7()

    # Task 11: Compare the Fitness Sharing GA with a GA using Duplicate Avoidance.
    # run_task_11()

    print("\n--- Project Execution Finished ---")