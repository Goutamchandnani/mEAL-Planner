# Meal Optimization Project

This project explores various computational intelligence techniques for meal optimization, focusing on creating balanced and nutritious meal plans based on a given food database. It leverages algorithms such as Genetic Algorithms (GA) and Artificial Bee Colony (ABC) to find optimal combinations of foods that meet specific dietary requirements.

## Project Structure

The project is organized into several Python scripts, each addressing a different aspect of meal optimization or comparing different algorithmic approaches.

- [`food_database.py`](food_database.py): Manages the loading and processing of food data from `foods.csv`.
- [`foods.csv`](foods.csv): A CSV file containing the nutritional information for various food items.
- [`meal.py`](meal.py): Defines the `Meal` class, representing a collection of food items and calculating its nutritional properties.
- [`main.py`](main.py): The main entry point for running the meal optimization process.
- [`prepare_data.py`](prepare_data.py): Script for preparing and preprocessing the food data.
- [`requirements.txt`](requirements.txt): Lists the Python dependencies required to run the project.
- [`task_3_meal_ga.py`](task_3_meal_ga.py): Implements meal optimization using a Genetic Algorithm.
- [`task_4_abc_vs_ga.py`](task_4_abc_vs_ga.py): Compares the performance of Artificial Bee Colony (ABC) algorithm against Genetic Algorithm (GA) for meal optimization.
- [`task_5_finetune.py`](task_5_finetune.py): Focuses on finetuning parameters for the optimization algorithms.
- [`task_6_fitness_sharing.py`](task_6_fitness_sharing.py): Explores the concept of fitness sharing in genetic algorithms to maintain diversity in the population.
- [`task_7_two_point_vs_ga.py`](task_7_two_point_vs_ga.py): Compares two-point crossover with standard genetic algorithm approaches.
- [`task_11_dup_vs_share.py`](task_11_dup_vs_share.py): Investigates the impact of duplicate solutions versus fitness sharing on optimization results.
- [`testing_data.csv`](testing_data.csv): Dataset used for testing the optimization models.
- [`training_data.csv`](training_data.csv): Dataset used for training or parameter tuning of the optimization models.

## Getting Started

To set up and run the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd CI Assignment
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the main optimization script:**
    ```bash
    python main.py
    ```
    (Note: You might need to specify which task to run or modify `main.py` to execute a specific task script.)

## Algorithms Explored

This project primarily focuses on:

-   **Genetic Algorithms (GA):** A metaheuristic inspired by the process of natural selection, used to generate high-quality solutions to optimization and search problems.
-   **Artificial Bee Colony (ABC) Algorithm:** A swarm intelligence-based algorithm inspired by the foraging behavior of honey bee colonies.

## Data

The project utilizes two main data files:

-   `foods.csv`: Contains detailed nutritional information (e.g., calories, protein, carbs, fat) for each food item.
-   `training_data.csv` and `testing_data.csv`: These files are likely used for evaluating the performance and robustness of the implemented optimization algorithms.

## Future Enhancements

-   Implement additional computational intelligence algorithms (e.g., Particle Swarm Optimization, Simulated Annealing).
-   Develop a user interface for easier interaction and visualization of meal plans.
-   Integrate with external food databases for a wider variety of food options.
-   Add constraints for allergies or dietary preferences.