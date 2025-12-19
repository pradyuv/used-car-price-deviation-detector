import pandas as pd
import numpy as np


"""
After some deliberation, I will split the columns as such:
Target:
- price

Core features:
- brand
- model
- model_year
- milage
- accident
- clean_title

Auxiliary features:
- fuel_type
- engine
- transmission

Metadata / dropped:
- ext_col
- int_col

"""


TARGET_COLS = ["price"]
CORE_FEATURES = ["brand", "model", "model_year", "milage", "accident", "clean_title"] # Mileage here is misspelt for some reason
AUX_FEATURES = ["fuel_type", "engine_displacement_liters", "transmission_type"]
DROP_COLUMNS = ["ext_col", "int_col"]
NUMERIC_COLS = ["price", "milage", "model_year", "engine_displacement_liters"]
CATEGORICAL_COLS = ["brand", "model", "fuel_type", "clean_title", "transmission_type"]
