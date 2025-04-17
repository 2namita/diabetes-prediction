import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from diabetes_model import __version__ as _version
from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import load_pipeline
from diabetes_model.processing.data_manager import pre_pipeline_preparation
from diabetes_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
diabetes_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = diabetes_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    # Example input data

    data_in = {'gender': [Female], 'age': ['39'], 'hypertension': [0], 'heart_disease': [0], 'smoking_history': ['current'], 'bmi': [66.88], 'HbA1c_level': ['6.5'], 'blood_glucose_level': ['126']}

    make_prediction(input_data = data_in)