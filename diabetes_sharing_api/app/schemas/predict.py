from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class DataInputSchema(BaseModel):
    gender: Optional[str]
    age: Optional[float]
    hypertension: Optional[int]
    heart_disease: Optional[int]
    smoking_history: Optional[str]
    HbA1c_level: Optional[float]
    blood_glucose_level: Optional[float]
    # bmi: Optional[float]    


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
