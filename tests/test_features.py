
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from diabetes_model.config.core import config
from diabetes_model.processing.features import  Mapper, LabelEncoderTransformer, SmokingOneHotEncoder


def test_SmokingOneHotEncoder(sample_input_data):
    # Given
    encoder = SmokingOneHotEncoder(variables = config.model_config_.smoking_history_var)
    
    assert sample_input_data[0].loc[75721, 'smoking_history'] == 'No Info'

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])
    
    # Then
    assert subject.loc[75721, 'smoking_history_No Info'] == 1


def test_LabelEncoderTransformer(sample_input_data):
    # Given
    encoder = LabelEncoderTransformer(variables = config.model_config_.gender_var)
    
    assert sample_input_data[0].loc[75721, 'gender'] == 'Female'

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[75721, 'gender'] == 0

