import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

from diabetes_model.config.core import config
from diabetes_model.processing.features import Mapper
from diabetes_model.processing.features import LabelEncoderTransformer, SmokingOneHotEncoder

diabetes_pipe = Pipeline([

       
    ######## One-hot encoding ########
    ('encode_smoking_histoy', SmokingOneHotEncoder(variables = config.model_config_.smoking_history_var)),

    ('encode_gender', LabelEncoderTransformer(variables = config.model_config_.gender_var)),

    # Scale features
    ('scaler', StandardScaler()),
    
    # Regressor
    #('model_rf', RandomForestRegressor(n_estimators = config.model_config_.n_estimators, 
    #                                   max_depth = config.model_config_.max_depth,
    #                                  random_state = config.model_config_.random_state))
    
    ('model_rf', RandomForestRegressor(n_estimators = config.model_config_.n_estimators, 
                                    max_depth = config.model_config_.max_depth,
                                    random_state = config.model_config_.random_state))]
    
    )


