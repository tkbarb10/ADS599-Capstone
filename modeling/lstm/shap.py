from modeling.lstm.model import LSTMSequenceModel, ShapWrap
import torch
import shap
from pathlib import Path
from utils.load_yaml_helper import load_yaml
import torch.nn as nn
from modeling.data_prep import lstm
import numpy as np
import matplotlib.pyplot as plt
import logging

settings = load_yaml("project_setup/settings.yaml")
hf_cfg = settings['hugging_face']
random_state = 10 # get from lstm.yaml file

root = Path(__file__).parents[2]
artifact_path = root / "modeling/lstm/artifacts"

for p in root.rglob("*.pt"):
    if 'weights' in str(p):
        weights_path = p

logger = logging.getLogger(__name__)

device = (
    torch.accelerator.current_accelerator().type  # type: ignore
    if torch.accelerator.is_available()
    else "cpu"
)
logger.info(f"Using device {device}")

seq_model_weights = torch.load(weights_path, map_location=device)

seq_model = LSTMSequenceModel()

seq_model.load_state_dict(seq_model_weights)

seq_model.train()

full_data, state_cols = lstm.load_and_prep_lstm(hf_cfg=hf_cfg)

modeling_data = lstm.remove_outlier_stays(full_data)

df_train, df_test, df_val = lstm.split_data(df=modeling_data, train_size=0.6)

train, test, val, scaler = lstm.scaling(train=df_train, test=df_test, val=df_val)

s_train, y_train, len_train = lstm.pad_stays(df=train, state_cols=state_cols)
s_train = torch.tensor(s_train)

reference_data = torch.randperm(s_train.shape[0], generator=torch.Generator().manual_seed(random_state))[:200]
baseline = s_train[reference_data]

shap_wrapper = ShapWrap(seq_model=seq_model).to(device)

explainer = shap.GradientExplainer(model=shap_wrapper, data=baseline)

