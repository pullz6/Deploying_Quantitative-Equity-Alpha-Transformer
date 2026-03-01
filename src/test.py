import torch
from pytorch_forecasting import TemporalFusionTransformer

# 1. Load the Pre-Trained Weights
model = TemporalFusionTransformer.load_from_checkpoint("model.ckpt")

# 2. Prepare Your Data
# Data must be a Pandas DataFrame with columns:
# ['Ticker', 'date', 'close', 'liquidity_flags', 'time_idx']

# 3. Generate Probabilistic Predictions
raw_prediction = model.predict(your_dataframe, mode="raw", return_x=True)

# 4. Extract Quantiles
interpretation = model.interpret_output(raw_prediction.output, reduction="sum")
print("Attention Weights:", interpretation["attention"].shape)
