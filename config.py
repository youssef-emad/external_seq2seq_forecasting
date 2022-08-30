from functools import partial

from torch.nn import MSELoss
from torch.optim import RMSprop

from controllers import sampler

# experiment parameters
input_window_size = 24
output_window_size = 12
min_window_size = input_window_size + output_window_size

experiment = {
    "name": "MC_experiment_D0.7_50",
    "input_window_size": input_window_size,
    "output_window_size": output_window_size,
    "min_window_size": input_window_size + output_window_size,
    "feature_columns": [f"oil_month_{i + 1}" for i in range(60)],
}

data = {
    "training": {
        "csv_path": "data_split_csv/train_data.csv",
        "dataloader_params": {"batch_size": 128, "shuffle": True},
        "sampler": partial(
            sampler,
            input_window_size=input_window_size,
            output_window_size=output_window_size,
            mode="random",
        ),
    },
    "validation": {
        "csv_path": "data_split_csv/test_data.csv",
        "dataloader_params": {"batch_size": 60, "shuffle": False},
        "sampler": partial(
            sampler,
            input_window_size=input_window_size,
            output_window_size=output_window_size,
            mode="deterministic",
        ),
    },
}

# generator parameters
generator = {
    "latent_size": 64,  # 32
    "n_filters": 186,  # 128
    "kernel_size": 5,
    "dropout": 0.5,
    "pool_size": 2,
    "rnn_cell_type": "lstm",
}

# training params
training_params = {
    "training": {
        "num_steps": 8000,
        "num_steps_model_saving": 1000,
    },
    "validation": {
        "num_steps_log": 1000,
        "num_batches": 100,
        "num_steps_per_prediction": 50,
    },
    "generator": {
        "optimizer": partial(RMSprop, lr=0.0001),
        "loss": MSELoss,
    },
    "metrics": {
        "training": {"generator": ["rmse", "mape", "mae"]},
        "validation": ["rmse", "mape", "mae", "picp", "pinaw"],
        "best_saving_criteria": {"metric": "validation/rmse", "mode": "min"},
    },
}
