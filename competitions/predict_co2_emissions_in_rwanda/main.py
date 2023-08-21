from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

def train_test_split(df, test_size=0.2, random_state=None):
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state)
    
    # Calculate test set size
    test_set_size = int(len(df) * test_size)
    
    # Split the DataFrame into train and test sets
    df_test = df_shuffled.iloc[:test_set_size]
    df_train = df_shuffled.iloc[test_set_size:]
    
    return df_train, df_test

class DataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_dir, num_workers = 1):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = os.path.join("../..", data_dir)
        self.num_workers = num_workers
        
    def train_dataloader(self):
        csv_path = os.path.join(self.data_dir, 'train.csv')
        df = pd.read_csv(csv_path)
        return DataLoader(Dataset(df=df), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        csv_path = os.path.join(self.data_dir, 'test.csv')
        df = pd.read_csv(csv_path)
        return DataLoader(Dataset(df=df), batch_size=self.batch_size, shuffle=False)
        
class Dataset(Dataset):
    def __init__(self, df, columns_to_remove = None, handle_nan = None) -> None:
        super().__init__()
        if columns_to_remove is None:
            self.columns_to_remove = ['ID_LAT_LON_YEAR_WEEK', 'year', 'week_no']

        self.data = df
        self.data = self.data.drop(columns=self.columns_to_remove)

        self.output_columns = ['emission']
        self.normalize(self.output_columns)

        self.input_columns = self.data.columns.drop(self.output_columns)
        self.normalize(self.input_columns)
        self.data.fillna(0, inplace=True)
        
    def normalize(self, columns):
        for column in columns:
            self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input = self.data.iloc[index][self.input_columns]
        input = torch.tensor(input, dtype=torch.float32)
        output = self.data.iloc[index][self.output_columns]
        output = torch.tensor(output, dtype=torch.float32)
        return input, output
      
class SimpleModel(L.LightningModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(input_size, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, output_size)
        )
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def predict(self, x):
        return self(x)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss
    




