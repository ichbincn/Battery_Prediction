import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import torch

# 加载装机量数据
装机量_df = pd.read_excel("装机量.xlsx")

# 处理时间特征
装机量_df['时间'] = pd.to_datetime(装机量_df['时间'])

# 标准化装机量数据（可选）
mean = 装机量_df.mean()
std = 装机量_df.std()
装机量_df = (装机量_df - mean) / std

# 定义训练数据
train_data = 装机量_df['总'].values

# 定义训练特征（此处只使用了总装机量，可以根据实际情况加入其他特征）
features_train = 装机量_df['总'].values

# 定义窗口大小（根据需要进行调整）
window_size = 8

# 将训练数据转换为序列
train_x = []
train_y = []
for i in range(len(train_data) - window_size):
    train_x.append(train_data[i:i+window_size])
    train_y.append(train_data[i+window_size])

# 将训练数据和训练标签转换为张量
train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
features_train_tensor = torch.tensor(features_train, dtype=torch.float32)

# 定义测试数据（这里示例为最后一个窗口）
test_x = train_data[-window_size:]
test_x_tensor = torch.tensor(test_x, dtype=torch.float32)

# 定义测试特征数据（这里示例使用最后一个月的总装机量作为测试特征）
test_features_tensor = torch.tensor(features_train[-1], dtype=torch.float32).unsqueeze(0)

# Define your Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.01):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim, noise_level
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)

    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1

    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x

    def decoder(self, x):
        h2 = self.fc2(x)
        return h2

    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode

# Define your Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=16):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x

# Define your model class with features
class NetWithFeatures(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_layers, nhead, dropout, noise_level):
        super(NetWithFeatures, self).__init__()
        # Define model architecture here
        self.autoencoder = Autoencoder(input_size=feature_size, hidden_dim=hidden_dim, noise_level=noise_level)
        self.pos = PositionalEncoding(d_model=hidden_dim, max_len=feature_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                                    dropout=dropout)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim + 1, 1)  # Output layer to predict battery scrap amount

    def forward(self, x, features):
        encode, _ = self.autoencoder(x)
        encode = self.pos(encode)
        out = self.cell(encode)
        features = features.unsqueeze(0).expand(out.size(0), -1, -1)  # Expand features to match sequence length
        out = torch.cat((out, features), dim=2)  # Concatenate sequence output with features
        out = self.linear(out)  # Output layer
        return out.squeeze(2)  # Squeeze out the last dimension to match the expected shape

# Assume you have loaded your data and split it into train_x, train_y, and features_train
# Convert data to tensors
train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
features_train_tensor = torch.tensor(features_train, dtype=torch.float32)

# Define model parameters
feature_size = train_x.shape[1]  # Assuming train_x is of shape (num_samples, feature_size)
hidden_dim = 32
num_layers = 1
nhead = 8
dropout = 0.0
noise_level = 0.01
lr = 0.01
weight_decay = 0.0
EPOCH = 1000

# Instantiate the model
model_with_features = NetWithFeatures(feature_size, hidden_dim, num_layers, nhead, dropout, noise_level)

# Define optimizer and loss function
optimizer_with_features = torch.optim.Adam(model_with_features.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

# Training loop with features
for epoch in range(EPOCH):
    optimizer_with_features.zero_grad()  # Clear gradients for this training step
    output = model_with_features(train_x_tensor, features_train_tensor)  # Forward pass
    loss = criterion(output, train_y_tensor)  # Calculate loss
    loss.backward()  # Backpropagation
    optimizer_with_features.step()  # Update parameters

    # Print loss or other metrics for monitoring training progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCH}], Loss: {loss.item():.4f}")

# After training, get the predictions
predictions = model_with_features(test_x_tensor, test_features_tensor)
