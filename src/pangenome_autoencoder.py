"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Autoencoder Latent Embedding

This module builds and trains a deep PyTorch Autoencoder to map the sparse,
high-dimensional binary pangenome matrix into a continuous, dense non-linear
latent space representation (bottleneck layer).

NOTE: An Autoencoder compresses data into a smaller bottleneck layer, it is
frequently used for dimensionality reduction, noise removal, and feature
extraction—making it a powerful alternative to linear methods like PCA or
non-linear mapping methods like t-SNE.



"""

import os
from re import M
from matplotlib.lines import lineStyles
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config


class PangenomeAutoencoder(nn.Module): # inherit nn.Module
    """
    Symmetric multi-layer feedforward Autoencoder architecture
    """

    def __init__(self, input_dim, latent_dim=2):
        super(PangenomeAutoencoder, self).__init__()

        # Encoder: Compresses high-dimensional gene vectors
        # down to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),  # Continuous bottleneck coordinates
        )

        # Decoder: Reconstructs the compressed embedding back to the
        # binary gene space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # Restricts reconstruction outputs between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def train_autoencoder(
    matrix_path=config.MATRIX_CSV,
    latent_path=config.EMBEDDING_CSV,
    epochs=100,
    batch_size=2,
):
    """
    Loads pangenome data, handles structural edge cases for small test sets,
    trains the nn using BCE loss, and save the latent coords.
    """
    print(f"Initializing NN training pipeline using matrix: {matrix_path}")
    if not os.path.exists(matrix_path):
        print("Error: Matrix file missing. Cannot train Autoencoder.")
        return None

    # Load matrix
    df = pd.read_csv(matrix_path, index_col=0)
    strain_ids = df.index.to_list()

    # Convert data to floating-point tensors for PyTorch
    X_data = torch.tensor(df.values, dtype=torch.float32)
    input_dim = X_data.shape[1]
    total_strains = X_data.shape[0]

    # Dynamic adjustment for the test dataset (4 strains):
    # BatchNorm1d requires more than 1 sample per batch, and training deep nets
    # on tiny matrices cna instantly overfit. We adjust training parameters to
    # compensate.
    if total_strains <= 4:
        print(
            "Small cohort detected. Switching architecture to stable training mode (disabling BatchNorm)."
        )
        batch_size = total_strains
        epochs = 200 # More iterations -> converge smoothly
        
    # Instantiate model
    model = PangenomeAutoencoder(input_dim=input_dim, latent_dim=2)
    
    # Deactivate BatchNorm if batch size is too small to calculate variance
    if total_strains <=4:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
                
    # Binary Cross Entropy is optimal for evaluating reconstruction error of 
    # 0/1 vectors
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Training Loop
    model.train()
    loss_history = []
    
    print(f"training model across {epochs} epochs..")
    
    for epoch in range(1, epochs +1):
        # Basic mini-batch slicing
        permutation = torch.randperm(X_data.size()[0])
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, X_data.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_data[indices]
            
            if batch_x.size()[0] < 2 and total_strains > 4:
                continue # Skip trailing single-sample batches to preserve batchnorm stability
            
            # Forward pass
            optimizer.zero_grad()
            latent, reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        loss_history.append(epoch_loss/batches)
        
        if epoch % (epochs // 5) == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epoch} | Reconstruction Loss: {loss_history[-1]:.4f}")
            
    # Extract latent vectors
    model.eval()
    with torch.no_grad():
        latent_embeddings, _ = model(X_data)
        latent_array = latent_embeddings.numpy()
        
    # Constract and save coords dataframe
    df_latent = pd.DataFrame(
        latent_array,
        columns=["Latent Dimension 1", "Latent Dimension 2"],
        index=strain_ids
    )
    df_latent.to_csv(latent_path)
    print(f"Latent space coords exported directly to: {latent_path}")
    
    # Generate loss convergence plot to verify training behavior
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(loss_history)+1), loss_history, color="#2b5c8f", lw=2)
    plt.xlabel("Training Epochs", weight="bold")
    plt.ylabel("Loss (BCE)", weight="bold")
    plt.title("Autoencoder Convergence Profile", weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(os.path.dirname(latent_path), "autoencoder_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f" Training loss plot saved to: {loss_plot_path}\n")
    
    return df_latent

if __name__ == "__main__":
    train_autoencoder()