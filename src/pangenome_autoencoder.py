"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Autoencoder Latent Embedding

This module builds and trains a deep PyTorch Autoencoder to map the sparse,
high-dimensional binary pangenome matrix into a continuous, dense non-linear
latent space representation.

This approach is hypothetical?

                     [ SEPARATION PHASE ]                       [ RECONSTRUCTION PHASE ]
Raw Strain Row                                Bottleneck Space                                  Reconstructed Row
[1, 0, 1, ..., 0]  ─────────► ENCODER ─────────►  ( X , Y )  ─────────► DECODER ─────────► [0.94, 0.02, 0.89, ..., 0.01]
(5900+ Gene Realities)                       (Hypothesized Group)                    (Model Confidence Vectors)

The encoder looks at the global variance of all 5,900+ accessory genes simultaneously.
It compresses this high-dimensional information down into two continuous variables.

The decoder acts as the validation mechanism for the encoder's groupings. It takes
those two coordinate points and tries to reconstruct the original 5,900-gene binary
blueprint.

The output is a vector of decimal probabilities. These decimals reflect the model's
confidence based on the local genetic density of that cluster. If the decoder can
reconstruct the original genome with high confidence, it proves that the encoder's
2D map successfully captured the true, underlying biological rules of the population.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config


class PangenomeAutoencoder(nn.Module):  # inherit nn.Module
    """
    Symmetric multi-layer feedforward Autoencoder architecture
    """

    def __init__(self, input_dim, latent_dim=4):
        super(PangenomeAutoencoder, self).__init__()

        # Encoder: Compresses high-dimensional gene vectors down to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
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
            #nn.Dropout(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # Restricts outputs to probabilities between 0 and 1
        )
        """
        The outcome of the decoder represent the confidence level that 
        a specifc gene (i.e zur, pectinase...) should be present, based 
        on where the NN placed that strain on the 2D map
        
        Possible interpretation?
        
        When the decoder outputs a decimal like 0.85 for a specific gene 
        in a specific strain, it is not making a random guess. Mathematically, 
        that 0.85 means:
        
        "Given the 2D coordinate where this strain has been placed, 85% of the 
        strains in the global training pool that sit at or near this exact 
        spatial location possess this gene."
        
        Because we are forcing the network to map 80 strains onto a tiny 2D map, 
        it is mathematically impossible for every strain to have its own unique 
        spot with perfect recreation. Strains with highly similar accessory 
        profiles are forced to crowd together into shared local neighborhoods.
        
        If a neighborhood is composed of 10 strains, and 8 of them contain a 
        specific penicillin-resistance gene while 2 lack it, the most mathematically 
        accurate guess the decoder can make for any strain landing in that zone is 
        0.80. It is a direct reflection of the local genetic density of that cluster.
        """

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def train_autoencoder(
    matrix_path=config.MATRIX_CSV,
    latent_path=config.EMBEDDING_CSV,
    epochs=100,
    batch_size=2,
    latent_dim=4,
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

    # Dynamic adjustment for the small dataset (ie. 4 strains):
    if total_strains <= 4:
        print(
            "Small cohort detected. Switching architecture to stable training mode (disabling BatchNorm)."
        )
        batch_size = total_strains
        epochs = 200  # More iterations -> converge smoothly

    # Instantiate model
    model = PangenomeAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Binary Cross Entropy is optimal for evaluating reconstruction error of
    # 0/1 vectors
    criterion = (
        nn.BCELoss()
    )  # is used to compute how far off the decoder's decimal predictions are from the true 0/1 genomic inputs
    #optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0)

    # Training Loop
    model.train()

    # Deactivate BatchNorm if batch size is too small to calculate variance
    if total_strains <= 4:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    loss_history = []
    print(f"training model across {epochs} epochs..")

    for epoch in range(1, epochs + 1):
        # Basic mini-batch slicing
        permutation = torch.randperm(
            X_data.size()[0]
        )  # Shuffle all strain row indexes randomly
        epoch_loss = 0.0
        batches = 0

        for i in range(0, X_data.size()[0], batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = X_data[indices]

            if batch_x.size()[0] < 2 and total_strains > 4:
                continue  # Skip trailing single-sample batches to preserve batchnorm stability

            # Forward pass
            optimizer.zero_grad()  # Clear out old gradients
            latent, reconstructed = model(batch_x)  # run the forward pass
            loss = criterion(
                reconstructed, batch_x
            )  # grade the model's performation on this batch

            # Backward pass
            loss.backward()  # trace the error background
            optimizer.step()  # updates the weights

            epoch_loss += loss.item()
            batches += 1

        loss_history.append(epoch_loss / batches)

        if epoch % (epochs // 5) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epoch} | Reconstruction Loss: {loss_history[-1]:.4f}"
            )

    # Extract latent vectors
    model.eval()
    with torch.no_grad():
        latent_embeddings, _ = model(X_data)
        latent_array = latent_embeddings.numpy()

    # Constract and save coords dataframe
    latent_columns = [f"Latent Dimension {i}" for i in range(1, latent_dim + 1)]
    df_latent = pd.DataFrame(
        latent_array,
        columns=latent_columns,
        index=strain_ids,
    )
    df_latent.to_csv(latent_path)
    print(f"Latent space coords ({latent_dim}D) exported directly to: {latent_path}")

    # Generate loss convergence plot to verify training behavior
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="#2b5c8f", lw=2)
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
