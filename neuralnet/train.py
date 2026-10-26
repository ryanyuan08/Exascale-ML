import numpy as np
import joblib
import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import r2_score

from constants import *
from nn import *

def cpu_train(X, y):
    device = get_device()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_scaler = StandardScaler(with_mean=False)  # divide by std only, so cq >= 0 <=> scaled target >= 0
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    train_dataset = ViscosityDataset(X_train_scaled, y_train_scaled)
    test_dataset = ViscosityDataset(X_test_scaled, y_test_scaled)

    # pin_memory=False: no benefit on CPU (only useful when sending data to a GPU)
    # num_workers=cpu_count(): let all cores share the data loading work
    train_loader = DataLoader(train_dataset, batch_size=CPU_BATCH_SIZE, shuffle=True,
                              num_workers=cpu_count(), pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=CPU_BATCH_SIZE, shuffle=False,
                             num_workers=cpu_count(), pin_memory=False)

    model = ArtificialViscosityNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_losses, test_losses = [], []
    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                test_loss += criterion(model(batch_X), batch_y)

        avg_train = (train_loss / len(train_loader)).item()
        avg_test = (test_loss / len(test_loader)).item()
        train_losses.append(avg_train)
        test_losses.append(avg_test)
        scheduler.step(avg_test)
        pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.6f} | Test: {avg_test:.6f}")

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            all_predictions.append(model(batch_X).numpy().ravel())
    y_predicted = np.concatenate(all_predictions)
    y_true = y_test_scaled

    print(f"\nTest R² = {r2_score(y_true, y_predicted):.6f}")

    print(train_losses)
    print()
    print(test_losses)
    plot_losses(train_losses, test_losses)

    return model, X_scaler, y_scaler


def gpu_train(X, y):
    device = get_device()
    assert device.type == "cuda" or device.type == "mps", "gpu_train requires a gpu device"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X
    del y

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train).astype(np.float32)
    del X_train
    X_test_scaled = X_scaler.transform(X_test).astype(np.float32)
    del X_test

    y_scaler = StandardScaler(with_mean=False)  # divide by std only, so cq >= 0 <=> scaled target >= 0
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    del y_train
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)
    del y_test

    # Move full dataset to GPU once — eliminates all CPU→GPU transfers during training
    X_train_t = torch.from_numpy(X_train_scaled).to(device)
    y_train_t = torch.from_numpy(y_train_scaled).view(-1, 1).to(device)
    X_test_t  = torch.from_numpy(X_test_scaled).to(device)
    y_test_t  = torch.from_numpy(y_test_scaled).view(-1, 1).to(device)
    n_train, n_test = X_train_t.shape[0], X_test_t.shape[0]
    del X_train_scaled
    del X_test_scaled
    del y_train_scaled


    model = ArtificialViscosityNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_losses, test_losses = [], []
    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        model.train()
        train_loss = 0
        n_train_batches = 0
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, GPU_BATCH_SIZE):
            idx = perm[start:start + GPU_BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            train_loss += loss
            n_train_batches += 1

        model.eval()
        test_loss = 0
        n_test_batches = 0
        with torch.no_grad():
            for start in range(0, n_test, GPU_BATCH_SIZE):
                test_loss += criterion(model(X_test_t[start:start + GPU_BATCH_SIZE]),
                                       y_test_t[start:start + GPU_BATCH_SIZE])
                n_test_batches += 1

        avg_train = (train_loss / n_train_batches).item()
        avg_test = (test_loss / n_test_batches).item()
        train_losses.append(avg_train)
        test_losses.append(avg_test)
        scheduler.step(avg_test)
        pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.6f} | Test: {avg_test:.6f}")

    model.eval()
    with torch.no_grad():
        y_predicted = model(X_test_t).cpu().numpy().ravel()
    y_true = y_test_scaled

    print(f"\nTest R² = {r2_score(y_true, y_predicted):.6f}")

    print(train_losses)
    print()
    print(test_losses)
    plot_losses(train_losses, test_losses)

    return model, X_scaler, y_scaler


if __name__ == "__main__":
    print(f"Loading feature matrices from {FEATURE_MATRICES_PATH}...")
    data = np.load(FEATURE_MATRICES_PATH)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    print(f"X shape: {X.shape} | y shape: {y.shape}")

    train_function = gpu_train if torch.cuda.is_available() or torch.backends.mps.is_available() else cpu_train
    model, X_scaler, y_scaler = train_function(X, y)

    OUTPUT_DIR.mkdir(exist_ok=True)

    X_mean = torch.tensor(X_scaler.mean_,  dtype=torch.float32)
    X_std  = torch.tensor(X_scaler.scale_, dtype=torch.float32)
    # The y scaler only divides by std (with_mean=False), so nothing is added back.
    # sklearn still fills in mean_ in that case, so it must NOT be used here.
    y_mean = torch.tensor(0.0, dtype=torch.float32)
    y_std  = torch.tensor(y_scaler.scale_, dtype=torch.float32)

    print(f"X_mean: {X_mean}")
    print(f"X_std:  {X_std}")
    print(f"y_mean: {y_mean}")
    print(f"y_std:  {y_std}")

    model = model.cpu()
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    def scaled_forward(x):
        return model((x - X_mean) / X_std) * y_std + y_mean

    example = torch.randn(1, X.shape[1])
    with torch.no_grad():
        scripted = torch.jit.trace(scaled_forward, example)

    scripted.save(str(OUTPUT_DIR / "model.pt"))
    print(f"Saved model to {OUTPUT_DIR}")