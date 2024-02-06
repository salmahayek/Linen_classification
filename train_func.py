import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=10, ckpt_name=None):
    """
    Train the given model using the provided data loaders and optimizer for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        valid_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        criterion: The loss function used for training.
        optimizer: The optimizer used for updating model weights.
        device (str): The device (e.g., 'cuda', 'cpu') on which to perform the training.
        epochs (int): The number of epochs to train the model (default: 10).
    """
    best_valid_loss = float('inf')
    best_f1_score = 0.0
    model_path = 'best_model' if not ckpt_name else ckpt_name

    model.to(device)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            dataloader = train_loader if phase == 'train' else valid_loader
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []
            all_outputs = []

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # Store labels, predictions, and outputs for metrics computation
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_outputs.extend(outputs.cpu().detach().numpy())

            # Calculate average loss for the epoch
            epoch_loss = running_loss / len(dataloader.dataset)

            # Calculate precision, recall, F1 score, accuracy, and AUC
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            accuracy = accuracy_score(all_labels, all_preds)
            # auc = roc_auc_score(all_labels, all_outputs[:, 1])

            # Save the best model based on F1 score
            if phase == 'valid' and f1 > best_f1_score:
                best_f1_score = f1
                best_model_path = f"{model_path}.pt"
                torch.save(model.state_dict(), best_model_path)

            # Print training/validation statistics
            print(f"Epoch {phase}: {epoch+1}/{epochs}\t{phase.capitalize()} Loss: {epoch_loss:.4f}\t"
                  f"Precision: {precision:.4f}\tRecall: {recall:.4f}\tF1 Score: {f1:.4f}\t"
                  f"Accuracy: {accuracy:.4f}")
            if phase == 'valid':
                print()
    
    best_model_path = f"{model_path}_{round(best_f1_score, 5)}.pt"
    torch.save(model.state_dict(), best_model_path)
    print(f"Best F1 Score: {best_f1_score}")
    return best_model_path