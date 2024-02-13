import torch
import torch.nn.functional as F


def best_predict(input, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    X = input

    # print(X.shape)
    # print(model)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        immediate_solution = model(X)

        # remove predictions for values already known
        mask = (X.max(2).values == 0).type(torch.int).unsqueeze(2)
        X_update = X.clone()

        while mask.sum() > 0:
            preds = model(X_update)

            mask = (X_update.max(2).values == 0).type(torch.float).unsqueeze(2)

            preds_updated = preds + (torch.nan_to_num((mask - 1) * float("Inf")))

            # get indices of values that the model is most certain about for each batch
            # this is the idx of batches which still have values to fill out
            idx = mask.nonzero(as_tuple=True)[0].unique()
            certain_idx = preds_updated.flatten(1).argmax(1)  # shape = [64]
            certain_cells = certain_idx // 9

            certain_numbers = preds_updated[idx, certain_cells[idx]].argmax(1) + 1
            certain_numbers_one_hot = F.one_hot(certain_numbers, 10)[:, 1:]

            # replace the most certain numbers in X
            X_update[idx, certain_cells[idx]] = certain_numbers_one_hot.type(
                torch.float
            )

        return X_update
