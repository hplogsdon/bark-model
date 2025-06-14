import torch


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"epoch: {i + 1}")
        train_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-------")
    print("Finished Training")


def train_epoch(model, data_loader, loss_fn, optimiser, device):
    loss: any
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropogate the loss and update model weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"Loss: {loss.item()}")
