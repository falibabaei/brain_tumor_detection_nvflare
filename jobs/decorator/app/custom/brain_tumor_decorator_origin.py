import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import nvflare.client as flare

import json

from load_data import load_data
from net import TumorNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    PATH = "./tumor_classification.pth"
    data_split_filename = "/home/se1131/brain_scan/Brain_Tumor_DataSet/Brain_Tumor_DataSet/data_split.json"

    with open(data_split_filename, "r") as file:
        data_split = json.load(file)
    # (2) initializes NVFlare client API
    flare.init()

    image_transform = transforms.Compose(
        [
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=(244, 244)),
            transforms.ToTensor(),
        ]
    )
    client_id = flare.get_site_name()
    (
        train_data,
        train_dataloader,
        valid_data,
        valid_dataloader,
    ) = load_data(data_split, client_id, image_transform)
    print(f"{client_id} has {len(train_dataloader)} trained images")
    image_datasets = {"train": train_data, "test": valid_data}
    image_dataloaders = {
        "train": train_dataloader,
        "test": valid_dataloader,
    }
    loss_func = nn.BCELoss()
    model = TumorNet()

    # (3) decorates with flare.train and load model from the first argument
    # wraps training logic into a method
    @flare.train
    def train_model(input_model=None, epochs=2, lr=0.001):
        """Return the trained model and train/test accuracy/loss"""
        # if not do_training:
        #    return None, None

        model.load_state_dict(input_model.params)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }
        for e in range(1, epochs + 1):
            print("Epoch {}/{}".format(e, epochs))
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # set model to training mode for training phase
                else:
                    model.eval()  # set model to evaluation mode for test phase

                running_loss = 0.0  # record the training/test loss for each epoch
                running_corrects = 0  # record the number of correct predicts by the model for each epoch

                for features, labels in image_dataloaders[phase]:
                    # send data to gpu if possible
                    features = features.to(device)
                    labels = labels.to(device)

                    # reset the parameter gradients after each batch to avoid double-counting
                    optimizer.zero_grad()

                    # forward pass
                    # set parameters to be trainable only at training phase
                    with torch.set_grad_enabled(phase == "train"):
                        outcomes = model(features)
                        pred_labels = (
                            outcomes.round()
                        )  # round up forward outcomes to get predicted labels
                        labels = labels.unsqueeze(1).type(torch.float)
                        loss = loss_func(
                            outcomes, labels
                        )  # calculate loss

                        # backpropagation only for training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # record loss and correct predicts of each bach
                    running_loss += loss.item() * features.size(0)
                    running_corrects += torch.sum(
                        pred_labels == labels.data
                    )

                # record loss and correct predicts of each epoch and stored in history
                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(
                    image_datasets[phase]
                )

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(
                        phase, epoch_loss, epoch_acc
                    )
                )
                history[phase + "_loss"].append(epoch_loss)
                history[phase + "_acc"].append(epoch_acc)
        torch.save(model.state_dict(), PATH)
        # (4) construct trained FL model
        output_model = flare.FLModel(params=model.cpu().state_dict())
        return output_model

    def evaluate(input_weights=None):
        """Return the trained model and train/test accuracy/loss"""
        # if not do_training:
        #    return None, None
        model = TumorNet()
        model.load_state_dict(input_weights)
        model.eval()  # set model to evaluation mode for test phase
        running_corrects = 0
        for features, labels in valid_dataloader:
            # send data to gpu if possible
            features = features.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outcomes = model(features)
                pred_labels = (
                    outcomes.round()
                )  # round up forward outcomes to get predicted labels
                labels = labels.unsqueeze(1).type(torch.float)

                running_corrects += torch.sum(
                    pred_labels == labels.data
                )

            # record loss and correct predicts of each epoch and stored in history
            acc = running_corrects.double() / len(valid_dataloader)

        return 100 * acc

    # (5) decorates with flare.evaluate and load model from the first argument
    @flare.evaluate
    def fl_evaluate(input_model=None):
        return evaluate(input_weights=input_model.params)

    while flare.is_running():
        # (6) receives FLModel from NVFlare
        input_model = flare.receive()
        # (7) call fl_evaluate method before training
        #       to evaluate on the received/aggregated model
        global_metric = fl_evaluate(input_model)
        print(
            f"Accuracy of the global model on the test images: {global_metric} %"
        )
        # call train method

        train_model(input_model, epochs=3, lr=0.001)
        metric = evaluate(input_weights=torch.load(PATH))
        print(
            f"Accuracy of the trained model on the test images: {metric} %"
        )


if __name__ == "__main__":
    main()
