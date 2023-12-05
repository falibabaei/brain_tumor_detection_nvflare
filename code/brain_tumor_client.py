import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


import nvflare.client as flare
import json

import load_data
from net import TumorNet
import argparse 

 

PATH = "./tumor_classification.pth"
data_split_filename = "/home/se1131/brain_scan/Brain_Tumor_DataSet/Brain_Tumor_DataSet/data_split.json"

with open(data_split_filename, "r") as file:
    data_split = json.load(file)


def main(batch_sz, epochs, lr):
    image_transform = transforms.Compose(
        [
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=(244, 244)),
            transforms.ToTensor(),
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
   
    net = TumorNet()

    # (2) initializes NVFlare client API
    flare.init()
    while flare.is_running():
        client_id = flare.get_site_name()
        client_id = flare.get_site_name()
        (
            train_data,
            train_dataloader,
            valid_data,
            valid_dataloader,
        ) = load_data.load_data(data_split, client_id, image_transform, batch_sz)
        image_datasets = {"train": train_data, "test": valid_data}
        image_dataloaders = {
            "train": train_dataloader,
            "test": valid_dataloader,
        }

        # while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (4) loads model from NVFlare
        net.load_state_dict(input_model.params)

        loss_func = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # (optional) use GPU to speed things up
        net.to(device)
        # (optional) calculate total steps
        steps = epochs * len(train_dataloader)
        for phase in ["train", "test"]:
            if phase == "train":
                        net.train()  # set model to training mode for training phase
            else:
                        net.eval()  # set model to evaluation mode for test phase

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
                    outcomes = net(features)
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

         
        print("Finished Training")

        PATH = "./tumor_brain_net.pth"
        torch.save(net.state_dict(), PATH)

        # (5) wraps evaluation logic into a method to re-use for
        #       evaluation on both trained and received model
        def evaluate(input_weights):
            net = TumorNet()
            net.load_state_dict(input_weights)
            # (optional) use GPU to speed things up
            net.to(device)

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in valid_dataloader:
                    # (optional) use GPU to speed things up
                    images, labels = data[0].to(device), data[1].to(
                        device
                    )
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(
                f"Accuracy of the network on the test images: {100 * correct // total} %"
            )
            return 100 * correct // total

        # (6) evaluate on received model for model selection
        accuracy = evaluate(input_model.params)
        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for brain tumor detection using nvflare.")

 
    parser.add_argument("--batch_sz", type=int, default=None, help="Specify the batch size")
    parser.add_argument("--epochs", type=int, default=None, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")

 
    args = parser.parse_args()

 
    main(batch_sz=args.batch_sz, epochs= args.epochs, lr=args.lr)
