import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


import nvflare.client as flare
import json

import load_data
from net import TumorNet

epochs = 1

PATH = "./tumor_classification.pth"
data_split_filename = "/home/se1131/brain_scan/Brain_Tumor_DataSet/Brain_Tumor_DataSet/data_split.json"

with open(data_split_filename, "r") as file:
    data_split = json.load(file)


def main():
    image_transform = transforms.Compose(
        [
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=(244, 244)),
            transforms.ToTensor(),
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    epochs = 2
    net = TumorNet()

    # (2) initializes NVFlare client API
    flare.init()
    client_id = flare.get_site_name()
    _, trainloader, _, testloader = load_data.load_data(
        data_split, client_id, image_transform
    )

    # while flare.is_running():
    # (3) receives FLModel from NVFlare
    input_model = flare.receive()
    print(f"current_round={input_model.current_round}")

    # (4) loads model from NVFlare
    net.load_state_dict(input_model.params)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # (optional) use GPU to speed things up
    net.to(device)
    # (optional) calculate total steps
    steps = epochs * len(trainloader)
    for epoch in range(
        epochs
    ):  # loop over the dataset multiple times
        print("Epoch {}/{}".format(epochs, epochs))
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # (optional) use GPU to speed things up
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outcomes = net(inputs)
            pred_labels = (
                outcomes.round()
            )  # round up forward outcomes to get predicted labels
            labels = labels.unsqueeze(1).type(torch.float)
            loss = criterion(outcomes, labels)  # calculate loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred_labels == labels.data)

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = running_corrects.double() / len(trainloader)
        print("  Loss: {:.4f}  ".format(epoch_loss))
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
            for data in testloader:
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
            f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
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
    main()
