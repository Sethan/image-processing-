import torch
import matplotlib.pyplot as plt
import utils
import dataloaders
import torchvision

from trainer import Trainer

torch.random.manual_seed(0)


class FullyConnectedModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # We are using 28x28 greyscale images.
        num_input_nodes = 28*28
        # Number of classes in the MNIST dataset
        num_classes = 10

        # Define our model
        self.classifier = torch.nn.Sequential(
			torch.nn.ReLU(),
            torch.nn.Linear(num_input_nodes, 64),
			torch.nn.Linear(64,num_classes),
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 28*28)
        out = self.classifier(x)
        return out

# ### Hyperparameters & Loss function

# Hyperparameters
batch_size = 64
learning_rate = .0192
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Model definition
model = FullyConnectedModel()


#copies the weight 9x784 array, for each weight[n] the 1x784 strip is converted to a 28x28 array and saved as a greyscale image
#commented out to avoid interference with the graphing
"""weight = next(model.classiﬁer.children ()).weight.data
for n in range(10):
	t=numpy.zeros((28, 28))
	for x in range(28):
		for y in range(28):
			t[x][y]=weight[n][x*28+y]
	im = plt.imshow(t,cmap='gray')
	plt.savefig("image"+str(n)+".png")"""
# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)
image_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.5,),(0.25,))
])
dataloader_train, dataloader_val = dataloaders.load_dataset(batch_size, image_transform=image_transform)


trainer = Trainer(
  model=model,
  dataloader_train=dataloader_train,
  dataloader_val=dataloader_val,
  batch_size=batch_size,
  loss_function=loss_function,
  optimizer=optimizer
)



train_loss_dict, val_loss_dict = trainer.train(num_epochs)


#This is a repeat to get the results for 4a
class FullyConnectedModel2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # We are using 28x28 greyscale images.
        num_input_nodes = 28*28
        # Number of classes in the MNIST dataset
        num_classes = 10

        # Define our model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_input_nodes, num_classes),
        )

    def forward(self, x):
        # Runs a forward pass on the images
        x = x.view(-1, 28*28)
        out = self.classifier(x)
        return out
		
		
model2 = FullyConnectedModel2()		

optimizer2 = torch.optim.SGD(model2.parameters(),
                            lr=learning_rate)
image_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.5,),(0.25,))
])


dataloader_train2, dataloader_val2 = dataloaders.load_dataset(batch_size, image_transform=image_transform)
trainer2 = Trainer(
  model=model2,
  dataloader_train=dataloader_train2,
  dataloader_val=dataloader_val2,
  batch_size=batch_size,
  loss_function=loss_function,
  optimizer=optimizer2
)
train_loss_dict2, val_loss_dict2 = trainer2.train(num_epochs)



# Plot loss
utils.plot_loss(train_loss_dict, label="Train Loss")
utils.plot_loss(val_loss_dict, label="Test Loss")
utils.plot_loss(train_loss_dict2, label="Train Loss 4a")
utils.plot_loss(val_loss_dict2, label="Test Loss 4a")
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Number of Images Seen")
plt.ylabel("Cross Entropy Loss")
plt.savefig("training_loss.png")

plt.show()
torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_val, model, loss_function)
print(f"Final Test Cross Entropy Loss: {final_loss}. Final Test accuracy: {final_acc}")
