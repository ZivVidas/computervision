import torch
from torch import nn

# Import torchvision
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from FashionMNISTModel0 import FashionMNISTModelV0

from helper_functions import accuracy_fn
from helper_functions import print_train_time,modelEval,test_step,train_step
from timeit import default_timer as timer
# Import tqdm for progress bar
from tqdm.auto import tqdm

print(f"Torch version:{torch.__version__}\ntorchvision version: {torchvision.__version__}")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

image , lable = train_data[0]
# print(train_data.targets[:5])
# print(train_data.data)
# print(train_data.targets)
# print(train_data.classes)



image, label = train_data[0]
sqimage = image.squeeze()
# print(f"Image : {image}")
# print(f"Image squeezed : {sqimage.shape}")
# plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(label)
# plt.show()

class_names = train_data.classes
# Plot more images
torch.manual_seed(42)
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)

# Split data to batchs
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Get the next single batch, which is a list of 32 images
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_batch.shape)

flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)
# print(output.squeeze().shape)

model_0 = FashionMNISTModelV0(input_shape=784,hidden_units=10,output_shape=len(class_names))
model_0.to("cpu")


# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


torch.manual_seed(42)
train_time_start_on_cpu = timer()

epochs = 3
# print(tqdm(range(epochs)))
# xx = range(3)
# print(xx)
# print(range(epochs))
print(model_0.state_dict())
torch.manual_seed(42)
for epoch in tqdm(range(epochs)):
    print(f"Epoch:{epoch}\n----------------")
    train_model0 = train_step(model_0,loss_fn,train_dataloader,optimizer,'cpu')

    eval_model0 = test_step(model_0,test_dataloader,loss_fn,accuracy_fn)
    print(f"name:{eval_model0.model_name}\nlss:{eval_model0.model_loss}\nacc:{eval_model0.model_acc}" )

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,end=train_time_end_on_cpu,device=str(next(model_0.parameters()).device))
    
print(train_model0.model.state_dict())
print(model_0.state_dict())


    

