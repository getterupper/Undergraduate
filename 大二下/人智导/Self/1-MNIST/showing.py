import torch
import torchvision 
from tqdm import tqdm
import matplotlib

import matplotlib.pyplot as plt

from training import Net

device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
																torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
			
BATCH_SIZE = 256
EPOCHS = 10
trainData = torchvision.datasets.MNIST('./data/',train = True,transform = transform,download = True)
testData = torchvision.datasets.MNIST('./data/',train = False,transform = transform)


trainDataLoader = torch.utils.data.DataLoader(dataset = trainData,batch_size = BATCH_SIZE,shuffle = True)
testDataLoader = torch.utils.data.DataLoader(dataset = testData,batch_size = BATCH_SIZE,shuffle = True)

model = Net()

model = torch.load('./model.pth')
lossF = torch.nn.CrossEntropyLoss() 

correct,totalLoss = 0,0
model.train(False)
with torch.no_grad():
		for testImgs,labels in testDataLoader:
				testImgs = testImgs.to(device)
				labels = labels.to(device)
				outputs = model(testImgs)
				loss = lossF(outputs,labels)
				predictions = torch.argmax(outputs,dim = 1)
																
				totalLoss += loss
				correct += torch.sum(predictions == labels)
				
		for j in range(12):
				plt.subplot(3,4,j+1)
				plt.tight_layout()
				plt.imshow(testImgs[j][0], cmap='gray', interpolation='none')
				plt.title("Ground Truth: {}".format(labels[j]))
				plt.xticks([])
				plt.yticks([])

		plt.show()
		
		for j in range(12):
				plt.subplot(3,4,j+1)
				plt.tight_layout()
				plt.imshow(testImgs[j][0], cmap='gray', interpolation='none')
				plt.title("Prediction: {}".format(predictions[j]))
				plt.xticks([])
				plt.yticks([])

		plt.show()
		
