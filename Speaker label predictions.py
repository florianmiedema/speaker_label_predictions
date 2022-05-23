## Loading packages
import pandas as pd
import json
import torch
import torchaudio
import random
import datetime
from torchaudio import transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.nn import init
from matplotlib import pyplot as plt 
import numpy as np
from torchviz import make_dot

## Setting seed for reproducibility
torch.manual_seed(222)

## Setting timer
start = datetime.datetime.now()

## Defining the variable device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

## Setting absolute PC path to audio file folder (end path with /)
path = '/'
## and path to store final files
save_path = path + 'Hand in files/'


######################## Step 1: Prepare Training Data ########################


## Loading the training labels
with open(path + 'train.json') as files:
    df_dict = json.load(files)

## Converting the json labels to a dataframe
df = pd.DataFrame(list(df_dict.items()))
df.columns = ['slice_file_name', 'label']
df['label'] = df['label'].astype(int)

## Saving all file names
file_names = list(df.iloc[:,0])

## Adjusting the file names to partial file paths
df['file'] = "data/data/train/" + df['slice_file_name']
df = df[['file', 'label']]

## Checking how many classes in dataset:
no_classes = df['label'].nunique()

## Subtracting 1 from all labels to avoid CUDA runtime errors
df['label'] = df['label'] - 1


##################### Step 2: Creating the Torch Dataset ######################


## Creating functions to modify the audio inputs 

class ProcessAudio():
    ## function i. loads the data into torch format
    @staticmethod
    def open(wav_file):
        signal, sample_rate = torchaudio.load(wav_file)
        return (signal, sample_rate)
    
    @staticmethod
    ## function ii. padds the audio with zeroes to make all files of equal size
    def padding(audio):
        signal, sample_rate = audio
        rows, signal_length = signal.shape
        max_length = 200000
    
        if (signal_length > max_length):
          # Truncate the signal to the given length
          signal = signal[:,:max_length]
        
        elif (signal_length < max_length):
          padding_begin_length = random.randint(0, max_length - signal_length)
          padding_end_length = max_length - signal_length - padding_begin_length
          padding_start = torch.zeros((rows, padding_begin_length))
          padding_end = torch.zeros((rows, padding_end_length))
        
          signal = torch.cat((padding_start, signal, padding_end), 1)
      
        return (signal, sample_rate)
    
    @staticmethod
    def timeshift(audio, shift_limit):
        ## function iii. shifts the audio files
        signal,sample_rate = audio
        _, signal_length = signal.shape
        shifted = int(random.random() * shift_limit * signal_length)
        return (signal.roll(shifted), sample_rate)
    
    @staticmethod
    def spectrogram(audio, mels=64, fft=1024, hop_length=None):
        ## function iv. converts audio to melspectrogram
        signal,sample_rate = audio
        maximum = 80
        spectrogram = transforms.MelSpectrogram(sample_rate, n_fft=fft, hop_length=hop_length, n_mels=mels)(signal)
        spectrogram = transforms.AmplitudeToDB(top_db=maximum)(spectrogram)
        return (spectrogram)
    
    @staticmethod
    def augmentation(spectrogram, maximum_masking=0.1, frequency_masks=1, time_masks=1):
        ## function v. augments the melspectrograms
        _, mels, steps = spectrogram.shape
        mask_value = spectrogram.mean()
        augmented_spectrogram = spectrogram
        frequency_mask = maximum_masking * mels
    
        for _ in range(frequency_masks):
            augmented_spectrogram = transforms.FrequencyMasking(frequency_mask)(augmented_spectrogram, mask_value)
            time_mask_param = maximum_masking * steps
                
        for _ in range(time_masks):
          augmented_spectrogram = transforms.TimeMasking(time_mask_param)(augmented_spectrogram, mask_value)
        
        return augmented_spectrogram


## The following class and functions will shape the final Pytorch dataset (Ketan Doshi, 2021).


class data_processing(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.shifting_percentage = 0.4
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, i):
        fullpath = '/' ## same path as 'path' variable
        wav_file = fullpath + self.df.loc[i, 'file']
        
        ## Extracting data from data file
        signal, sample_rate = torchaudio.load(wav_file)
        
        ## Saving the class id of the audio file 
        label_id = self.df.loc[i, 'label']
        
        audio = signal, sample_rate
        duration_audio = ProcessAudio.padding(audio)
        shifted_audio = ProcessAudio.timeshift(duration_audio, self.shifting_percentage)
        specgram = ProcessAudio.spectrogram(shifted_audio, mels=64, fft=1024, hop_length=None)
        augmentated_spectrogram = ProcessAudio.augmentation(specgram, maximum_masking=0.1, frequency_masks=2, time_masks=2)
    
        return augmentated_spectrogram, label_id
        
    
##################### Step 3: Inspecting the Dataset ##########################


## Creating the dataset 
df = data_processing(df, path)     

## inspect
print('\nFirst row of data set: ', next(iter(df)), '\n')
# Print how many items are in the data set
print('Length of data set: ', len(df), '\n')
# if test above are ok -> data is loaded correctly


##################### Step 4: Splitting the Dataset ##########################


## Randomly splitting the training set and validation set 80%/20%
nrows = len(df)
nrows_train = round(nrows * 0.8)
nrows_val = nrows - nrows_train
trainingset, validationset = random_split(df, [nrows_train, nrows_val])

## Create training and validation data loaders. 16 is the max that our machine can handle.
dataloader_train = DataLoader(trainingset, batch_size=16, shuffle=True)
dataloader_val = DataLoader(validationset, batch_size=16, shuffle=False)
    
## take data sample
sample = next(iter(dataloader_train))
plt.plot(sample[0].reshape(-1).cpu().numpy())


########################## Step 5a: Create Model ###############################


## The training and validation procedures (Sections 5a, 6a & 7) are Inspired on (Ketan Doshi, 2021).


class NetworkArchitecture (nn.Module):
    def __init__(self):
        super().__init__()
        convolution = []

        ## Layer 1 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        convolution += [self.conv1, self.relu1, self.bn1]

        ## Layer 2 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        convolution += [self.conv2, self.relu2, self.bn2]

        ## Layer 3 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        convolution += [self.conv3, self.relu3, self.bn3]

        ## Layer 4 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        convolution += [self.conv4, self.relu4, self.bn4]
        
        ## Layer 5 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        convolution += [self.conv4, self.relu4, self.bn4]

        ## Linear activation
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features= no_classes)
        self.conv = nn.Sequential(*convolution)
  
    ## Feed Forward
    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        return x

## compiling the network
TrainedNetwork = NetworkArchitecture()
TrainedNetwork = TrainedNetwork.to(device)

# Device still gpu?
next(TrainedNetwork.parameters()).device
print(TrainedNetwork)


######################## Step 5b: Visualize Model #############################


tens = sample[0].to(device)
yhat = TrainedNetwork(tens)
make_dot(yhat, params=dict(list(TrainedNetwork.named_parameters()))).render(save_path + 'DL_architecture', format="png")


########################### Step 6a: Training #################################


## for the first 5 classes, monitor how many correct predictions there are at each epoch
classes_5 = {'a': [], 'b': [], 'c': [], 'd': [], 'e': []}

def insample(model, dataloader_train, iterations):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                steps_per_epoch=int(len(dataloader_train)),
                                                epochs=iterations,
                                                anneal_strategy='linear')

  ## Loop for epochs
  for epoch in range(iterations):
    track_loss = 0.0
    correct_prediction = total_prediction = 0
    store_predictions = []
    store_truelabels = []

    ## Iterate over batches 
    for i, data in enumerate(dataloader_train):
        ## store data on GPU
        inputs = data[0].to(device)
        labels = data[1].type(torch.LongTensor)
        labels = labels.to(device)

        ## Standardize
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        optimizer.zero_grad()

        ## FeedForward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ## Tracking loss
        track_loss += loss.item()

        ## Obtain predictions
        _, prediction = torch.max(outputs,1)
        store_predictions.append(prediction)
        store_truelabels.append(labels)
        
        ## Obtain prediction score
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
    
    ## Track performance 
    batch_amount = len(dataloader_train)
    avg_loss = track_loss / batch_amount
    acc = correct_prediction/total_prediction
    
    ## specific class accuracy at each epoch
    inspect = pd.Series(np.concatenate([x.cpu().detach().numpy() for x in store_predictions]))
    trueclass = pd.Series(np.concatenate([x.cpu().detach().numpy() for x in store_truelabels]))
    predframe = pd.DataFrame({'predicted': inspect, 'true': trueclass})
    
    ## check for class 1,2,3,4,5 the number of correct/total
    ## 1:
    acc1 = predframe[predframe.predicted == 1]
    acc1 = sum(acc1['predicted'] == acc1['true']) / len(acc1)
    classes_5['a'].append(acc1)
    ## 2:
    acc2 = predframe[predframe.predicted == 2]
    acc2 = sum(acc2['predicted'] == acc2['true']) / len(acc2)
    classes_5['b'].append(acc2)
    ## 3:
    acc3 = predframe[predframe.predicted == 3]
    acc3 = sum(acc3['predicted'] == acc3['true']) / len(acc3)
    classes_5['c'].append(acc3)
    ## 4:
    acc4 = predframe[predframe.predicted == 4]
    acc4 = sum(acc4['predicted'] == acc4['true']) / len(acc4)
    classes_5['d'].append(acc4)
    ## 5:
    acc5 = predframe[predframe.predicted == 5]
    acc5 = sum(acc5['predicted'] == acc5['true']) / len(acc5)
    classes_5['e'].append(acc5)
    
    ## print full accuracy at each epoch
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  ## indicate that training is finished
  print('Finished Training')
  
## call function with specified epochs
iterations= 10
insample(TrainedNetwork, dataloader_train, iterations)


########################### Step 6b: Training #################################


## writing dictionary to dataframe
plotframe = pd.DataFrame.from_dict(classes_5)

## save to PC
plotframe.to_csv(path + 'accuracy_plot.csv')

## create xaxis
x = np.arange(0,11)

## insert zeroes at start
plotframe1 = pd.DataFrame([[0] * len(plotframe.columns)], columns=plotframe.columns)
plotframe = plotframe1.append(plotframe, ignore_index=True)

## create frame filled with ones
dupli = plotframe.copy()
dupli[['a', 'b', 'c', 'd', 'e']] = 1

## subtract frames to create error rate from accuracy scores
plotframe = dupli - plotframe

## design plot
plt.plot(x, plotframe['a'], label = "Speaker ID = 1")
plt.plot(x, plotframe['b'], label = "Speaker ID = 2")
plt.plot(x, plotframe['c'], label = "Speaker ID = 3")
plt.plot(x, plotframe['d'], label = "Speaker ID = 4")
plt.plot(x, plotframe['e'], label = "Speaker ID = 5")
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.legend()
plt.show()


###################### Step 7: Validation Set Performance #####################


# Checking model performance 
def validate(model, dataloader_val):
  correct_prediction = 0
  total_prediction = 0

  with torch.no_grad():
    for data in dataloader_val:
      ## Store data on GPU
      inputs = data[0].to(device)
      labels = data[1].type(torch.LongTensor)
      labels = labels.to(device)

      ## Standardize
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      ## Save predictions
      outputs = model(inputs)
      _, prediction = torch.max(outputs,1)
      
      ## Assess performance
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
  
  # Printing the accuracy and total number of predictions 
  acc = 1 - correct_prediction/total_prediction 
  print(f'Error Rate: {acc:.2f}, Total items: {total_prediction}')
  
## Try model on the validation set to inspect generalizability 
validate(TrainedNetwork, dataloader_val)


################### Step 8: Loading and Modifying Test Data ###################


## Loading the test labels
with open(path + 'test.json') as test:
    df_dict = json.load(test)

## Converting the json labels to a dataframe
df_test = pd.DataFrame(list(df_dict.items()))
df_test.columns = ['slice_file_name', 'label']

## fill test IDs with zeroes
df_test['label'] = 0

## Adjusting the file names to partial file paths
df_test['file'] = "data/data/test/" + df_test['slice_file_name']

## saving backup of dataframe (file names)
df_test_backup = df_test['slice_file_name']

## Creating the dataset 
df_test = data_processing(df_test, path)     

## load test data with custom dataloader
df_test = DataLoader(df_test, batch_size=16, shuffle=False)


#################### Step 9: Getting the Final Predictions ####################


## empty list to store 10.000 predicted labels
test_predictions = []

def oos_predictor(model, test):
  
  with torch.no_grad():
    for data in test:
        
      ## Select inputs
      inputs = data[0].to(device)
      
      ## Standardize them
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      ## Obtain predicted labels
      outputs = model(inputs)

      ## Extract these
      _, prediction = torch.max(outputs,1)
      test_predictions.append(prediction.cpu().detach().numpy())
      
  return test_predictions
  
## activating the model
oos_predictor(TrainedNetwork, df_test)

## concatenate predictions into a single array
final_predictions = np.concatenate(test_predictions, axis=0)

## add +1 to all predictions as we subtracted 1 earlier (to avoid runtime errors)
final_predictions = final_predictions + 1

## convert to pandas series
finalframe = pd.concat([df_test_backup, pd.Series(final_predictions)], axis = 1)
finalframe.columns = ['file', 'label']

## both columns to string type
finalframe['file'] = finalframe['file'].astype(str)
finalframe['label'] = finalframe['label'].astype(str)

## Closing timer
end = datetime.datetime.now()
print(end-start)


################### Step 10: Converting File to Json Format ###################


convert = {}
for i in range(len(finalframe)):
    convert[finalframe.iloc[i,0]] = finalframe.iloc[i,1]
    
# Exporting the dictionary as a json file 
with open(save_path + 'test.json', 'w') as write_file:
    json.dump(convert, write_file, sort_keys=True)
