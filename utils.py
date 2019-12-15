# @Author: Ibrahim Salihu Yusuf <Ibrahim>
# @Date:   2019-12-10T11:24:45+02:00
# @Email:  sibrahim1396@gmail.com
# @Project: Audio Classifier
# @Last modified by:   Ibrahim
# @Last modified time: 2019-12-12T13:55:29+02:00



import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn


class AudioDataset(Dataset):
    """
    A rapper class for the UrbanSound8K dataset.
    """

    def __init__(self, file_path, audio_paths, folds):
        """
        Args:
            file_path(string): path to the audio csv file
            root_dir(string): directory with all the audio folds
            folds: integer corresponding to audio fold number or list of fold number if more than one fold is needed
        """
        self.audio_file = pd.read_csv(file_path)
        self.folds = folds
        self.audio_paths = glob.glob(audio_paths + '/*' + str(self.folds) + '/*')



    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        audio_path = self.audio_paths[idx]
        audio, rate = torchaudio.load(audio_path, normalization=True)
        audio = audio.mean(0, keepdim=True)
        c, n = audio.shape
        zero_need = 160000 - n
        audio_new = F.pad(audio, (zero_need //2, zero_need //2), 'constant', 0)
        audio_new = audio_new[:,::5]

        #Getting the corresponding label
        audio_name = audio_path.split(sep='/')[-1]
        labels = self.audio_file.loc[self.audio_file.slice_file_name == audio_name].iloc[0,-2]

        return audio_new, labels


def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)

def train(model, train_loader, optimizer, criterion):
  model.train()
  train_loss = 0
  train_correct = 0
  for data, label in train_loader:
    data = data.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    out = model(data)
    train_correct += (torch.argmax(out, dim=1).eq_(label).sum()).item()
    loss = criterion(out, label)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
  avg_loss = train_loss/len(train_loader)
  accuracy = train_correct/(len(train_loader.dataset))
  return avg_loss, accuracy

def test(model, test_loader, criterion):
  with torch.no_grad():
    model.eval()
    test_correct = 0
    test_loss = 0
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        out2 = model(data)
        loss2 = criterion(out2, label)
        test_loss += loss2.item()
        test_correct += (torch.argmax(out2, dim=1).eq_(label).sum()).item()
    avg_loss = test_loss/len(test_loader)
    accuracy = test_correct/len(test_loader.dataset)
  return avg_loss, accuracy
