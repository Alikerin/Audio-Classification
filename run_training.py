# @Author: Ibrahim Salihu Yusuf <Ibrahim>
# @Date:   2019-12-10T12:28:39+02:00
# @Email:  sibrahim1396@gmail.com
# @Project: Audio Classifier
# @Last modified by:   Ibrahim
# @Last modified time: 2019-12-12T13:59:25+02:00



from models import *
from utils import *
import json, time


import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

file_path = '/data/UrbanSound8K/metadata/UrbanSound8K.csv'
audio_paths = '/data/UrbanSound8K/audio'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


avg_loss = []
overall_result = {}

M3 = M3().to(device).apply(init_weights)
M5 = M5().to(device).apply(init_weights)
M11 = M11().to(device).apply(init_weights)
M18 = M18().to(device).apply(init_weights)
M34 = M34_res().to(device).apply(init_weights)

models = [M3, M5, M11, M18, M34]
models_name = ["M3", "M5", "M11", "M18", "M34"]

def main():
    for model_idx, model in enumerate(models):
      writer_path = "/results/{}".format(models_name[model_idx])
      # comment = "{}".format(models_name[model_idx])
      writer = SummaryWriter(writer_path)
      inter_result = []
      for i in range(10):
        print("-------------- {} Fold {} -----------------".format(models_name[model_idx], i))
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_folds = folds.pop(i)
        train_folds = folds

        train_dataset = AudioDataset(file_path, audio_paths, train_folds)
        test_dataset = AudioDataset(file_path, audio_paths, test_folds)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.023)

        epochs = 100

        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []

        for i in range(epochs):
          t0 = time.time()
          train_loss, train_acc = train(model, train_loader, optimizer, criterion)
          t1 = time.time()
          print("Train Loss at {}/{} is {} | Train accuracy:{} | Time:{}".format(i+1, epochs, train_loss, train_acc, t1-t0))

          t0 = time.time()
          test_loss, test_acc = test(model, test_loader, criterion)
          t1 = time.time()
          print("Test Loss at {}/{} is {} | Test accuracy:{} | Time:{}".format(i+1, epochs, train_loss, train_acc, t1-t0))

          train_losses.append(train_loss)
          train_accuracy.append(train_acc)
          test_losses.append(test_loss)
          test_accuracy.append(test_acc)

          scheduler.step()

          writer.add_scalar("Fold{}/Loss/Test".format(i), test_loss, i)
          writer.add_scalar("Fold{}/Accuracy/Test".format(i), test_acc, i)
          writer.add_scalar("Fold{}/Loss/Train".format(i), train_loss, i)
          writer.add_scalar("Fold{}/Accuracy/Train".format(i), train_acc, i)

        inter_result.append({'train_loss':train_losses, 'train_accuracy':train_accuracy, 'test_loss':test_losses, 'test_accuracy':test_accuracy})
        #writer
      overall_result["{}".format(models_name[model_idx])] = inter_result
      out_filename = '/results/{}_result.json'.format(models_name[model_idx])
      writer.close()
      try:
            with open(out_filename, 'w') as f:
                json.dump(inter_result, f)
      except:
        pass
if __name__ == "__main__":
    main()
    print("\n\nEnd of training\nLogging results..")
    with open('/data/overall_results.json', 'w') as f:
        json.dump(overall_result, f)
