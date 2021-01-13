# EEGCN
Classification of EEG signals into silence vs listening with Graph Neural Network (PyTorch)

utils: training (train) and evaluating (test) routines
model: contain classifiers to solve the problem (torch.nn.Module classes)
main: script to run, it loads dataset and performs training of a selected model on it.

* Run the training script:
``` 
python main.py --hidden_channels 200 --num_features 50 --epochs 150 --exp_name exp --data EEG_data/train_data.pt --lr 1e-4 --activation relu --pooling avg
```

