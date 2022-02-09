# lstm-rnn-nfkb-trajectories

Long-Short Term Memory Network: multiclassification model of ligand identity across 5 polarization states given time series NFkB trajectories
Parameters: 2 LSTM layers, 1 Linear layer, N (num_layers) = 2, L (sequence length/dimension) = 1, H (hidden_size) = 98, input_size = 1, output_size = 9

Models:
lstm1.pth - trained on replicated dataset
lstm2.pth - trained on unreplicated dataset
lstm3.pth - sandbox model