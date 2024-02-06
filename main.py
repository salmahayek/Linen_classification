from experiment import experiment

# Set hyperparameters
lr = 0.00001
batch_size = 32

# Choose the model name from ["mobilenetv2", "shufflenetv2", "shufflenet", "squeezenet"]
model_name = "squeezenet"

# Set the number of epochs for training
epochs = 50

# Specify the directory containing the dataset
data_dir = "/content/Linen_Dataset"

# Run the experiment
experiment(lr, batch_size, model_name, epochs, data_dir)