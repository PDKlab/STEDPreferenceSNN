# Automated multi-objective preference articulation

This repo contains the neural network architecture for learning a preference articulation function between several objectives. This is the SNN architecture presented used in the paper [Toward intelligent nanoscopy: A machine learning approach for automated optimization of super-resolution optical microscopy](https://tdb). It contains the code for training a model from scratch, and deplyoing a model on a server using the Docker file.

### Dependencies
- Install [pytorch](http://pytorch.org/)
- Install these python packages:
```shell
pip install numpy scikit-learn scikit-image request
```

### Training a network
Use the script ``python train.py``:
```shell
usage: train.py [-h] [-m MARGIN] [-bs BATCH_SIZE] [-ep NB_EPOCHS]
                [-rs RANDOM_STATE] [--cuda]
                data_path results_path

train a network model

positional arguments:
  data_path             path to the data to use
  results_path          where to save the results

optional arguments:
  -h, --help            show this help message and exit
  -m MARGIN, --margin MARGIN
                        margin of the loss
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        SGD batch size
  -ep NB_EPOCHS, --nb-epochs NB_EPOCHS
                        SGD number of epochs
  -rs RANDOM_STATE, --random-state RANDOM_STATE
                        random state of train/valid/test split
  --cuda                use GPU or not
```

# Installation Instructions for server use

### Installing docker

- Getting the Docker installation file from [https://www.docker.com/community-edition#/download](https://www.docker.com/community-edition#/download) depending of your operating system. 
- Starting the installation and following instruction. It should be straightforward, though docker can take few minutes to start.

### Building the image

- Clone the repository: `git clone https://github.com/PDKlab/STEDPreferenceSNN`
- Enter the repository: `cd STEDPreferenceSNN`
- Build the Docker image: `docker build -t pnet .`

### Starting the server

You have two choices to start the server:

1. Start with pre-installed trained models in the image:

   `docker run --rm -p 5000:5000 pnet /bin/bash -c "cd /workspace/executable/ && python server.py trained_models/<experiment>"`

   - `<experiment>` is one of the pre-installed experiment folder

2. Start with your experiment, create a Docker volume linking to your experiment folder: 

   `docker run --rm -v "<my-experiment-folder>:/mnt/experiment" -p 5000:5000 pnet /bin/bash -c "cd /workspace/executable/ && python server.py /mnt/experiment"`

### Using the virtual API

Once a server is started, you can talk to it with the virtual `VirtualNet` API .

- Copy the file `src/models/virtual.py` file in your code repository.
- Import the VirtualNet in your code with the line `from virtual import VirtualNet`.
- Instanciate a `VirtualNet` with an ip address and you can use its `.predict(set)` method.
- `set` should be a set of objective to evaluate.

### Question?

Feel free to send me an email @ louis-emile.robitaille.1@ulaval.ca
