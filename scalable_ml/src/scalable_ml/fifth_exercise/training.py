from tqdm import tqdm
import numpy as np
import tempfile
import torch
import scalable_ml.model_utilities as util
from ray import train, tune
import os

def train_model(ml_model, device, train_loader, val_loader, epochs, learning_rate=1e-3, weight_decay=1e-5, checkpoint_metric="val_accuracy", save=False, checkpoint_dir=None, exercise=1):
    """
    Trains a pytorch model

    For the optimizer we use the Adam algorithm
    - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    - https://arxiv.org/abs/1412.6980

   For the loss function we use the mean squared error (squared L2 norm) or DSSIM (derived from SSIM)
   - https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
   - https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html

    :param ml_model:
    :type ml_model: torch Model object
    :param device: device on which training is performed
    :type device: basestring
    :param train_loader: training dataset
    :type train_loader: torch DataLoader
    :param epochs: number of epochs used for training
    :type epochs: int
    :param val_loader:  training validation set
    :type train_loader: torch DataLoader
    :param learning_rate: learning rate for optimizer such as Adam
    :param weight_decay: L2-penalty for large weight coefficients
    """
    params = [
        {'params': ml_model.parameters()}
    ]

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()

    losses = {'train_loss': [], 'val_loss': []}
    accuracy = {'train_accuracy': [], 'val_accuracy': []}

    # ? Retrieve checkpoint if it exists (Ray Tune 2.0+)
    if checkpoint_metric in ["val_accuracy","train_accuracy"]: start_metric=0
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint_dict = torch.load(checkpoint_path, map_location=device)
                start_epoch = checkpoint_dict["epoch"] + 1
                ml_model.load_state_dict(checkpoint_dict["model_state"])
                optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
                best_val_loss = checkpoint_dict.get("best_val_loss", float("inf"))
                print(f"Resuming from epoch {start_epoch}")
                
    # Save checkpoint
    if checkpoint_dir is None: checkpoint_dir = train.get_context().get_trial_dir()
    else: os.makedirs(checkpoint_dir, exist_ok=True)
        
    # loop for training
    for epoch in range(start_epoch, epochs):
        train_loss, train_accuracy = train_epoch(ml_model, device, train_loader, loss_function, optimizer, exercise)
        val_loss, val_accuracy = val_epoch(ml_model, device, val_loader, loss_function, exercise)

        print(f'Epoch {epoch + 1}/{epochs} \t train loss {train_loss:8f} \t train accuracy {train_accuracy:1f}')
        print(f'Epoch {epoch + 1}/{epochs} \t val loss {val_loss:8f} \t val accuracy {val_accuracy:1f}')

        #Store losses and accuracy
        losses['train_loss'].append(train_loss)
        accuracy['train_accuracy'].append(train_accuracy)
        losses['val_loss'].append(val_loss)
        accuracy['val_accuracy'].append(val_accuracy)

    
        if (checkpoint_metric=="val_accuracy")&(val_accuracy>start_metric): improve=True; start_metric=val_accuracy
        if (checkpoint_metric=="train_accuracy")&(train_accuracy>start_metric): improve=True; start_metric=train_accuracy

        if save&improve:
            #checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pt")
            torch.save({"epoch": epoch,
                        "model_state": ml_model.state_dict(),
                        "optimizer_state": optimizer.state_dict()},
                       checkpoint_path,)
            improve=False

        #Report results with checkpoint
        #checkpoint=train.Checkpoint.from_directory(checkpoint_dir)
        #checkpoint=train.Checkpoint(path=checkpoint_path)

        train.report({'training_accuracy': train_accuracy,
                      'val_accuracy': val_accuracy,
                      'training_loss': train_loss,
                      'val_loss': val_loss})
        
    return losses, accuracy


def train_epoch(ml_model, device, dataloader, loss_fn, optimizer, exercise):
    """
    Performs one training epoch.

    :param ml_model: Model that is being trained
    :type ml_model: list of torch Model object
    :param device: device on which training is performed
    :type device: basestring
    :param dataloader: training dataset
    :type dataloader: torch DataLoader
    :param loss_fn: loss-function
    :type loss_fn: loss-function object
    :param optimizer: model optimizer
    :type optimizer: torch optimizer object
    :param exercise: exercise sheet (1,2,3, ...)
    :type exercise: int
    :return: mean of training loss during epoch
    :rtype: float
    """

    train_loss = []
    train_accuracy = []

    # required since we shift model to eval model in accuracy function
    ml_model.train()  # sets model to train mode, i.e. activates dropout layers not used for inference

    # create progress bar
    #loop = tqdm(dataloader)

    # iterate over batches generated by dataloader
    for ix, (image_batch, target_batch) in enumerate(dataloader):
        # each batch consists of 128 images
        image_batch, target_batch = image_batch.to(device), target_batch.to(device)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Forward pass: make prediction for this batch
        pred = ml_model(image_batch)

        # compute loss
        loss = loss_fn(pred, target_batch)

        # Backward propagation for computing the gradient of loss function
        loss.backward()

        # Update the weights and the biases of our current model
        optimizer.step()

        # update progress bar
        # loop.set_postfix(loss=loss.detach().cpu().numpy())

        train_loss.append(loss.detach().cpu().numpy())

        # accuracy of the batch
        if exercise==1:
            is_correct = util.accuracy(image_batch, target_batch, ml_model)
        elif exercise==2:
            is_correct = util.segmentation_accuracy(image_batch, target_batch, ml_model)
            
        train_accuracy.append(is_correct)

    return np.mean(train_loss), np.mean(train_accuracy)


def val_epoch(ml_model, device, val_loader, loss_fn, exercise):
    ml_model.eval()  # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.

    val_loss = []
    val_accuracy = []

    with torch.no_grad():
        for ix, (val_image_batch, val_target_batch) in enumerate(val_loader):

            val_image_batch, val_target_batch = val_image_batch.to(device), val_target_batch.to(device)

            pred = ml_model(val_image_batch)

            # compute loss
            loss = loss_fn(pred, val_target_batch)
            val_loss.append(loss.detach().cpu().numpy())

            # compute accuracy
            if exercise==1:
                is_correct = util.accuracy(val_image_batch, val_target_batch, ml_model)
            if exercise==2:
                is_correct = util.segmentation_accuracy(val_image_batch, val_target_batch, ml_model)

            val_accuracy.append(is_correct)

    return np.mean(val_loss), np.mean(val_accuracy)
