Dependencies:
    Please download the following repositories -
    numpy=1.19.5+
    pandas=1.3.2+
    tensorflow=2.6.0+
   
The Jupyter Notebooks:
    In each folder, there will be a .ipynb file labelled Model#.ipynb, which were used to train the following models.
    Model1 - Sequential, Three Dense Layers.
    Model2 - Sequential, Conv2D => MaxPool => Flatten => Two Dense Layers.
    Model3 - NonSequential, (Conv2D, Conv2D, Conv2D, MaxPool) => Concatenate => Flatten => Two Dense Layers.
    
    Each note book should have two cells:
    Cell 1 - Training the Model
    Cell 2 - Testing the Model
    
    Running the Training Cell will take the parameters initialized at the top, train the model, and log the history.
    Running the Testing Cell will take the appropriate "FINAL" model and test the accuracy of x_test and y_test.
   
The Logs/History:
    Each folder will have a numerous amount of logs, each describing a different attempt at running the model.
    These logs contain the parameters, the training history with loss and accuracy at each epoch, and the final scores of the training and test sets.
    The log labeled "FINAL_HISTORY_MODEL#.txt" contains the data for the log with the best testing score.

The Models:
    Similarly to The Logs, each training attempt resulted in its own model.
    The model labeled "FINAL_MODEL#.h5" is the model with the best testing score, and is used in the Testing Cell.