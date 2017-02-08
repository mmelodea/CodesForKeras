#Type of stacking layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, SReLU
#The array python manager
import numpy
from ROOT import TFile, TTree

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Using NumPy function to load file
#(eight input variables and one output value - 0=no diabets, 1=diabets)
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#Building the models
#The first layer must have the right number of inputs (8 in this case)
#The "Dense" class stands for fully connected layers. We specify the number of neurons in the first argument
#the initialization method in the second (init) and the activation function in the third (activation)
# create model
model = Sequential()
#12 neurons
model.add(Dense(1000, init='uniform', input_dim=8))
model.add( PReLU() )
model.add(Dense(25, init='uniform'))
model.add( PReLU() )
model.add(Dense(5, init='uniform'))
model.add( PReLU() )
#We use sigmod here to ensure the network output is between 0 and 1
#1 neuron, predict the class: diabets or not
model.add(Dense(1, init='uniform', activation='sigmoid'))

#Compiling the model uses the efficient numerical libraries in the backend (Theano/TensorFlow) and chooses
#the best way to represent the network for training and making predictions to run on the current hardware
#When Compiling is need to set some properties (training a network means finding the best set of weights)
#We must specify a loss function, an optimizer and any optional metrics to be collected and reported
#In this case, we use the gradient descent algorithm "adam" as optimizer
#And, we collect the classification accuracy as the metrics
#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#training the model
#this is where the work happens on the CPU/GPU
model.fit(X, Y, nb_epoch=100, batch_size=10, verbose=2)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#Now, if one wants to use the trained model to predict the classification for a new dataset, just needs to
#call the predict member. But, remember: in the current example we have a sigmoid function in the output
#what gives us values between 0 and 1. To get a binary classification it's need to round the values.
# calculate predictions
predictions = model.predict(X)
# round predictions
#rounded = [round(x) for x in predictions]
#print(rounded)

#saving model and weights
model_json = model.to_json()
smodel = "diabetes_model.json"
sweight = "diabetes_weights.h5"
with open(smodel,"w") as json_file:
  json_file.write(model_json)
model.save_weights(sweight)


#saving root file with events and disc
f2 = TFile( "diabetes_classified.root", 'recreate' )
tree2 = TTree("diabetes", "DNN_Discriminant")
v1 = numpy.zeros(1, dtype=float)
v2 = numpy.zeros(1, dtype=float)
v3 = numpy.zeros(1, dtype=float)
v4 = numpy.zeros(1, dtype=float)
v5 = numpy.zeros(1, dtype=float)
v6 = numpy.zeros(1, dtype=float)
v7 = numpy.zeros(1, dtype=float)
v8 = numpy.zeros(1, dtype=float)
d  = numpy.zeros(1, dtype=float)
tree2.Branch( 'v1', v1, 'v1/D' )
tree2.Branch( 'v2', v2, 'v2/D' )
tree2.Branch( 'v3', v3, 'v3/D' )
tree2.Branch( 'v4', v4, 'v4/D' )
tree2.Branch( 'v5', v5, 'v5/D' )
tree2.Branch( 'v6', v6, 'v6/D' )
tree2.Branch( 'v7', v7, 'v7/D' )
tree2.Branch( 'v8', v8, 'v8/D' )
tree2.Branch( 'disc', d, 'disc/D' )

for i in range(len(Y)):
  d[0] = predictions[i]
  v1[0] = X[i][0]
  v2[0] = X[i][1]
  v3[0] = X[i][2]
  v4[0] = X[i][3]
  v5[0] = X[i][4]
  v6[0] = X[i][5]
  v7[0] = X[i][6]
  v8[0] = X[i][7]  
  
  tree2.Fill()
  
f2.Write()
