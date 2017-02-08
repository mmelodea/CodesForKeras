from ROOT import *
from math import *
from keras.models import model_from_json
import datetime

json_file = open('diabetes_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('diabetes_weights.h5')

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adamax',metrics=['accuracy'])

seed = int()
rdm = TRandom3(seed)
a = int(9*rdm.Rndm())
b = int(9*rdm.Rndm())
c = int(9*rdm.Rndm())

print("///sigmoid output function")
print "double sigmoid_%i%i%i(double z){" %(a,b,c)
print "\treturn 1./(1+TMath::Exp(-z));"
print "}\n"

print("///max function (ReLU)")
print "double relu_%i%i%i(double value){" %(a,b,c)
print "\tdouble max_val = (value > 0)? value:0;"
print "\treturn max_val;"
print "}\n\n"

print("///PReLU function")
print "double prelu_%i%i%i(double slope, double value){" %(a,b,c)
print "\tdouble max_val = (value > 0)? value:value*slope;"
print "\treturn max_val;"
print "}\n\n"



nlayers = len(model.layers)
ilayer = nlayers
layer_types = [-1 for i in range(nlayers)]
#creates each neuron-input function
print("///Set of neuron functions with their respective weight and bias")
while( ilayer-1 >=0 ):
  #g=model.layers[ilayer-1].get_config() #contains the weights and the bias
  h=model.layers[ilayer-1].get_weights() #contains the weights and the bias
  #print("Loading layer %i" % (ilayer-1))
  #print (g)
  #print (h)
  #h[0][i][j] --> weights from i-esimo input in the j-esimo neuron, h[1][j] --> bias for each neuron
  
  layer_type = len(h)
  layer_types[ilayer-1] = layer_type
  layer_ninputs = len(h[0])
  layer_nneurons = 1
  if(layer_type == 2):
    layer_nneurons = len(h[0][0])
  ineuron = layer_nneurons-1
  #print("Searching max absolute weight...")
  
  if(layer_type == 2):
    print("double l%i_func_%i%i%i(std::vector<double> &inputs, int neuron){" % (ilayer,a,b,c))
    print "\tint n_inputs = inputs.size();"
  else:
    print("double l%i_func_%i%i%i(int neuron){" % (ilayer,a,b,c))
    
  print "\tdouble z = 0;"
  print("\tswitch(neuron){")
  while( ineuron >=0 ): #look at each neuron (number of func/layer)
    if(layer_type == 2):
      print("\t\tcase %i:" % (layer_nneurons-ineuron-1))
    n_inputs = layer_ninputs-1
    i_input = n_inputs
    while( i_input >=0 ):
      weight = 0
      if(layer_type == 2):
	weight = h[0][n_inputs-i_input][layer_nneurons-ineuron-1]
	print "\t\t\tz += %.10f*inputs[%i];" %(weight,n_inputs-i_input)
      else:
	weight = h[0][n_inputs-i_input]
	print "\t\tcase %i:" % (n_inputs-i_input)
	print "\t\t\tz = %.10f;" % weight
	print "\t\tbreak;"	  
      i_input -= 1
      
    if(layer_type == 2):
      bias = h[1][layer_nneurons-ineuron-1]
      print("\t\t\tz += %.10f;" % bias)      
      print "\t\tbreak;"
    ineuron -= 1
    
  print "\t}"
  print "\n\treturn z;"
  print "}"
  ilayer -= 1


#raw_input("Close?")




#create the network archictecture
print "\n\n"
for ilayer in range(len(model.layers)):
  #g=layer.get_config()  #contains the network configuration
  h=model.layers[ilayer].get_weights() #contains the weights and the bias
  #print("Loading layer %i" % (ilayer))
  #print (g)
  #print (h)
  #h[0][i][j] --> weights from i-esimo input in the j-esimo neuron, h[1][j] --> bias for each neuron
  
  layer_type = len(h)
  layer_ninputs = len(h[0])
  layer_nneurons = 1
  if(layer_type == 2):
    layer_nneurons = len(h[0][0])
  
  #the first hidden layer
  if ilayer == 0:
    print "double dnn_%i%i%i(std::vector<double> &inputs){" % (a,b,c)
    print "\tstd::vector<double> l%i_neurons_z_%i%i%i;" % (ilayer+1,a,b,c)
    print "\tfor(int l%i_n=0; l%i_n<%i; l%i_n++)" % (ilayer+1,ilayer+1,layer_nneurons,ilayer+1)
    if(layer_types[ilayer+1] == layer_type):
      print "\t\tl%i_neurons_z_%i%i%i.push_back( relu_%i%i%i(l%i_func_%i%i%i(inputs,l%i_n)) );" % (ilayer+1,a,b,c,a,b,c,ilayer+1,a,b,c,ilayer+1)
    else:
      print "\t\tl%i_neurons_z_%i%i%i.push_back( l%i_func_%i%i%i(inputs,l%i_n) );" % (ilayer+1,a,b,c,ilayer+1,a,b,c,ilayer+1)
    print "//------ end layer %i ------\n\n" % (ilayer+1)
    #raw_input("Close?")
  
  
  #here's the complication (where out layer neurons appear)
  if ilayer > 0 and ilayer < nlayers-1:
    print "\tstd::vector<double> l%i_neurons_z_%i%i%i;" % (ilayer+1,a,b,c)
    if(layer_types[ilayer+1] == layer_type):	
      print "\tfor(int l%i_n=0; l%i_n<%i; l%i_n++)" % (ilayer+1,ilayer+1,layer_nneurons,ilayer+1)
      print "\t\tl%i_neurons_z_%i%i%i.push_back( relu_%i%i%i(l%i_func_%i%i%i(l%i_neurons_z_%i%i%i,l%i_n)) );" % (ilayer+1,a,b,c,a,b,c,ilayer+1,a,b,c,ilayer,a,b,c,ilayer+1)
    elif(layer_type == 1 and layer_types[ilayer+1] != layer_type):
      print "\tfor(int l%i_n=0; l%i_n<%i; l%i_n++)" % (ilayer+1,ilayer+1,layer_ninputs,ilayer+1)
      print "\t\tl%i_neurons_z_%i%i%i.push_back( prelu_%i%i%i(l%i_func_%i%i%i(l%i_n),l%i_neurons_z_%i%i%i[l%i_n]) );" % (ilayer+1,a,b,c,a,b,c,ilayer+1,a,b,c,ilayer+1,ilayer,a,b,c,ilayer+1)
    else:
      print "\tfor(int l%i_n=0; l%i_n<%i; l%i_n++)" % (ilayer+1,ilayer+1,layer_ninputs,ilayer+1)
      print "\t\tl%i_neurons_z_%i%i%i.push_back( l%i_func_%i%i%i(l%i_neurons_z_%i%i%i,l%i_n) );" % (ilayer+1,a,b,c,ilayer+1,a,b,c,ilayer,a,b,c,ilayer+1)
    print "//------ end layer %i ------\n\n" % (ilayer+1)
    #raw_input("Close?")
  
  
  #last layer (output)
  if ilayer == nlayers-1:
    print "\n\treturn sigmoid_%i%i%i( l%i_func_%i%i%i(l%i_neurons_z_%i%i%i,0) );" % (a,b,c,ilayer+1,a,b,c,ilayer,a,b,c)
    print "}"
    
#raw_input("Close?")
