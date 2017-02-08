from ROOT import *
from math import *
from keras.models import model_from_json

network_orientation = "horizontal"
#network_orientation = "vertical"

def screen_opt( n_nodes, max_nodes ):
  return n_nodes/float(max_nodes) + max_nodes/2.
  
#function to map neurons positions
def neuron_pos(a, n_inputs, i_input):
  return a*( 1 - (2*i_input)/(n_inputs-1.) )

#function to control line color based on the normalized weight from each synapse
def w_to_c(w):
  colors = [-10,-8,-5,-1,+4]
  return colors[int(4*w)]

#function to control line width based on the normalized weight from each synapse
def w_to_l(w):
  return int(4*w) + 1


json_file = open('diabetes_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('diabetes_weights.h5')

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adamax',metrics=['accuracy'])

#1 for neuron into layer, 2 for neuron outside layer
#0nly for the hidden layers
nlayers = len(model.layers)
layer_type = [-1 for i in range(nlayers)]  
ipoint = 0
max_nodes = 0
for ilayer in range(nlayers):
  h = model.layers[ilayer].get_weights() #contains the weights and the bias
  #print h
  layer_ninputs = len(h[0])
  layer_type[ilayer] = 1
  if(len(h) == 1):
    layer_type[ilayer] = 2
  nodes = 0
  for i_input in range(layer_ninputs):
    nodes += 1
    
  #finds maximum number of nodes
  if(nodes > max_nodes):
    max_nodes = nodes

#print("Max nodes: %i" % max_nodes)
#print the nodes
#gROOT.SetBatch()
cv = TCanvas("cv","",0,0,1400,800)
network  = TGraph()
total_inputs = 0
for ilayer in range(nlayers):
  h = model.layers[ilayer].get_weights() #contains the weights and the bias
  #h[0][i][j] --> weights from i-esimo input in the j-esimo neuron, h[1][j] --> bias for each neuron
  layer_ninputs = len(h[0])
  #print "Layer: %f" % (ilayer+ilayer*(layer_ninputs/float(max_nodes)))
  for i_input in range(layer_ninputs):
    pinput = neuron_pos(screen_opt(layer_ninputs, max_nodes),layer_ninputs,i_input)
    if(network_orientation == "horizontal"):
      network.SetPoint(ipoint,ilayer,pinput)
    if(network_orientation == "vertical"):
      network.SetPoint(ipoint,pinput,ilayer)
    ipoint += 1
    if(ilayer == 0):
      total_inputs += 1
    
#sets the last neuron(output)
if(network_orientation == "horizontal"):
  network.SetPoint(ipoint,nlayers,0)
if(network_orientation == "vertical"):
  network.SetPoint(ipoint,0,nlayers)

#sets the visual aspect
network.SetMarkerStyle(20)
network.SetMarkerSize(3)
network.GetXaxis().CenterTitle()
network.GetYaxis().SetLabelColor(0)
network.GetYaxis().SetAxisColor(0)
network.GetXaxis().SetLabelColor(0)
network.GetXaxis().SetAxisColor(0)
network.SetTitle("network architecture after training")
network.Draw("AP")

input_layer_id = TPaveText(0.05,0.02,0.15,0.08,"NDC")
input_layer_id.AddText("Input Layer")
input_layer_id.SetFillStyle(0)
input_layer_id.SetBorderSize(0)
input_layer_id.Draw()

hidden_layer_id = TPaveText(0.46,0.02,0.6,0.08,"NDC")
hidden_layer_id.AddText("Hidden Layers")
hidden_layer_id.SetFillStyle(0)
hidden_layer_id.SetBorderSize(0)
hidden_layer_id.Draw()

output_layer_id = TPaveText(0.82,0.02,0.98,0.08,"NDC")
output_layer_id.AddText("Output Layer")
output_layer_id.SetFillStyle(0)
output_layer_id.SetBorderSize(0)
output_layer_id.Draw()
#raw_input("Close?")


#create the network archictecture
il = 0
l = [TLine() for i in range(int(100*ipoint))]
#print("layers in the model: %i" % nlayers)
for ilayer in range(nlayers):
  if(ilayer < nlayers-1):
    #print("Loading layer %i" % (ilayer))  
    h = model.layers[ilayer].get_weights() #contains the weights and the bias
    layer_ninputs = len(h[0])
    layer_nneurons = layer_ninputs
    if(layer_type[ilayer] != 2):
      layer_nneurons = len(h[0][0])
    #print "layer_nneurons: %i" % layer_nneurons

    #loop over neurons searching the max weight
    for ineuron in range(layer_nneurons):
      max_weight = 0
      for i_input in range(layer_ninputs):
	weight = 0
	#print layer_type[ilayer]
	if(layer_type[ilayer] == 1):
	  weight = fabs(h[0][i_input][ineuron])
	else:
	  weight = fabs(h[0][i_input])
	  
	if weight > max_weight:
	  max_weight = weight
      #maps the current layer nodes position
      pinput_f = neuron_pos(screen_opt(layer_nneurons, max_nodes),layer_nneurons,ineuron)
      
      #maps the previous layer nodes position
      if(max_weight == 0):
	max_weight = 1.
      for i_input in range(layer_ninputs):
	norm_weight = 0
	if(layer_type[ilayer] == 1):
	  norm_weight = fabs(h[0][i_input][ineuron])/max_weight
	else:
	  norm_weight = fabs(h[0][i_input])/max_weight
	#lines yi
	pinput_i = neuron_pos(screen_opt(layer_ninputs, max_nodes),layer_ninputs,i_input)
	if(layer_type[ilayer] == 2 and pinput_i != pinput_f):
	  continue
	
	#creates the line connections (synapses)
	if(network_orientation == "horizontal"):
	  l[il] = TLine(ilayer,pinput_i,ilayer+1,pinput_f)
	if(network_orientation == "vertical"):
	  l[il] = TLine(pinput_i,ilayer,pinput_f,ilayer+1)
	#l[il].SetLineColor( kBlue + w_to_c(norm_weight) )
	l[il].SetLineWidth( w_to_l(norm_weight) )
	l[il].Draw()
	#gPad.Update()
	il += 1
	#print("line/xi/yi/xf/yf/width: {0}/{1}/{2}/{3}/{4}/{5}".format(il,ilayer,pinput_i,ilayer+1,pinput_f,int(10*norm_weight)))

	  
  #output layer is printed separated
  else:    
    #print "Loading output layer"  
    h = model.layers[ilayer].get_weights() #contains the weights and the bias
    layer_ninputs = len(h[0])
    layer_nneurons = len(h[0][0])
  
    for ineuron in range(layer_nneurons):
      max_weight = 0
      for i_input in range(layer_ninputs):
	#print("\tInput %i" % i_input)
	weight = fabs(h[0][i_input][ineuron])
	if weight > max_weight:
	  max_weight = weight
      for i_input in range(layer_ninputs):
	norm_weight = fabs(h[0][i_input][ineuron])/max_weight
	pinput_i = neuron_pos(screen_opt(layer_ninputs, max_nodes),layer_ninputs,i_input)
	pinput_f = 0
	if(network_orientation == "horizontal"):
	  l[il] = TLine(ilayer,pinput_i,ilayer+1,pinput_f)
	if(network_orientation == "vertical"):
	  l[il] = TLine(pinput_i,ilayer,pinput_f,ilayer+1)
	#l[il].SetLineColor( kBlue + w_to_c(norm_weight) )
	l[il].SetLineWidth( w_to_l(norm_weight) )
	l[il].Draw()
	il += 1
	#gPad.Update()
	#print("line/xi/yi/xf/yf/width: {0}/{1}/{2}/{3}/{4}/{5}".format(il,ilayer,pinput_i,ilayer+1,pinput_f,int(10*norm_weight)))
  

gPad.Update()
cv.Print("network_architecture_" + network_orientation + ".png")
cv.Draw()
#cv.Close()

raw_input("Close?")
