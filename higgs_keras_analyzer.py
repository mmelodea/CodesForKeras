#For analyze data
import numpy, math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json

from ROOT import *

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load datasets
testing = numpy.loadtxt("Higgs13TeV_2or_more_jets_vbf_sig_bkgs_118-130GeV_classified.csv", delimiter=",")
# split into input (X) and output (Y) variables
X_test = testing[:,0:2]
Y_test = testing[:,3]
weight = testing[:,4]

json_file = open('higgs_keras_model_118-130.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_118-130_weights.h5')

#compile the model
loaded_model.compile(loss='binary_crossentropy', optimizer='adamax',metrics=['accuracy'])
#scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

mpredictions = loaded_model.predict(X_test)
#total = len(predictions)


def dnn3(dnn_ggh, dnn_non_ggh):
  g = 0.818789/(0.818789 + 0.033430 + 0.027271 + 0.029377 + 0.099949 + 0.008776);
  return 1./(1 + g*(1-dnn_ggh)/dnn_ggh + (1-g)*(1-dnn_non_ggh)/dnn_non_ggh);

total = len(Y_test)
cpredictions = [0 for i in range(total)]
for i in range(total):
  cpredictions[i] = dnn3(X_test[i][0],X_test[i][1])


msig_hist = TH1D("msig_hist","predictions",100,-0.05,1.05)
msig_hist.SetLineColor(kBlue)
mbkg_hist = TH1D("mbkg_hist","predictions",100,-0.05,1.05)
mbkg_hist.SetLineColor(kRed)
csig_hist = TH1D("csig_hist","predictions",100,-0.05,1.05)
csig_hist.SetLineColor(kCyan)
cbkg_hist = TH1D("cbkg_hist","predictions",100,-0.05,1.05)
cbkg_hist.SetLineColor(kViolet)

for i in range(total):
  if(Y_test[i] == 1):
    msig_hist.Fill( mpredictions[i], weight[i] )
    csig_hist.Fill( cpredictions[i], weight[i] )
  else:
    mbkg_hist.Fill( mpredictions[i], weight[i] )
    cbkg_hist.Fill( cpredictions[i], weight[i] )
  

msig_eff = TGraph()
mbkg_eff = TGraph()
csig_eff = TGraph()
cbkg_eff = TGraph()
mg = TGraph()
cg = TGraph()
ipoint = 0
ipoint2 = 0
integral = 0
integral2 = 0
fpr_1 = 0
fpr_1_2 = 0
for icut in range(100):
  if(icut % 10 == 0):
    print("Testing cut: %i" % icut)
  cut = icut/100.
  ntp = 0
  ntn = 0
  nfp = 0
  nfn = 0
  ntp2 = 0
  ntn2 = 0
  nfp2 = 0
  nfn2 = 0
  for i in range(total):
    if( mpredictions[i] > cut and Y_test[i] == 1):
      ntp = ntp + weight[i]
    if( mpredictions[i] < cut and Y_test[i] == 0): 
      ntn = ntn + weight[i]
    if( mpredictions[i] < cut and Y_test[i] == 1): 
      nfn = nfn + weight[i]
    if( mpredictions[i] > cut and Y_test[i] == 0): 
      nfp = nfp + weight[i]

    if( cpredictions[i] > cut and Y_test[i] == 1):
      ntp2 = ntp2 + weight[i]
    if( cpredictions[i] < cut and Y_test[i] == 0): 
      ntn2 = ntn2 + weight[i]
    if( cpredictions[i] < cut and Y_test[i] == 1): 
      nfn2 = nfn2 + weight[i]
    if( cpredictions[i] > cut and Y_test[i] == 0): 
      nfp2 = nfp2 + weight[i]
  
  if((nfn+ntp) != 0 and (ntn+nfp) != 0):
    tpr = ntp/float(ntp+nfn)
    fpr = nfp/float(nfp+ntn)
    mg.SetPoint(ipoint, fpr, tpr)
    msig_eff.SetPoint(ipoint,cut,tpr)
    mbkg_eff.SetPoint(ipoint,cut,fpr)
    if(ipoint > 0):
      integral += math.fabs(fpr_1-fpr)*tpr
    fpr_1 = fpr
    ipoint = ipoint + 1
      
  if((nfn2+ntp2) != 0 and (ntn2+nfp2) != 0):
    tpr = ntp2/float(ntp2+nfn2)
    fpr = nfp2/float(nfp2+ntn2)
    cg.SetPoint(ipoint2, fpr, tpr)
    csig_eff.SetPoint(ipoint2,cut,tpr)
    cbkg_eff.SetPoint(ipoint2,cut,fpr)
    if(ipoint2 > 0):
      integral2 += math.fabs(fpr_1_2-fpr)*tpr
    fpr_1_2 = fpr
    ipoint2 = ipoint2 + 1
  
  
leg = TLegend(0.3,0.7,0.6,0.9)  
leg.AddEntry(msig_eff,"DNN_{3}","l")
leg.AddEntry(mbkg_eff,"DNN_{3}","l")
leg.AddEntry(csig_eff,"Combined","l")
leg.AddEntry(cbkg_eff,"Combined","l")

  
cv = TCanvas("cv","",10,10,800,600)
msig_hist.Draw("hist")
mbkg_hist.Draw("hist,same")
csig_hist.Draw("hist,same")
cbkg_hist.Draw("hist,same")
leg.Draw()

cv2 = TCanvas("cv2","",40,10,800,600)
tgm = TMultiGraph()
mg.SetLineColor(kBlue)
tgm.Add( mg )
cg.SetLineColor(kRed)
tgm.Add( cg )
tgm.Draw("AL")
tgm.SetTitle("ROC area (DNN3/combined): {0}/{1}".format(round(integral,3),round(integral2,3)))
tgm.GetXaxis().SetRangeUser(0,1)
tgm.GetXaxis().SetTitle("Background efficiency")
tgm.GetYaxis().SetRangeUser(0,1)
tgm.GetYaxis().SetTitle("Signal efficiency")
leg2 = TLegend(0.3,0.7,0.6,0.9)  
leg2.AddEntry(mg,"DNN_{3}","l")
leg2.AddEntry(cg,"Combined","l")
leg2.Draw()

cv3 = TCanvas("cv3","",80,10,800,600)
gm = TMultiGraph()
msig_eff.SetLineColor(kBlue)
gm.Add( msig_eff )
mbkg_eff.SetLineColor(kRed)
gm.Add( mbkg_eff )
csig_eff.SetLineColor(kCyan)
gm.Add( csig_eff )
cbkg_eff.SetLineColor(kViolet)
gm.Add( cbkg_eff )
gm.Draw("AL")
leg.Draw()


raw_input("Press enter to exit...")
