from ROOT import TFile, TTree, TMath
from array import array
 
 
#For test model
import numpy, math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#loads the model for vbf vs all bkgs together
smodel = "DNN_optimized_results/vbf_all_bkgs/higgs_keras_model_118-130.json"
sweight = "DNN_optimized_results/vbf_all_bkgs/model_118-130_weights.h5"
json_file = open(smodel,'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(sweight)

#loads the model for vbf vs ggh
smodel2 = "DNN_optimized_results/vbf_ggh/higgs_keras_model_118-130.json"
sweight2 = "DNN_optimized_results/vbf_ggh/model_118-130_weights.h5"
json_file2 = open(smodel2,'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
loaded_model2 = model_from_json(loaded_model_json2)
loaded_model2.load_weights(sweight2)

#loads the model for vbf vs bkgs-ggh
smodel3 = "DNN_optimized_results/vbf_remaining_bkgs/higgs_keras_model_118-130_remaining_bkgs.json"
sweight3 = "DNN_optimized_results/vbf_remaining_bkgs/model_118-130_weights_remaining_bkgs.h5"
json_file3 = open(smodel3,'r')
loaded_model_json3 = json_file3.read()
json_file3.close()
loaded_model3 = model_from_json(loaded_model_json3)
loaded_model3.load_weights(sweight3)


#loads the model for DNN outputs
smodel4 = "DNN_optimized_results/DNN_decide_cut/higgs_keras_model_118-130.json"
sweight4 = "DNN_optimized_results/DNN_decide_cut/model_118-130_weights.h5"
json_file4 = open(smodel4,'r')
loaded_model_json4 = json_file4.read()
json_file4.close()
loaded_model4 = model_from_json(loaded_model_json4)
loaded_model4.load_weights(sweight4)


# Optimizers
#opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False); #sthocastic gradient descent
#opt = Adam() #adaptive moment estimation (best known)
#opt = RMSprop();
#opt = Adagrad();
#opt = Adadelta();
#opt = Nadam();
opt = Adamax();

#compile the model for vbf vs all bkgs together
loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

#compile the model for vbf vs ggh
loaded_model2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

#compile the model for vbf vs bkgs-ggh
loaded_model3.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

#compile the model for vbf vs bkgs-ggh
loaded_model4.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])


paths = ["histos2e2mu_25ns",
	 "histos4e_25ns",
	 "histos4mu_25ns"
	 ]

files = [    "VBF_HToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8",
	     "GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8",
	     "WminusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8",
	     "WplusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8",
	     "ZH_HToZZ_4LFilter_M125_13TeV_powheg2-minlo-HZJ_JHUgenV6_pythia8",
	     "ttH_HToZZ_4LFilter_M125_13TeV_powheg2_JHUgenV6_pythia8",
	     "ZZTo4L_13TeV_powheg_pythia8",
	     "GluGluToZZTo2e2mu_BackgroundOnly_13TeV_MCFM",
	     "GluGluToZZTo2e2tau_BackgroundOnly_13TeV_MCFM",
	     "GluGluToZZTo2mu2tau_BackgroundOnly_13TeV_MCFM",
	     "GluGluToZZTo4e_BackgroundOnly_13TeV_MCFM",
	     "GluGluToZZTo4mu_BackgroundOnly_13TeV_MCFM",
	     "GluGluToZZTo4tau_BackgroundOnly_13TeV_MCFM",
	     "DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8",
	     "QCD_Pt_1000to1400_TuneCUETP8M1_13TeV_pythia8",
	     "QCD_Pt_10to15_TuneCUETP8M1_13TeV_pythia8",                                              
	     "QCD_Pt_120to170_TuneCUETP8M1_13TeV_pythia8",
	     "QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8",                                               
	     "QCD_Pt_15to30_TuneCUETP8M1_13TeV_pythia8",                                                   
	     "QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8",                                                 
	     "QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8",                                               
	     "QCD_Pt_2400to3200_TuneCUETP8M1_13TeV_pythia8",                                               
	     "QCD_Pt_300to470_TuneCUETP8M1_13TeV_pythia8",                                                 
	     "QCD_Pt_30to50_TuneCUETP8M1_13TeV_pythia8",                                                   
	     "QCD_Pt_3200toInf_TuneCUETP8M1_13TeV_pythia8",                                                
	     "QCD_Pt_470to600_TuneCUETP8M1_13TeV_pythia8",                                                 
	     "QCD_Pt_50to80_TuneCUETP8M1_13TeV_pythia8",                                                   
	     "QCD_Pt_5to10_TuneCUETP8M1_13TeV_pythia8",                                                    
	     "QCD_Pt_600to800_TuneCUETP8M1_13TeV_pythia8",                                                 
	     "QCD_Pt_800to1000_TuneCUETP8M1_13TeV_pythia8",
	     "QCD_Pt_80to120_TuneCUETP8M1_13TeV_pythia8",
	     "ST_t-channel_top_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1",
	     "ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1",
	     "TTTo2L2Nu_13TeV-powheg",
	     "WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8",
	     "WWTo2L2Nu_13TeV-powheg",
	     "WZ_TuneCUETP8M1_13TeV-pythia8"
]

for ifile in files:
  for ipath in paths:
    file_name = "/home/mmelodea/Documents/HiggsRunII/FixedHiggs13TeV/"+ipath+"/output_"+ifile
    file_name_root = file_name + ".root"
    if(file_name_root == "/home/mmelodea/Documents/HiggsRunII/FixedHiggs13TeV/histos4mu_25ns/output_ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1.root"):
      continue
    print("Classifying file %s" % file_name_root)
    f = TFile( file_name_root )
    tree = f.Get('HZZ4LeptonsAnalysisReduced')
    
    f2 = TFile( file_name + "_classified" + ".root", 'recreate' )
    tree2 = TTree("HZZ4LeptonsAnalysisReduced", "DNN_Discriminant")
    tree2.SetDirectory( 0 )
    n = numpy.zeros(1, dtype=float)
    n2 = numpy.zeros(1, dtype=float)
    n3 = numpy.zeros(1, dtype=float)
    n4 = numpy.zeros(1, dtype=float)
    tree2.Branch( 'dnn_vbf_bkgs', n, 'dnn_vbf_bkgs/D' )
    tree2.Branch( 'dnn_vbf_ggh', n2, 'dnn_vbf_ggh/D' )
    tree2.Branch( 'dnn_vbf_bkgs_no_ggh', n3, 'dnn_vbf_bkgs_no_ggh/D' )
    tree2.Branch( 'dnn_vbf_final', n4, 'dnn_vbf_final/D' )
    
 
    X_test = [[0 for i in range(21)] for j in range(tree.GetEntries())]

    ientry = 0
    for i in tree:
      #if(ientry > 10):
	#break
      #print("Mass4l: %f" % i.f_mass4l)
      X_test[ientry][0] = i.f_lept1_pt
      X_test[ientry][1] = i.f_lept1_eta
      X_test[ientry][2] = i.f_lept1_phi
      X_test[ientry][3] = i.f_lept2_pt
      X_test[ientry][4] = i.f_lept2_eta
      X_test[ientry][5] = i.f_lept2_phi
      X_test[ientry][6] = i.f_lept3_pt
      X_test[ientry][7] = i.f_lept3_eta
      X_test[ientry][8] = i.f_lept3_phi
      X_test[ientry][9] = i.f_lept4_pt
      X_test[ientry][10] = i.f_lept4_eta
      X_test[ientry][11] = i.f_lept4_phi
      X_test[ientry][12] = i.f_jet1_pt
      X_test[ientry][13] = i.f_jet1_eta
      X_test[ientry][14] = i.f_jet1_phi
      X_test[ientry][15] = i.f_jet2_pt
      X_test[ientry][16] = i.f_jet2_eta
      X_test[ientry][17] = i.f_jet2_phi
      X_test[ientry][18] = i.f_jet1_e*TMath.CosH(i.f_jet1_eta)
      X_test[ientry][19] = i.f_jet2_e*TMath.CosH(i.f_jet2_eta)
      X_test[ientry][20] = i.f_njets_pass
      ientry = ientry + 1


    if(tree.GetEntries() > 1):
      Y_pred  = loaded_model.predict(X_test)
      Y_pred2 = loaded_model2.predict(X_test)
      Y_pred3 = loaded_model3.predict(X_test)
      dnn_test = [[-1 for i in range(2)] for j in range(len(Y_pred))]
      for j in range(len(Y_pred)):
	dnn_test[j][0] = Y_pred2[j][0]
	dnn_test[j][1] = Y_pred3[j][0]

      #print Y_pred2[0][0]
      Y_pred4 = loaded_model4.predict(dnn_test)
	
      for i in range(len(Y_pred)):
	#print("Prediction: %f" % Y_pred[i])
	n[0] = Y_pred[i]
	n2[0] = Y_pred2[i]
	n3[0] = Y_pred3[i]
	n4[0] = Y_pred4[i]
	tree2.Fill()
    
    else:
      n[0] = 1.
      n2[0] = 1.
      n3[0] = 1.
      n4[0] = 1.
      tree2.Fill()
 
    tree2.Write()
    f2.Close()
    f.Close()
