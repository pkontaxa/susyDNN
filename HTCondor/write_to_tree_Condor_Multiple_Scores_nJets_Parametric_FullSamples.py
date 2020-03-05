#Restrict to one gpu

import pandas as pd
import numpy as np
import root_numpy
import root_numpy
import root_pandas
import glob
import hickle

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.models import load_model

import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)

# Path to the original friend trees
work_dir = '/afs/cern.ch/work/p/pakontax/public/susyDNN/susyDNN'
input_dir = work_dir+'/friend_trees_for_Kimmo/'

# Directory where the new friend trees with the DNN output are saved
#out_dir = '/eos/cms/store/cmst3/user/pakontax/FRIENDS_With_DNN_OUTPUT_V2/'

#out_dir = '@OUTDIR'
out_dir ='/eos/cms/store/user/pakontax/OUTPUT_TOTAL_ADVERSARIAL_nJet_PARAMETRIC_7BINS_AllSamples_Friends_for_Kimmo/'

# Choose the mass point from the list of mass points
mass_point_list = ['total_SIGNAL']
mass_point = str(mass_point_list[0])

# Load the corresponding trained DNN model
def make_loss_model(c):
       def loss_model(y_true, y_pred):
               return c * K.binary_crossentropy(y_true, y_pred)
       return loss_model

######### WO ADVERSARIAL ############################################
model_path_wo_Adversarial = work_dir+'/models_Parametric_26Jan2020/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_wo_Adv_AllSamples_Parametric.h5'
dnn_model_wo_Adversarial = load_model(model_path_wo_Adversarial, custom_objects={'loss_model': make_loss_model(c=1.0)})
####################################################################

########### Lambda = 2 ############################################
model_path_lambda2 = work_dir+'/models_Parametric_26Jan2020/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_lambda2_AllSamples_7BINS_Epochs200_Parametric_SGDlr0p01.h5'
dnn_model_lambda2 = load_model(model_path_lambda2, custom_objects={'loss_model': make_loss_model(c=1.0)})
###################################################################

########### Lambda = 3 ############################################
#model_path_lambda3 = work_dir+'/models_Parametric_ONLY_tt2lep/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_lambda3_onlyTT2l_7BINS_Epochs200_Parametric_SGDlr0p01.h5'
#dnn_model_lambda3 = load_model(model_path_lambda3, custom_objects={'loss_model': make_loss_model(c=1.0)})
###################################################################

########### Lambda = 4 ############################################
model_path_lambda4 = work_dir+'/models_Parametric_26Jan2020/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_lambda4_AllSamples_7BINS_Epochs200_Parametric_SGDlr0p01.h5'
dnn_model_lambda4 = load_model(model_path_lambda4, custom_objects={'loss_model': make_loss_model(c=1.0)})
###################################################################

########### Lambda = 5 ############################################
model_path_lambda5 = work_dir+'/models_Parametric_26Jan2020/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_lambda5_AllSamples_7BINS_Epochs200_Parametric_SGDlr0p01.h5'
dnn_model_lambda5 = load_model(model_path_lambda5, custom_objects={'loss_model': make_loss_model(c=1.0)})
###################################################################

########### Lambda = 6 ############################################
model_path_lambda6 = work_dir+'/models_Parametric_26Jan2020/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_lambda6_AllSamples_7BINS_Epochs200_Parametric_SGDlr0p01.h5'
dnn_model_lambda6 = load_model(model_path_lambda6, custom_objects={'loss_model': make_loss_model(c=1.0)})
###################################################################

########### Lambda = 7 ############################################
model_path_lambda7 = work_dir+'/models_Parametric_26Jan2020/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_lambda7_AllSamples_7BINS_Epochs200_Parametric_SGDlr0p01.h5'
dnn_model_lambda7 = load_model(model_path_lambda7, custom_objects={'loss_model': make_loss_model(c=1.0)})
###################################################################


########### Lambda = 8 ############################################
#model_path_lambda8 = work_dir+'/models_Parametric_ONLY_tt2lep/'+'susyDNN_model_' + mass_point + '_nJets4_dPhi05_lambda8_onlyTT2l_7BINS_Epochs200_Parametric_SGDlr0p01.h5'
#dnn_model_lambda8 = load_model(model_path_lambda8, custom_objects={'loss_model': make_loss_model(c=1.0)})
###################################################################



#for friend_tree in friend_tree_list:
friend_tree = "@INFILE"
sample_name = friend_tree.replace(input_dir, '').replace('.root', '') # Get rid of the file path and extension
friend_tree_out=out_dir+sample_name+'_DNN_Output_'+str(mass_point)+'.root'

#load_dir = work_dir+'/preprocessedData/'
load_dir = work_dir+'/preprocessedData_Parametric_26Jan2020/'
normScaler = joblib.load(load_dir+'normScaler_'+mass_point+'.pkl')
normalized_features = hickle.load(load_dir+'Normalized_input_features_'+str(mass_point)+'.hkl')

# Load the friend tree to a dataframe
#df = root_pandas.read_root(friend_tree)

normalized_features2 = ['MET', 'LT', 'dPhi', 'HT', 'nJets30Clean', 'Jet1_pt', 'Jet2_pt', 'dM_Go_LSP', 'nResolvedTop', 'nTop_Total_Combined', 'nBCleaned_TOTAL']

for df in root_pandas.read_root(friend_tree,chunksize=100000):

        # Load the list of normalized variables and the StandardScaler
	#load_dir = work_dir+'/preprocessedData/'

	#normScaler = joblib.load(load_dir+'normScaler_'+mass_point+'.pkl')
	#normalized_features = hickle.load(load_dir+'Normalized_input_features_'+str(mass_point)+'.hkl')

	# Process the friend tree to be suitable for the DNN
	df_dnn_input = df[normalized_features].copy()
	df_dnn_input[normalized_features] = normScaler.transform(df_dnn_input[normalized_features].values)

        df_dnn_input2 = df_dnn_input[normalized_features2].copy()


	# Get the prediction from the DNN for each event
	dnn_output_wo_Adversarial = np.squeeze( dnn_model_wo_Adversarial.predict(df_dnn_input2) ) # squeeze to get rid of one useless dimension
        dnn_output_lambda2 = np.squeeze( dnn_model_lambda2.predict(df_dnn_input2) )
        #dnn_output_lambda3 = np.squeeze( dnn_model_lambda3.predict(df_dnn_input2) )
        dnn_output_lambda4 = np.squeeze( dnn_model_lambda4.predict(df_dnn_input2) )
        dnn_output_lambda5 = np.squeeze( dnn_model_lambda5.predict(df_dnn_input2) )
        dnn_output_lambda6 = np.squeeze( dnn_model_lambda6.predict(df_dnn_input2) )
        dnn_output_lambda7 = np.squeeze( dnn_model_lambda7.predict(df_dnn_input2) )
        #dnn_output_lambda8 = np.squeeze( dnn_model_lambda8.predict(df_dnn_input2) )

	# Add the DNN output to the original friend tree
	dnn_output_wo_Adversarial_col_name = 'DNN_Output_' + mass_point + '_wo_Adversarial_nJets4_dPhi05_PARAMETRIC_AllSamples'
	df[dnn_output_wo_Adversarial_col_name] = dnn_output_wo_Adversarial
        
        dnn_output_lambda2_col_name = 'DNN_Output_' + mass_point + '_nJets4_dPhi05_lambda2_PARAMETRIC_AllSamples'
        df[dnn_output_lambda2_col_name] = dnn_output_lambda2

        #dnn_output_lambda3_col_name = 'DNN_Output_' + mass_point + '_nJets4_dPhi05_lambda3_PARAMETRIC_only_tt2l'
        #df[dnn_output_lambda3_col_name] = dnn_output_lambda3

        dnn_output_lambda4_col_name = 'DNN_Output_' + mass_point + '_nJets4_dPhi05_lambda4_PARAMETRIC_AllSamples'
        df[dnn_output_lambda4_col_name] = dnn_output_lambda4
        
        dnn_output_lambda5_col_name = 'DNN_Output_' + mass_point + '_nJets4_dPhi05_lambda5_PARAMETRIC_AllSamples'
        df[dnn_output_lambda5_col_name] = dnn_output_lambda5

        dnn_output_lambda6_col_name = 'DNN_Output_' + mass_point + '_nJets4_dPhi05_lambda6_PARAMETRIC_AllSamples'
        df[dnn_output_lambda6_col_name] = dnn_output_lambda6

        dnn_output_lambda7_col_name = 'DNN_Output_' + mass_point + '_nJets4_dPhi05_lambda7_PARAMETRIC_AllSamples'
        df[dnn_output_lambda7_col_name] = dnn_output_lambda7

        #dnn_output_lambda8_col_name = 'DNN_Output_' + mass_point + '_nJets4_dPhi05_lambda8_PARAMETRIC_only_tt2l'
        #df[dnn_output_lambda8_col_name] = dnn_output_lambda8
        
	# Extract the name of the sample
	#sample_name = friend_tree.replace(input_dir, '').replace('.root', '') # Get rid of the file path and extension

	# Save the new friend tree to the output directory
	#friend_tree_out=out_dir+sample_name+'_DNN_Output_'+str(mass_point)+'.root'
	df.to_root(friend_tree_out, mode='a', key='sf/t')

print 'Processed: ', sample_name
