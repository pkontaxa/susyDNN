import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
import pandas as pd
import root_numpy
import root_pandas

import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)
import keras.callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from Plots_Losses import plot_losses
#mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']

mass_point_list = ['15_10']

lam = 5.

# Train a neural network separately for each mass point
for mass_point in mass_point_list:
	print 'Training mass point: ' + str(mass_point)

	data_dir = '/home/pantelis/Desktop/susyDNN/susyDNN/preprocessedData/Preprocessed_nJets_geq4_dPhi05/'
	train_path = data_dir+'train_set_'+str(mass_point)+'.root'
	test_path = data_dir+'test_set_'+str(mass_point)+'.root'

	# Read the ROOT files for background and signal samples and put them into dataframes
	train = root_pandas.read_root(train_path, 'tree')
	test = root_pandas.read_root(test_path, 'tree')

	# Drop the sample names at this point
	train = train.drop(columns=['sampleName'])
	test = test.drop(columns=['sampleName'])

	# Separate input features and the target + include dPhi on test_z for adversarial training
	train_y = train['target']
	test_y = test['target']

	train_x = train.drop(columns=['target'])
	test_x = test.drop(columns=['target'])

        #train_x = train.drop(columns=['target','MET','LT','HT','nJets30Clean','Jet1_pt','Jet2_pt','nResolvedTop','nTop_Total_Combined','nBCleaned_TOTAL'])
        #test_x = test.drop(columns=['target','MET','LT','HT','nJets30Clean','Jet1_pt','Jet2_pt','nResolvedTop','nTop_Total_Combined','nBCleaned_TOTAL']) 

        ### DPHI Bins ##############
        #DPHI_BINS = np.linspace(0., 1., 11, endpoint=True)
        #DPHI_BINS = np.array([0., 0.2226, 0.2229, 0.2790, 1.])

        #DPHI_BINS = np.array([0., 0.05, 0.1, 0.15, 1.])  #Description: nJet>=4 && dPhi>0.5 | nJet bins=4,5,6,>=7 (4 bins in total)
        DPHI_BINS = np.array([0., 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 1])  #Description: nJet>=4 && dPhi>0.5 | nJet bins=4,5,6,7,8,9,>=10 (7 bins in total)

        print DPHI_BINS
        dPhiData= train["nJets30Clean"].values.flatten()
	
	dPhiData_Test = test["nJets30Clean"].values.flatten() #Testing sample
        #print dPhiData
        dPhiBins = np.digitize(dPhiData, DPHI_BINS)-1
        dPhiBins_Test = np.digitize(dPhiData_Test,  DPHI_BINS)-1 #Testing sample
        #print dPhiBins

        ### to_categorical Test ############
        dPhiBins_cat = keras.utils.to_categorical(dPhiBins)
        dPhiBins_cat_Test = keras.utils.to_categorical(dPhiBins_Test) #Testing sample
        #print dPhiBins_cat
        ###################################
      
        dPhiClasses= np.zeros((len(dPhiBins), len(DPHI_BINS)))
        for i in range(len(dPhiData)):
            dPhiClasses[i, dPhiBins[i]]=1
     
        #### Convert from np array to pandas DataFrame ###
        column_to_be_added=np.arange(len(dPhiData))
        column_to_be_added_Test=np.arange(len(dPhiData_Test)) #Testing sample
	

        #column_to_be_added.astype(np.int64)
        dPhiClasses_w_indices=np.column_stack((column_to_be_added, dPhiBins_cat))
        dPhiClasses_w_indices.astype(np.int64) 

        dPhiClasses_w_indices_Test=np.column_stack((column_to_be_added_Test, dPhiBins_cat_Test)) #Testing sample
        dPhiClasses_w_indices_Test.astype(np.int64)  #Testing sample 

  
        #print dPhiClasses_w_indices
        df_Convert=pd.DataFrame(data=dPhiClasses_w_indices[0:,1:], index=dPhiClasses_w_indices[0:,0])
        df_Convert_v2= df_Convert.astype('int64', copy=False)

        df_Convert_Test=pd.DataFrame(data=dPhiClasses_w_indices_Test[0:,1:], index=dPhiClasses_w_indices_Test[0:,0]) #Testing sample
        df_Convert_v2_Test= df_Convert_Test.astype('int64', copy=False) #Testing sample

        #print df_Convert_v2[train_y==1]
        ##################################################


        '''
        DPHI_BINS = np.linspace(-1., 3., 20, endpoint=True)
        #print DPHI_BINS.size
        dPhiData= train["dPhi"]
        #print dPhiData
        dPhiBins = np.digitize(dPhiData, DPHI_BINS)-1
        #print dPhiBins
        ''' 
        ###########################

	### Build the neural network ###
	from keras.models import Sequential,Model
	from keras.layers import Input,Dense,Activation,Dropout
	from keras import optimizers
	from sklearn.utils import class_weight

	# Define the architecture
        inputs = Input(shape = (train_x.shape[1], ))
        premodel = Dense(100, kernel_initializer='normal', activation='relu')(inputs) 
        premodel = Dropout(0.2)(premodel)
        premodel = Dense(100, kernel_initializer='normal', activation='relu')(premodel)
        premodel = Dropout(0.2)(premodel)
        premodel = Dense(50, kernel_initializer='normal', activation='relu')(premodel)
        premodel = Dense(1, kernel_initializer='normal', activation='sigmoid')(premodel)
 
        model = Model(input=[inputs], output=[premodel])


        # Adversarial network architecture
        advPremodel = model(inputs)
        advPremodel = Dense(100, kernel_initializer='normal', activation='relu')(advPremodel)
        advPremodel = Dropout(0.2)(advPremodel)
        advPremodel = Dense(100, kernel_initializer='normal', activation='relu')(advPremodel)
        advPremodel = Dropout(0.2)(advPremodel)
        advPremodel = Dense(50, kernel_initializer='normal', activation='relu')(advPremodel)
        advPremodel = Dense(DPHI_BINS.size-1, kernel_initializer='normal', activation='softmax')(advPremodel) 
         
        advmodel = Model(input=[inputs], output=[advPremodel])       
                
        ###########################################

        #### Make Loss functions ####
        def make_loss_model(c):
		def loss_model(y_true, y_pred):
			return c * K.binary_crossentropy(y_true, y_pred)
		return loss_model

        def make_loss_advmodel(c):
	        def loss_advmodel(z_true, z_pred):
			return c * K.categorical_crossentropy(z_true, z_pred)
		return loss_advmodel
        ############################
      
   	opt_model = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

        model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model)

        #opt_DRf = keras.optimizers.SGD(momentum=0., lr=0.001)
        opt_DRf = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        DRf = Model(input=[inputs], output=[model(inputs), advmodel(inputs)])
        DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
        #DRf.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], loss_weights=[1, -lam], optimizer=opt_DRf)

        #opt_DfR = keras.optimizers.SGD(momentum=0., lr=0.001)
        opt_DfR = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        print ("Learning rate for adversary: 0.001")
        #opt_DfR = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	DfR = Model(input=[inputs], output=[advmodel(inputs)])
        DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

        batch_size_1 = 500000
        indices_1 = np.random.permutation(len(train_x))[:batch_size_1]
        classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])

        #Pretraining of "model"
	model.trainable = True
        advmodel.trainable = False
        #model.fit(train_x.iloc[indices_1], train_y.iloc[indices_1], epochs=2)   

        model.fit(train_x, train_y, epochs=10)
        print ' - first test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)

        #Pretraining of "advmodel"
	if lam >= 0.0:
		model.trainable = False
		advmodel.trainable = True

                model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model)
                DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
 
		#DfR.fit(train_x.iloc[indices_1], df_Convert_v2.iloc[indices_1], nb_epoch=2)                
                DfR.fit(train_x[train_y==0], df_Convert_v2[train_y==0], epochs=15)
        
        #Adversarial training
        losses = {"L_f": [], "L_r": [], "L_f - L_r": []}
	
        batch_size = 128
        #print batch_size
        
        num_epochs=200        
        for i in range(num_epochs):
	    	print i
		l = DRf.evaluate(test_x, [test_y, df_Convert_v2_Test])
		losses["L_f - L_r"].append(l[0][None][0])
		losses["L_f"].append(l[1][None][0])
		losses["L_r"].append(-l[2][None][0])
		print(losses["L_r"][-1] / lam)

		plot_losses(i, losses, lam, num_epochs)

            	#Fit "model"
                model.trainable = True
		advmodel.trainable = False

                model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model)
                DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
                DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)

                #print "line179"
		indices = np.random.permutation(len(train_x))[:batch_size]
                #model.train_on_batch(train_x.iloc[indices],train_y.iloc[indices]) 

 		DRf.train_on_batch(train_x.iloc[indices], [train_y.iloc[indices], df_Convert_v2.iloc[indices]])
                #DRf.fit(train_x.iloc[indices_1], [train_y.iloc[indices_1], df_Convert_v2.iloc[indices_1]], verbose=1)
	    	
		#Fit "advmodel"
		if lam >= 0.0:
			model.trainable = False
			advmodel.trainable = True

	                model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model)
        	        DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
			
			#DfR.fit(train_x.iloc[indices_1], df_Convert_v2.iloc[indices_1], batch_size=batch_size, nb_epoch=1, verbose=1)
			DfR.fit(train_x[train_y==0], df_Convert_v2[train_y==0], batch_size=batch_size, epochs=1, verbose=1)
        
       # Save the model
        save_path = '/home/pantelis/Desktop/susyDNN/susyDNN/models/'
        save_name = 'susyDNN_model_'+str(mass_point)+"_nJets4_dPhi05_lambda5_7BINS_Epochs200"
        
        model.save(save_path+save_name+'.h5') 

        # Print final test set scores
        #test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
        #print ' - test set loss: ', round(test_loss, 4)
        #print ' - test set accuracy: ', round(test_acc, 4)
        print ' -second test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)
