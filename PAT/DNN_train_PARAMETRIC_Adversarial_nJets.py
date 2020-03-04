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

mass_point_list = ['total_SIGNAL']

lam = 4.

# Train a neural network separately for each mass point
for mass_point in mass_point_list:
	print 'Training mass point: ' + str(mass_point)

	data_dir = '/home/pantelis/Desktop/susyDNN/susyDNN/preprocessedData_Parametric_ONLY_tt2lep/'
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
       
        NJET_BINS = np.array([0., 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 1])  #Description: nJet>=4 && dPhi>0.5 | nJet bins=4,5,6,7,8,9,>=10 (7 bins in total)
        #NJET_BINS = np.array([0., 0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 1])  #Description: nJet>=4 && dPhi>0.5 | nJet bins=4,5,6,7,8,9,10,11,12,13,14,15,16,17 >=18 (15 bins in total)

        print NJET_BINS
        nJetData= train["nJets30Clean"].values.flatten()
	
	nJetData_Test = test["nJets30Clean"].values.flatten() #Testing sample        
        nJetBins = np.digitize(nJetData, NJET_BINS)-1
        nJetBins_Test = np.digitize(nJetData_Test,  NJET_BINS)-1 #Testing sample
       

        ### to_categorical Test ############
        nJetBins_cat = keras.utils.to_categorical(nJetBins)
        nJetBins_cat_Test = keras.utils.to_categorical(nJetBins_Test) #Testing sample
        ###################################
      
        nJetClasses= np.zeros((len(nJetBins), len(NJET_BINS)))
        for i in range(len(nJetData)):
            nJetClasses[i, nJetBins[i]]=1
       
        #### Convert from np array to pandas DataFrame ###
        column_to_be_added=np.arange(len(nJetData))
        column_to_be_added_Test=np.arange(len(nJetData_Test)) #Testing sample
	

        nJetClasses_w_indices=np.column_stack((column_to_be_added, nJetBins_cat))
        nJetClasses_w_indices.astype(np.int64) 

        nJetClasses_w_indices_Test=np.column_stack((column_to_be_added_Test, nJetBins_cat_Test)) #Testing sample
        nJetClasses_w_indices_Test.astype(np.int64)  #Testing sample 

        df_Convert=pd.DataFrame(data=nJetClasses_w_indices[0:,1:], index=nJetClasses_w_indices[0:,0])
        df_Convert_v2= df_Convert.astype('int64', copy=False)

        df_Convert_Test=pd.DataFrame(data=nJetClasses_w_indices_Test[0:,1:], index=nJetClasses_w_indices_Test[0:,0]) #Testing sample
        df_Convert_v2_Test= df_Convert_Test.astype('int64', copy=False) #Testing sample
        ##################################################


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
        advPremodel = Dense(NJET_BINS.size-1, kernel_initializer='normal', activation='softmax')(advPremodel) 
         
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

        model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])

        opt_DRf = keras.optimizers.SGD(momentum=0., lr=0.01)
        DRf = Model(input=[inputs], output=[model(inputs), advmodel(inputs)])
        DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)        

        opt_DfR = keras.optimizers.SGD(momentum=0., lr=0.01)
      
	DfR = Model(input=[inputs], output=[advmodel(inputs)])
        DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)

        batch_size_1 = 500000
        indices_1 = np.random.permutation(len(train_x))[:batch_size_1]
        classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])

        #Pretraining of "model"
	model.trainable = True
        advmodel.trainable = False          

        numberOfEpochs = 100
        batchSize = 256
	earlystop1 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)        

        model.fit(               train_x,
                                 train_y,
                                 epochs=numberOfEpochs,
				 batch_size = batchSize, # add batch_size (Nov26)
				 callbacks=[earlystop1],
				 validation_split=0.1,
				 shuffle=True)

        print ' - first test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)

        save_path = '/home/pantelis/Desktop/susyDNN/susyDNN/models_Parametric_ONLY_tt2lep/'
        save_name = 'susyDNN_model_'+str(mass_point)+"_nJets4_dPhi05_wo_Adv_onlyTT2l_Parametric"
        model.save(save_path+save_name+'.h5')


        #Pretraining of "advmodel"
	if lam >= 0.0:
		model.trainable = False
		advmodel.trainable = True

                model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
                DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
                DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
 
		#DfR.fit(train_x.iloc[indices_1], df_Convert_v2.iloc[indices_1], nb_epoch=2)                
                adv_numberOfEpochs = 100
                earlystop2 = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10)

                DfR.fit(         train_x[train_y==0],
				 df_Convert_v2[train_y==0],
				 callbacks=[earlystop2],
				 epochs=adv_numberOfEpochs)
        
        #Adversarial training
        losses = {"L_f": [], "L_r": [], "L_f - L_r": []}	
        batch_size = 128      
        
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

                model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])               
                DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
                DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
         
		indices = np.random.permutation(len(train_x))[:batch_size]
                #model.train_on_batch(train_x.iloc[indices],train_y.iloc[indices]) 

 		DRf.train_on_batch(train_x.iloc[indices], [train_y.iloc[indices], df_Convert_v2.iloc[indices]])             
	    	
		#Fit "advmodel"
		if lam >= 0.0:
			model.trainable = False
			advmodel.trainable = True

	                model.compile(loss=[make_loss_model(c=1.0)], optimizer=opt_model, metrics=['accuracy'])
                        DRf.compile(loss=[make_loss_model(c=1.0), make_loss_advmodel(c=-lam)], optimizer=opt_DRf)
        	        DfR.compile(loss=[make_loss_advmodel(c=1.0)], optimizer=opt_DfR)
					
			DfR.fit(train_x[train_y==0], df_Convert_v2[train_y==0], batch_size=batch_size, epochs=1, verbose=1)
        
        # Save the model       
        save_path = '/home/pantelis/Desktop/susyDNN/susyDNN/models_Parametric_ONLY_tt2lep/'
        save_name = 'susyDNN_model_'+str(mass_point)+"_nJets4_dPhi05_lambda"+str(lam)+"_onlyTT2l_7BINS_Epochs200_Parametric_SGDlr0p01"
        
        model.save(save_path+save_name+'.h5') 

        print ' -second test set roc auc: ', round(roc_auc_score(test_y,model.predict(test_x)), 4)
