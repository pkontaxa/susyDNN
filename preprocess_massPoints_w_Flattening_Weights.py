import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
import pandas as pd
import root_numpy
import root_pandas
import sys

#sys.path.insert(1, '../')
import utilities as u

work_dir = '/home/pantelis/Desktop/susyDNN/susyDNN/'
out_dir = work_dir+'/preprocessedData_TEST_Flatteness/'
skimmed_dir = work_dir+'preprocessedData/skimmed_nJets_geq4_dPhi0/'

#mass_point_list = ['15_10', '19_01', '19_08', '19_10', '22_01', '22_08']
mass_point_list = ['15_10']

#reweight_bins=[[0., 3.14]]
#reweight_bins = [np.linspace(0., 3.14, 11, endpoint=True)]
reweight_bins = [np.linspace(0., 3.2, 22, endpoint=True)] 
reweight_var=['dPhi']

for mass_point in mass_point_list:

    ### LOAD SIGNAL FILE ###
    signal_file = skimmed_dir+'evVarFriend_T1tttt_MiniAOD_'+str(mass_point)+'_skimmed.root'
    df_sig = root_pandas.read_root(signal_file)

    # Encode the sample name to a new dataframe column
    sample_name_sig = signal_file.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '') # Strip away the irrelevant file path and extension
    df_sig['sampleName'] = u.sample_encode(sample_name_sig)
    df_sig['weights'] = 1

    ### Get dPhi histo (Sig) for usage in weight evaluation ####
    x_sig=np.minimum(df_sig[reweight_var[0]], max(reweight_bins[0]))
    hist_sig, x_sig_edges = np.histogram(x_sig, bins = reweight_bins[0], density=1)
  
    hist_sig = np.asfarray(hist_sig, dtype=np.float32)
    hist_sig_no_zeros = hist_sig[hist_sig>0]
    ############################################################

    ### LOAD BACKGROUND FILES ###
    bkg_filelist = work_dir+'background_file_list_Pantelis_nJets_geq4_dPhi0.txt'
    bkg_files = [line.rstrip('\n') for line in open(bkg_filelist)]

    # Initialize the background dataframe
    df_bkg = root_pandas.read_root(bkg_files[0])

    # Encode the sample names to a new dataframe column
    # This enables even splitting of each sample and the track keeping of individual events
    sample_name_bkg = bkg_files[0].replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
    df_bkg['sampleName'] = u.sample_encode(sample_name_bkg)

    # Remove the first sample from the list of the bkg samples as it was used for the initialization
    bkg_files = bkg_files[1:]

    # Read the rest of the background files and append them to the dataframes 
    # Initialize the dPhi reweighting
    result = {}
    x_bkg = np.minimum(df_bkg[reweight_var[0]], max(reweight_bins[0]))
    hist_bkg_init, x_edges_init = np.histogram(x_bkg, bins=reweight_bins[0],density=1)
    hist_bkg_init = np.asfarray(hist_bkg_init, dtype=np.float32)

    result[sample_name_bkg] = {'x_edged':x_edges_init.tolist(), 'hist_bkg': hist_bkg_init, 'raw_hist_bkg':hist_bkg_init[:].tolist()}

    #Perform initial reweighting
    dPhi_weighted_data = df_bkg['dPhi'].values.flatten()
    dPhiBins = np.digitize(dPhi_weighted_data, reweight_bins[0])-1


    hist_bkg_init_no_zeros = hist_bkg_init[hist_bkg_init > 0] 
    hist_ratio_init = hist_sig/hist_bkg_init    
    hist_ratio_init[np.isinf(hist_ratio_init)] = 0 # get rid of inf  


    df_bkg['weights'] = 1

    for i, i_variable in enumerate(dPhi_weighted_data):
	    df_bkg.loc[i, 'weights'] = hist_ratio_init[dPhiBins[i]]     

    SampleNames = []
    SampleNames.append(sample_name_bkg) 
    for fname in bkg_files:
        
        df_new = root_pandas.read_root(fname)
        sample_name = fname.replace(skimmed_dir+'evVarFriend_', '').replace('_skimmed.root', '')
        df_new['sampleName'] = u.sample_encode(sample_name)       
      
        #pos = (df_new['sampleName'] == 1)        
        #### Perform reweighting ###########
        x = np.minimum(df_new[reweight_var[0]], max(reweight_bins[0]))        
        hist_bkg, x_edges = np.histogram(x, bins=reweight_bins[0],density=1)
        
        hist_bkg = np.asfarray(hist_bkg, dtype=np.float32)
        
        result[sample_name] = {'x_edged':x_edges.tolist(), 'hist_bkg': hist_bkg, 'raw_hist_bkg':hist_bkg[:].tolist()}
        
        SampleNames.append(sample_name) 
       
        
        #Perform reweighting
        dPhi_weighted_data = df_new['dPhi'].values.flatten()
        dPhiBins = np.digitize(dPhi_weighted_data, reweight_bins[0])-1
        hist_bkg_no_zeros = hist_bkg[hist_bkg > 0]  
        hist_ratio = hist_sig/hist_bkg 
        hist_ratio[np.isinf(hist_ratio)] = 0 # get rid of inf  
        df_new['weights'] = 1

        for i, i_variable in enumerate(dPhi_weighted_data): 
		df_new.loc[i, 'weights'] = hist_ratio[dPhiBins[i]]	
                 
        ####################################

        df_bkg = pd.concat([df_bkg, df_new])
        print sample_name
       
    # Add target values to background and signal
    df_bkg['target'] = 0
    df_sig['target'] = 1

    # # Show statistics about the samples
    # print '---- Background ----'
    # print df_bkg.describe(include='all')
    # print '-------- Signal ---------'
    # print df_sig.describe(include='all')

    ### SPLIT INTO TRAINING AND TEST SETS ###
    # Initialize the dataframes
    df_bkg_train = pd.DataFrame()
    df_bkg_test = pd.DataFrame()

    df_sig_train = pd.DataFrame()
    df_sig_test = pd.DataFrame()

    # Do the splitting on each individual sample and choose the size of the split (by default 80/20 split)
    train_split_size = 0.8

    for sample_name in pd.unique(df_bkg['sampleName']):
        df_sample = df_bkg[df_bkg['sampleName'] == sample_name]

        sample_size = df_sample.shape[0]
        split_point = np.ceil(sample_size*train_split_size).astype(np.int32)

        df_sample_train = df_sample.iloc[:split_point, :]
        df_sample_test = df_sample.iloc[split_point:, :]

        df_bkg_train = pd.concat([df_bkg_train, df_sample_train])
        df_bkg_test = pd.concat([df_bkg_test, df_sample_test])

    for sample_name in pd.unique(df_sig['sampleName']):
        df_sample = df_sig[df_sig['sampleName'] == sample_name]

        sample_size = df_sample.shape[0]
        split_point = np.ceil(sample_size*train_split_size).astype(np.int32)

        df_sample_train = df_sample.iloc[:split_point, :]
        df_sample_test = df_sample.iloc[split_point:, :]

        df_sig_train = pd.concat([df_sig_train, df_sample_train])
        df_sig_test = pd.concat([df_sig_test, df_sample_test])

    # Merge the training and test sets
    train = pd.concat([df_bkg_train, df_sig_train], ignore_index=True)
    test = pd.concat([df_bkg_test, df_sig_test], ignore_index=True)

    # Shuffle the training set to properly mix up the samples for the DNN training
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    ### NORMALIZATION ###
    import hickle
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.externals import joblib

    float_cols = ['MET', 'LT', 'HT', 'dPhi', 'Jet1_pt', 'Jet2_pt']
    discrete_cols = ['nTop_Total_Combined', 'nJets30Clean', 'nResolvedTop', 'nBCleaned_TOTAL']
    norm_cols = float_cols + discrete_cols

    # Convert norm_cols to float64 to avoid annoying warning messages from StandardScaler
    train[norm_cols] = train[norm_cols].astype(np.float64)
    test[norm_cols] = test[norm_cols].astype(np.float64)

    # Save the variable names for later usage
    # Note: not sure if it is necessary to save All_input_features
    hickle.dump(list(train), out_dir+'All_input_features_'+str(mass_point)+'.hkl')
    hickle.dump(norm_cols, out_dir+'Normalized_input_features_'+str(mass_point)+'.hkl')

    # Use the StandardScaler to calculate the mean and the standard deviation for each input feature in the training set
   
    normScaler = MinMaxScaler(feature_range=(0.001,0.999)).fit( train[norm_cols].values )

    # Save the scaler for later use
    joblib.dump(normScaler, out_dir+'normScaler_'+mass_point+'.pkl')

    # Normalize the training and the
    train[norm_cols] = normScaler.transform(train[norm_cols].values)
    test[norm_cols] = normScaler.transform(test[norm_cols].values)

    # Save the training and test sets
    train_out=out_dir+'train_set_'+str(mass_point)+'.root'
    train.to_root(train_out, key='tree')

    test_out=out_dir+'test_set_'+str(mass_point)+'.root'
    test.to_root(test_out, key='tree')

    print 'Mass point ', mass_point, ' processed!'


































































