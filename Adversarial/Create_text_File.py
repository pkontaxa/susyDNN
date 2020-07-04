import sys, os, re
from os import listdir
from os.path import isfile, join

out_dir ='/afs/cern.ch/work/p/pakontax/private/susyDNN/susyDNN/plots_Adversarial_28June2020_Splitted_Signal/'

def fill_txt_file(i, lam, JS1, JS2, Signal_Uncompressed_Ineff, Signal_Compressed_Ineff, Bkg_Uncompressed_Ineff):
	
	script_Name="Best_Trainings_lambda"+str(lam)+".txt"
               
        if os.path.exists(out_dir+script_Name):
        	f1 = open(out_dir+script_Name, 'a')
        else:
                f1 = open(out_dir+script_Name, 'w')

        f1.write("Epoch: "+str(i))
        f1.write('\nDNN output Jensen-Shannon distance (nJet = [6,7,8] vs. nJet = [4,5]): ' + str(JS1))
        f1.write('\nDNN output Jensen-Shannon distance (nJet >= 9 vs. nJet = [6,7,8]): ' + str(JS2))
        f1.write('\nSignal Inefficiency Compressed: ' + str(Signal_Compressed_Ineff))          
        f1.write('\nSignal Inefficiency Uncompressed: ' + str(Signal_Uncompressed_Ineff))
        f1.write('\nBkg Inefficiency Uncompressed: ' + str(Bkg_Uncompressed_Ineff))
        f1.write('\n\n')
        
        f1.close()
