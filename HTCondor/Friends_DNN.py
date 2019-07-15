####### author : Pantelis Kontaxakis ########

import sys,os, re, pprint
import re
from os import listdir
from os.path import isfile, join
import argparse
import commands
import subprocess
import shutil
from ROOT import TFile

import glob


write_to_tree_file="./write_to_tree_Condor.py"

if os.path.exists("submit_DNN_HTC.sh"):
	os.remove("submit_DNN_HTC.sh")

condTEMP = './submit.condor'
wrapTEMP = './wrap.sh'
workarea = '/afs/cern.ch/work/p/pakontax/public/susyDNN/susyDNN'
exearea = '/afs/cern.ch/work/p/pakontax/public/susyDNN/susyDNN/HTCondor'

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Runs DNN on HTC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--indir', help='List of datasets to process', metavar='indir')
        parser.add_argument('--outdir', help='output directory',default=None, metavar='outdir')

	args = parser.parse_args()
	indire = args.indir
	outdire = args.outdir

        if  os.path.exists(outdire):
                des = raw_input(" this dir is already exist : "+str(outdire)+" do you want to remove it [y/n]: ")
                if ( "y" in des or "Y" in des or "Yes" in des) :
                        shutil.rmtree(str(outdire))
                        os.makedirs(str(outdire))
                elif ( "N" in des or  "n" in des or  "No" in des ): print str(outdire) , "will be ovewritten by the job output -- take care"
                else :
                        raise ValueError( "do not understand your potion")
        else : os.makedirs(str(outdire))
        commands = []
        to_merge=''
        logsdir = outdire+"/logs"
        os.makedirs(logsdir)

	# List all the friend trees and process them one by one
	friend_tree_list =  glob.glob(indire+'/*.root')

	for rf in friend_tree_list:
		script_Name=outdire+"/"+rf.split("/")[-1].replace(".root",".py")
		#script_Name=exearea+"/ForSubmission/"+rf.split("/")[-1].replace(".root",".py")
		os.system("cp "+write_to_tree_file+" "+script_Name)
		s1 = open(script_Name).read()
		s1 = s1.replace('@INFILE', rf).replace('@OUTDIR',outdire)
		f1 =open(script_Name, 'w')
		f1.write(s1)
		f1.close()

                cmd = "python "+script_Name

		dirname = outdire+"/"+rf.split("/")[-1].replace(".root","")
		textname = rf.split("/")[-1].replace(".root","")
		os.makedirs(str(dirname))
		os.system("cp "+condTEMP+" "+dirname+"/Condor"+textname+".submit")
		os.system("cp "+wrapTEMP+" "+dirname+"/Warp"+textname+".sh")
		s1 = open(dirname+"/Condor"+textname+".submit").read()
		s1 = s1.replace('@EXESH', dirname+"/Warp"+textname+".sh").replace('@LOGS',logsdir).replace('@NAME',rf.split("/")[-1].replace(".root",""))
		f1 = open(dirname+"/Condor"+textname+".submit", 'w')
		f1.write(s1)
		f1.close()
		s2 = open(dirname+"/Warp"+textname+".sh").read()
		s2 = s2.replace('@WORKDIR',workarea).replace('@EXEDIR',exearea).replace('@COMMAND',cmd)
                f2 = open(dirname+"/Warp"+textname+".sh", 'w')
		f2.write(s2)
		f2.close()
                file = open('submit_DNN_HTC.sh','a')
		file.write("\n")
		file.write("condor_submit "+dirname+"/Condor"+textname+".submit")
		file.close
		
	os.system('chmod a+x submit_DNN_HTC.sh')
	print 'script "submit_DNN_HTC.sh" is READY FOR BATCH'

		
		
	
