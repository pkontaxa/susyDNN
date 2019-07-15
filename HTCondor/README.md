 Instructions on how to use HTCondor for parallelization of the code that writes the DNN output to the original trees (working on lxplus)

1) Modifications for Friends_DNN.py: Change the "workarea" to point to your "susyDNN" directory and "exearea" to point to your "HTCondor" directory

2) Modifications for write_to_tree_Condor.py: Change "work_dir" to point to the "susyDNN" directory and input_dir to the softlinks of the friend trees (Hard-coded for now - will optimize it soon). Finally, set the out_dir to point to the directory where the new friend trees will be stored.

3) Modification for wrap.sh: Modify the source directory to point to your "venv/bin/activate" directory

4) Execute: python Friends_DNN.py --indir /path/to/the/friend_trees --outdir /output/to/store/necessary/scripts/ForSubmission

NOTE: The outdir should be anything but an EOS directory, since EOS does not support batch submission on HTCondor (yet)

5) ./submit_DNN_HTC.sh
