#!/bin/bash
source /etc/profile
# TODO: these could be filled in from a template
#CMSSW_RELEASE_BASE="/cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/CMSSW_9_4_4"

#Pantelis source /cvmfs/cms.cern.ch/cmsset_default.sh

. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.16.00/x86_64-centos7-gcc48-opt/bin/thisroot.sh

source /afs/cern.ch/work/p/pakontax/public/susyDNN/susyDNN/venv/bin/activate

workdir=@WORKDIR
#melalibdir=${CMSSW_BASE}/lib/slc6_amd64_gcc630/
exedir=`echo @EXEDIR`
export LD_LIBRARY_PATH=${melalibdir}:$LD_LIBRARY_PATH
#cd ${workdir}
#eval `scramv1 runtime -sh`
cd ${exedir}
#export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
#export X509_USER_PROXY=@X509
@COMMAND

