def sample_encode(sample_name):
    encode_dict={'SIGNAL_Compressed' : 1,
                 'SIGNAL_Uncompressed' : 2, 
                            'TTJets_SingleLeptonFromTbar_ext' : 11,
                            'TTJets_SingleLeptonFromT_ext' : 12,
                            'TTJets_DiLepton_ext' : 16,
                            'WJetsToLNu_HT400to600_ext' : 19,
                            'WJetsToLNu_HT600to800_ext' : 20,
                            'WJetsToLNu_HT800to1200_ext' : 21,
                            'WJetsToLNu_HT1200to2500_ext' : 22,
                            'WJetsToLNu_HT2500toInf_ext' : 23}
    return encode_dict.get(sample_name)

def sample_decode(number):
        decode_dict={1 : 'SIGNAL_Compressed',
                     2 : 'SIGNAL_Uncompressed',   
                                11 : 'TTJets_SingleLeptonFromTbar_ext',
                                12 : 'TTJets_SingleLeptonFromT_ext',
                                16 : 'TTJets_DiLepton_ext',
                                19 : 'WJetsToLNu_HT400to600_ext',
                                20 : 'WJetsToLNu_HT600to800_ext',
                                21 : 'WJetsToLNu_HT800to1200_ext',
                                22 : 'WJetsToLNu_HT1200to2500_ext',
                                23 : 'WJetsToLNu_HT2500toInf_ext'}
        return decode_dict.get(number)
