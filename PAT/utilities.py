def sample_encode(sample_name):
    encode_dict={'total_SIGNAL' : 1,
                            'TTJets_DiLepton' : 16,
			    'TTDileptonic_MiniAOD' : 34,
			    'TTJets_DiLepton_ext' : 35}
    return encode_dict.get(sample_name)

def sample_decode(number):
        decode_dict={1 : 'total_SIGNAL',
                                16 : 'TTJets_DiLepton',
				34 : 'TTDileptonic_MiniAOD',
				35 : 'TTJets_DiLepton_ext'}
        return decode_dict.get(number)


