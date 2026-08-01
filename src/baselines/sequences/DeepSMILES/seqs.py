import deep_smiles
from deep_smiles import DecodeError

DSMILES_converter = deep_smiles.Converter(rings=True, branches=True)

if __name__ == '__main__':
    print("DeepSMILES version: %s" % deep_smiles.__version__)
    converter = deep_smiles.Converter(rings=True, branches=True)
    print(converter)  # record the options used

    encoded = converter.encode("c1cccc(C(=O)Cl)c1")
    print("Encoded: %s" % encoded)

    try:
        decoded = converter.decode(encoded)
    except deep_smiles.DecodeError as e:
        decoded = None
        print("DecodeError! Error message was '%s'" % e.message)

    if decoded:
        print("Decoded: %s" % decoded)