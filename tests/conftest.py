import pytest
import sequencelib as sq
import random

# Not using these anymore, but keep to remind myself about format
@pytest.fixture() 
def DNA_string(scope="function"):
    seqstring = random.choices(list(sq.Const.DNA), k=75)
    seqstring = "".join(seqstring)
    return seqstring

@pytest.fixture() 
def DNA_string_ambig(scope="function"):
    seqstring = random.choices(list(sq.Const.DNA_maxambig), k=75)
    seqstring = "".join(seqstring)
    return seqstring

@pytest.fixture() 
def Protein_string(scope="function"):
    seqstring = random.choices(list(sq.Const.Protein), k=75)
    seqstring = "".join(seqstring)
    return seqstring

@pytest.fixture() 
def Protein_string_ambig(scope="function"):
    seqstring = random.choices(list(sq.Const.Protein_maxambig), k=75)
    seqstring = "".join(seqstring)
    return seqstring

@pytest.fixture() 
def ASCII_string(scope="function"):
    seqstring = random.choices(list(sq.Const.ASCII), k=75)
    seqstring = "".join(seqstring)
    return seqstring

