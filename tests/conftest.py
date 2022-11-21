import pytest
import sequencelib as sq
import random

# Python note: some code-duplication, could use fixture parametization, but clearer this way I think...
@pytest.fixture() 
def DNA_string(scope="function"):
    seqstring = random.choices(list(sq.Const.DNA), k=75)
    return seqstring

@pytest.fixture() 
def DNA_string_ambig(scope="function"):
    seqstring = random.choices(list(sq.Const.DNA_maxambig), k=75)
    return seqstring

@pytest.fixture() 
def Protein_string(scope="function"):
    seqstring = random.choices(list(sq.Const.Protein), k=75)
    return seqstring

@pytest.fixture() 
def Protein_string_ambig(scope="function"):
    seqstring = random.choices(list(sq.Const.Protein_maxambig), k=75)
    return seqstring

@pytest.fixture() 
def ASCII_string(scope="function"):
    seqstring = random.choices(list(sq.Const.ASCII), k=75)
    return seqstring

@pytest.fixture() 
def ASCII_string_ambig(scope="function"):
    seqstring = random.choices(list(sq.Const.ASCII), k=75)
    return seqstring
