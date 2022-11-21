import pytest
import sequencelib as sq

###############################################################################

class Test_find_seqtype:
    
    def test_DNA_noambig(self, DNA_string):
        assert sq.find_seqtype(DNA_string) == "DNA"
        assert sq.find_seqtype(list(DNA_string)) == "DNA"
        assert sq.find_seqtype(set(DNA_string)) == "DNA"
        
    def test_DNA_ambig(self, DNA_string_ambig):
        assert sq.find_seqtype(DNA_string_ambig) == "DNA"
        assert sq.find_seqtype(list(DNA_string_ambig)) == "DNA"
        assert sq.find_seqtype(set(DNA_string_ambig)) == "DNA"
        
    def test_Protein_noambig(self, Protein_string):
        assert sq.find_seqtype(Protein_string) == "protein"
        assert sq.find_seqtype(list(Protein_string)) == "protein"
        assert sq.find_seqtype(set(Protein_string)) == "protein"
        
    def test_Protein_ambig(self, Protein_string_ambig):
        assert sq.find_seqtype(Protein_string_ambig) == "protein"
        assert sq.find_seqtype(list(Protein_string_ambig)) == "protein"
        assert sq.find_seqtype(set(Protein_string_ambig)) == "protein"

    def test_ASCII(self, ASCII_string):
        assert sq.find_seqtype(ASCII_string) == "ASCII"
        assert sq.find_seqtype(list(ASCII_string)) == "ASCII"
        assert sq.find_seqtype(set(ASCII_string)) == "ASCII"

###############################################################################