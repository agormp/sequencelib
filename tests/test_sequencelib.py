import pytest
import sequencelib as sq
import random


###################################################################################################
###################################################################################################

# Tests for loose functions

###################################################################################################
###################################################################################################

class Test_find_seqtype:
    
    seqlen = 75
    
    def test_DNA_noambig(self):
        DNA_string = "".join(random.choices(list(sq.Const.DNA), k=self.seqlen))
        assert sq.find_seqtype(DNA_string) == "DNA"
        assert sq.find_seqtype(list(DNA_string)) == "DNA"
        assert sq.find_seqtype(set(DNA_string)) == "DNA"
        
    def test_DNA_ambig(self):
        DNA_string_ambig = "".join(random.choices(list(sq.Const.DNA_maxambig), k=self.seqlen))
        assert sq.find_seqtype(DNA_string_ambig) == "DNA"
        assert sq.find_seqtype(list(DNA_string_ambig)) == "DNA"
        assert sq.find_seqtype(set(DNA_string_ambig)) == "DNA"
        
    def test_Protein_noambig(self):
        Protein_string = "".join(random.choices(list(sq.Const.Protein), k=self.seqlen))
        assert sq.find_seqtype(Protein_string) == "protein"
        assert sq.find_seqtype(list(Protein_string)) == "protein"
        assert sq.find_seqtype(set(Protein_string)) == "protein"
        
    def test_Protein_ambig(self):
        Protein_string_ambig = "".join(random.choices(list(sq.Const.Protein_maxambig), k=self.seqlen))
        assert sq.find_seqtype(Protein_string_ambig) == "protein"
        assert sq.find_seqtype(list(Protein_string_ambig)) == "protein"
        assert sq.find_seqtype(set(Protein_string_ambig)) == "protein"

    def test_ASCII(self):
        ASCII_string = "".join(random.choices(list(sq.Const.ASCII), k=self.seqlen))
        assert sq.find_seqtype(ASCII_string) == "ASCII"
        assert sq.find_seqtype(list(ASCII_string)) == "ASCII"
        assert sq.find_seqtype(set(ASCII_string)) == "ASCII"
        
    def test_unrecognized_raises(self):
        ASCII_string = "".join(random.choices(list(sq.Const.ASCII), k=self.seqlen))
        unknown = ASCII_string + "ØÆÅ=)&%#"
        with pytest.raises(sq.SeqError):
            sq.find_seqtype(unknown)

###################################################################################################

class Test_seqtype_attributes:
    
    def test_DNA(self):
        assert (sq.seqtype_attributes("DNA") 
                == (set("ACGTURYMKWSBDHVN"), set("URYMKWSBDHVN")))

    def test_Protein(self):
        assert (sq.seqtype_attributes("protein") 
                == (set("ACDEFGHIKLMNPQRSTVWYBZX"), set("BXZ")))
        
    def test_ASCII(self):
        assert (sq.seqtype_attributes("ASCII") 
                == (set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,._"), set()))
    
    def test_unknown_raises(self):
        with pytest.raises(sq.SeqError):
            sq.seqtype_attributes("normannisk")

###################################################################################################

class Test_indices:
    
    def test_singlesubstring(self):
        inputstring = "AAAAAAAAAA AAAAAAAA AAAAAAA Here AAAAAAAA AAAAAA"
        assert sq.indices(inputstring, "Here") == set([28])
        
    def test_triplesubstring(self):
        inputstring = "AAAAAAAAAA Here AAAAAAAA AAAAAAA Here AAAAAAAA AAAHereAAA"
        assert sq.indices(inputstring, "Here") == set([11,33,50])
        
    def test_overlapping(self):
        inputstring = "AAAAAAAAAA hehehehe AAAAAAA hehe AAAAA"
        assert sq.indices(inputstring, "hehe") == set([11,13,15,28])
    
###################################################################################################

class Test_remove_comments:
    
    def test_unnested_1chardelim(self):
        input = "This sentence [which is an example] contains one comment"
        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == "This sentence  contains one comment")

    def test_unnested__1chardelim_multiline(self):
        input = """This sentence [which is an example of a string with
                a multiline un-nested comment] contains one comment"""
        expexted_output = """This sentence  contains one comment"""
        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == expexted_output)

    def test_nested(self):
        input = "This sentence [which is an example [or is it?]] contains nested comments"
        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == "This sentence  contains nested comments")
                
    def test_nested__1chardelim_multiline(self):
        input = """This sentence [which is also an example] is far more complicated.
        [or is it [and here 'it' refers to the sentence]]. It contains nested
        comments [like[this]] and newlines. It also contains nested comments
        spread over multiple lines like this: [I am not sure this type of comment
        will ever appear in actual sequences [it might in trees though]]. The end"""

        expexted_output = """This sentence  is far more complicated.
        . It contains nested
        comments  and newlines. It also contains nested comments
        spread over multiple lines like this: . The end"""

        assert (sq.remove_comments(input, leftdelim="[", rightdelim="]")
                == expexted_output)

    def test_unnested_multichardelim(self):
        input = "This sentence <B>which is an example<E> contains one comment"
        assert (sq.remove_comments(input, leftdelim="<B>", rightdelim="<E>")
                == "This sentence  contains one comment")

    def test_nested__multichardelim_multiline(self):
        input = """This sentence <com>which is also an example</com> is far more complicated.
        <com>or is it <com>and here 'it' refers to the sentence</com></com>. It contains nested
        comments <com>like<com>this</com></com> and newlines. It also contains nested comments
        spread over multiple lines like this: <com>I am not sure this type of comment
        will ever appear in actual sequences <com>it might in trees though</com></com>. The end"""

        expexted_output = """This sentence  is far more complicated.
        . It contains nested
        comments  and newlines. It also contains nested comments
        spread over multiple lines like this: . The end"""

        assert (sq.remove_comments(input, leftdelim="<com>", rightdelim="</com>")
                == expexted_output)


###################################################################################################
###################################################################################################

# Tests for class XXX

###################################################################################################
###################################################################################################

