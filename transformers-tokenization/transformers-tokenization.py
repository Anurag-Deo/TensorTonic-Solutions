import numpy as np
from typing import List, Dict
from collections import defaultdict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.st = set()
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        # self.st.insert(self.pad_token)
        # self.st.insert(self.unk_token)
        # self.st.insert(self.bos_token)
        # self.st.insert(self.eos_token)
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        for text in texts:
            tokens = text.split()
            self.st.update(tokens)
        i = 4
        for item in self.st:
            self.word_to_id[item] = i
            self.id_to_word[i] = item
            i += 1
        self.vocab_size = len(self.word_to_id)
        
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        tokens = text.split()
        encoded = []
        for item in tokens:
            encoded.append(self.word_to_id.get(item, self.word_to_id[self.unk_token]))
        return encoded
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        decoded = ""
        for _id in ids:
            decoded += self.id_to_word.get(_id,"")
            decoded += " "
        return decoded.strip()
