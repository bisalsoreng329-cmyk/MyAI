"""Tokenizer Module for Project Aura

Provides Byte-Pair Encoding (BPE) tokenization for converting text to tokens.

Classes:
    BPETokenizer: Main tokenizer class
    Vocabulary: Vocabulary management

Example:
    >>> tokenizer = BPETokenizer()
    >>> tokenizer.train(['corpus.txt'], vocab_size=10000)
    >>> tokens = tokenizer.encode('Hello, world!')
    >>> text = tokenizer.decode(tokens)
"""

from .encoder import BPETokenizer
from .vocab import Vocabulary
from .special_tokens import SpecialTokens

__all__ = ['BPETokenizer', 'Vocabulary', 'SpecialTokens']
