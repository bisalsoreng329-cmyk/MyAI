"""BPE Tokenizer Implementation

Main tokenizer class for encoding/decoding text using learned BPE vocabulary.
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import os

from .vocab import Vocabulary
from .special_tokens import SpecialTokens
from .bpe import BPEMerger

logger = logging.getLogger(__name__)


class BPETokenizer:
    """Byte-Pair Encoding Tokenizer
    
    Tokenizes text by:
    1. Learning merge operations from corpus (BPE training)
    2. Applying merges to convert text to subword tokens
    3. Mapping tokens to integer IDs
    
    Attributes:
        vocab (Vocabulary): Token vocabulary
        merges (List[Tuple[str, str]]): Learned merge operations
        special_tokens (SpecialTokens): Special token definitions
        word_freq (Dict[str, int]): Word frequency from training corpus
    """
    
    def __init__(self, vocab_size: int = 10000):
        """Initialize BPE tokenizer.
        
        Args:
            vocab_size (int): Target vocabulary size (default: 10000)
        """
        self.vocab = Vocabulary()
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = SpecialTokens()
        self.word_freq: Dict[str, int] = {}
        self.vocab_size = vocab_size
        self.bpe_merger = BPEMerger()
        
        # Initialize with special tokens
        for token in self.special_tokens.to_list():
            self.vocab.add_token(token, special=True)
    
    def _count_words(self, texts: List[str]) -> Dict[str, int]:
        """Count word frequencies in texts.
        
        Args:
            texts (List[str]): List of text strings
            
        Returns:
            Dict[str, int]: Word frequencies
        """
        word_freq = defaultdict(int)
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] += 1
        return dict(word_freq)
    
    def _get_vocab_for_training(self, word_freq: Dict[str, int]) -> Dict[Tuple[str, ...], int]:
        """Convert word frequencies to character-level vocabulary.
        
        Args:
            word_freq (Dict[str, int]): Word frequencies
            
        Returns:
            Dict[Tuple[str, ...], int]: Character sequences with frequencies
        """
        vocab = {}
        for word, freq in word_freq.items():
            chars = list(word) + ['</w>']
            vocab[tuple(chars)] = freq
        return vocab
    
    def train(self, texts: List[str], vocab_size: Optional[int] = None) -> None:
        """Train tokenizer on corpus.
        
        Args:
            texts (List[str]): List of text strings to train on
            vocab_size (Optional[int]): Override target vocabulary size
        """
        if vocab_size:
            self.vocab_size = vocab_size
        
        logger.info(f"Training BPE tokenizer with target vocab size {self.vocab_size}")
        
        # Count word frequencies
        self.word_freq = self._count_words(texts)
        logger.info(f"Found {len(self.word_freq)} unique words")
        
        # Initialize character vocabulary
        vocab = self._get_vocab_for_training(self.word_freq)
        
        # Calculate number of merges needed
        num_merges = self.vocab_size - len(self.special_tokens.to_list()) - 256  # 256 for bytes
        
        # Learn BPE merges
        self.merges = self.bpe_merger.learn_merges(vocab, num_merges)
        
        # Build token vocabulary
        self._build_vocab_from_merges()
        logger.info(f"Training complete. Vocab size: {self.vocab.size()}")
    
    def _build_vocab_from_merges(self) -> None:
        """Build vocabulary from learned merges."""
        # Add all individual characters
        for i in range(256):
            self.vocab.add_token(chr(i))
        
        # Add end-of-word marker
        self.vocab.add_token('</w>')
        
        # Add all merge results
        for pair in self.merges:
            merged_token = ''.join(pair)
            self.vocab.add_token(merged_token)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text (str): Text to encode
            add_special_tokens (bool): Add BOS/EOS tokens
            
        Returns:
            List[int]: Token IDs
        """
        if not self.merges:
            logger.warning("Tokenizer not trained. Train before encoding.")
            return []
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.vocab.get_id(self.special_tokens.bos_token))
        
        # Split by whitespace and encode each word
        words = text.split()
        for word in words:
            # Apply BPE merges
            subwords = self.bpe_merger.encode(word, self.merges)
            
            # Convert to token IDs
            for subword in subwords:
                token_id = self.vocab.get_id(subword)
                if token_id == -1:
                    # Unknown token
                    token_id = self.vocab.get_id(self.special_tokens.unk_token)
                token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.vocab.get_id(self.special_tokens.eos_token))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids (List[int]): Token IDs to decode
            
        Returns:
            str: Decoded text
        """
        tokens = [self.vocab.get_token(tid) for tid in token_ids]
        
        # Remove special tokens
        special_tokens_set = set(self.special_tokens.to_list())
        tokens = [t for t in tokens if t not in special_tokens_set]
        
        # Join tokens and remove end-of-word markers
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file.
        
        Args:
            filepath (str): Path to save tokenizer
        """
        import json
        
        data = {
            'vocab': self.vocab.token2id,
            'merges': self.merges,
            'special_tokens': self.special_tokens.to_dict(),
            'vocab_size': self.vocab_size,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved tokenizer to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load tokenizer from file.
        
        Args:
            filepath (str): Path to load tokenizer from
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab.load(filepath.replace('.json', '_vocab.json'))
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        logger.info(f"Loaded tokenizer from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            int: Number of tokens in vocabulary
        """
        return self.vocab.size()
    
    def __repr__(self) -> str:
        return (f"BPETokenizer(vocab_size={self.vocab.size()}, "
                f"merges={len(self.merges)}, trained={len(self.merges) > 0})")
