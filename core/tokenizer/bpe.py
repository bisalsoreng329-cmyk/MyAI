"""Byte-Pair Encoding (BPE) Algorithm Implementation

Core BPE tokenization algorithm for merging frequently occurring byte pairs.
"""

from typing import List, Tuple, Dict, Counter as CounterType
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class BPEMerger:
    """Performs Byte-Pair Encoding merge operations.
    
    The BPE algorithm iteratively identifies the most frequent pair of tokens
    and merges them, reducing vocabulary while preserving statistical patterns.
    """
    
    def __init__(self):
        """Initialize BPE merger."""
        self.merges: List[Tuple[str, str]] = []
    
    @staticmethod
    def get_stats(vocab: Dict[Tuple[str, ...], int]) -> Counter:
        """Calculate frequency of adjacent token pairs.
        
        Args:
            vocab (Dict[Tuple[str, ...], int]): Word vocabulary with frequencies
            
        Returns:
            Counter: Pair frequencies
        """
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    @staticmethod
    def merge_vocab(pair: Tuple[str, str], v_in: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge a pair of tokens throughout vocabulary.
        
        Args:
            pair (Tuple[str, str]): Token pair to merge
            v_in (Dict[Tuple[str, ...], int]): Input vocabulary
            
        Returns:
            Dict[Tuple[str, ...], int]: Vocabulary with merged pairs
        """
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in v_in:
            new_word = ' '.join(word).replace(bigram, replacement).split()
            v_out[tuple(new_word)] = v_in[word]
        
        return v_out
    
    def learn_merges(self, vocab: Dict[Tuple[str, ...], int], num_merges: int) -> List[Tuple[str, str]]:
        """Learn BPE merge operations from vocabulary.
        
        Args:
            vocab (Dict[Tuple[str, ...], int]): Word vocabulary with frequencies
            num_merges (int): Number of merge operations to perform
            
        Returns:
            List[Tuple[str, str]]: List of merge operations
        """
        self.merges = []
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                logger.warning(f"No more pairs to merge after {i} iterations")
                break
            
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            if (i + 1) % max(1, num_merges // 10) == 0:
                logger.info(f"BPE Progress: {i + 1}/{num_merges} merges completed")
        
        logger.info(f"BPE learning complete: {len(self.merges)} merges")
        return self.merges
    
    def encode(self, word: str, merges: List[Tuple[str, str]]) -> List[str]:
        """Apply learned merges to encode a word.
        
        Args:
            word (str): Word to encode
            merges (List[Tuple[str, str]]): List of merge operations
            
        Returns:
            List[str]: Encoded word as list of subword tokens
        """
        word_tokens = list(word) + ['</w>']
        
        for merge in merges:
            bigram = tuple(merge)
            merged_word = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and 
                    (word_tokens[i], word_tokens[i + 1]) == bigram):
                    merged_word.append(''.join(bigram))
                    i += 2
                else:
                    merged_word.append(word_tokens[i])
                    i += 1
            word_tokens = merged_word
        
        return word_tokens
