"""Vocabulary Management for BPE Tokenizer

Handles vocabulary creation, storage, and retrieval.
"""

import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class Vocabulary:
    """Manages BPE vocabulary and token mappings.
    
    Attributes:
        token2id (Dict[str, int]): Maps tokens to their integer IDs
        id2token (Dict[int, str]): Maps token IDs back to tokens
        merges (List[Tuple[str, str]]): List of BPE merge operations
        special_tokens (Dict[str, int]): Special token mappings
    """
    
    def __init__(self):
        """Initialize vocabulary."""
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens: Dict[str, int] = {}
        self._next_id = 0
    
    def add_token(self, token: str, special: bool = False) -> int:
        """Add token to vocabulary.
        
        Args:
            token (str): Token string to add
            special (bool): Whether this is a special token
            
        Returns:
            int: Token ID
        """
        if token in self.token2id:
            return self.token2id[token]
        
        token_id = self._next_id
        self.token2id[token] = token_id
        self.id2token[token_id] = token
        self._next_id += 1
        
        if special:
            self.special_tokens[token] = token_id
        
        logger.debug(f"Added token '{token}' with ID {token_id}")
        return token_id
    
    def add_merge(self, merge: Tuple[str, str]) -> None:
        """Record a BPE merge operation.
        
        Args:
            merge (Tuple[str, str]): Pair of tokens to merge
        """
        self.merges.append(merge)
        logger.debug(f"Recorded merge: {merge[0]} + {merge[1]}")
    
    def get_id(self, token: str) -> int:
        """Get token ID from token string.
        
        Args:
            token (str): Token string
            
        Returns:
            int: Token ID, or -1 if not found
        """
        return self.token2id.get(token, -1)
    
    def get_token(self, token_id: int) -> str:
        """Get token string from token ID.
        
        Args:
            token_id (int): Token ID
            
        Returns:
            str: Token string, or "<UNK>" if not found
        """
        return self.id2token.get(token_id, "<UNK>")
    
    def size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            int: Number of tokens in vocabulary
        """
        return len(self.token2id)
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file.
        
        Args:
            filepath (str): Path to save vocabulary
        """
        data = {
            'token2id': self.token2id,
            'merges': self.merges,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load vocabulary from JSON file.
        
        Args:
            filepath (str): Path to load vocabulary from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token2id = data['token2id']
        self.id2token = {int(k): v for k, v in 
                        {int(k): v for k, v in data['token2id'].items() 
                         if isinstance(k, str)}.items()}
        self.merges = data['merges']
        self.special_tokens = data.get('special_tokens', {})
        self._next_id = max(self.token2id.values()) + 1
        logger.info(f"Loaded vocabulary from {filepath} (size: {self.size()})")
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.size()
    
    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token2id
    
    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size()}, merges={len(self.merges)})"
