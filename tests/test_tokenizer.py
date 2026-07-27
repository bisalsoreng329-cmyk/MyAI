"""Test suite for Tokenizer Module

Comprehensive unit tests for BPE tokenization.
"""

import pytest
import tempfile
import os
from core.tokenizer import BPETokenizer, Vocabulary, SpecialTokens
from core.tokenizer.bpe import BPEMerger


class TestSpecialTokens:
    """Test special token definitions."""
    
    def test_special_tokens_initialization(self):
        """Test SpecialTokens initialization."""
        tokens = SpecialTokens()
        assert tokens.pad_token == "<PAD>"
        assert tokens.unk_token == "<UNK>"
        assert tokens.bos_token == "<BOS>"
        assert tokens.eos_token == "<EOS>"
    
    def test_special_tokens_to_dict(self):
        """Test converting special tokens to dictionary."""
        tokens = SpecialTokens()
        token_dict = tokens.to_dict()
        assert token_dict['pad'] == "<PAD>"
        assert token_dict['unk'] == "<UNK>"
        assert len(token_dict) == 7
    
    def test_special_tokens_to_list(self):
        """Test converting special tokens to list."""
        tokens = SpecialTokens()
        token_list = tokens.to_list()
        assert len(token_list) == 7
        assert "<PAD>" in token_list
        assert "<MASK>" in token_list


class TestVocabulary:
    """Test vocabulary management."""
    
    def test_vocabulary_initialization(self):
        """Test vocabulary initialization."""
        vocab = Vocabulary()
        assert vocab.size() == 0
        assert len(vocab) == 0
    
    def test_add_token(self):
        """Test adding tokens to vocabulary."""
        vocab = Vocabulary()
        token_id = vocab.add_token("hello")
        assert token_id == 0
        assert vocab.size() == 1
        assert vocab.get_id("hello") == 0
    
    def test_add_duplicate_token(self):
        """Test adding duplicate tokens returns same ID."""
        vocab = Vocabulary()
        id1 = vocab.add_token("hello")
        id2 = vocab.add_token("hello")
        assert id1 == id2
        assert vocab.size() == 1
    
    def test_special_token_tracking(self):
        """Test tracking special tokens."""
        vocab = Vocabulary()
        vocab.add_token("<PAD>", special=True)
        vocab.add_token("hello", special=False)
        assert "<PAD>" in vocab.special_tokens
        assert "hello" not in vocab.special_tokens
    
    def test_get_token(self):
        """Test retrieving token from ID."""
        vocab = Vocabulary()
        token_id = vocab.add_token("world")
        assert vocab.get_token(token_id) == "world"
    
    def test_get_unknown_token(self):
        """Test retrieving unknown token."""
        vocab = Vocabulary()
        assert vocab.get_token(999) == "<UNK>"
    
    def test_vocabulary_contains(self):
        """Test checking token existence."""
        vocab = Vocabulary()
        vocab.add_token("hello")
        assert "hello" in vocab
        assert "world" not in vocab
    
    def test_add_merge(self):
        """Test recording merge operations."""
        vocab = Vocabulary()
        merge = ("he", "llo")
        vocab.add_merge(merge)
        assert len(vocab.merges) == 1
        assert vocab.merges[0] == merge
    
    def test_vocabulary_save_load(self):
        """Test saving and loading vocabulary."""
        vocab1 = Vocabulary()
        vocab1.add_token("hello")
        vocab1.add_token("world", special=True)
        vocab1.add_merge(("he", "llo"))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            vocab1.save(filepath)
            vocab2 = Vocabulary()
            vocab2.load(filepath)
            
            assert vocab2.size() == 2
            assert "hello" in vocab2
            assert "world" in vocab2
            assert len(vocab2.merges) == 1
        finally:
            os.unlink(filepath)


class TestBPEMerger:
    """Test BPE merge operations."""
    
    def test_get_stats_basic(self):
        """Test basic pair frequency counting."""
        vocab = {
            ('h', 'e', 'l', 'l', 'o', '</w>'): 5,
            ('w', 'o', 'r', 'l', 'd', '</w>'): 3,
        }
        stats = BPEMerger.get_stats(vocab)
        assert stats[('h', 'e')] == 5
        assert stats[('l', 'l')] == 5 + 3  # Appears in both words
    
    def test_merge_vocab_basic(self):
        """Test merging a pair in vocabulary."""
        vocab = {
            ('h', 'e', 'l', 'l', 'o', '</w>'): 5,
        }
        merged = BPEMerger.merge_vocab(('h', 'e'), vocab)
        # Should contain merged 'he' instead of 'h' and 'e'
        assert merged[('he', 'l', 'l', 'o', '</w>')] == 5
    
    def test_encode_basic(self):
        """Test encoding a word with BPE."""
        merger = BPEMerger()
        merges = [('h', 'e'), ('he', 'l'), ('hel', 'l')]
        result = merger.encode('hello', merges)
        # Result should contain merged tokens
        assert '</w>' in result  # End marker should be present
        assert len(result) > 0
    
    def test_learn_merges(self):
        """Test learning BPE merges."""
        vocab = {
            ('h', 'e', 'l', 'l', 'o', '</w>'): 10,
            ('w', 'o', 'r', 'l', 'd', '</w>'): 5,
            ('t', 'e', 's', 't', '</w>'): 3,
        }
        merger = BPEMerger()
        merges = merger.learn_merges(vocab, num_merges=5)
        assert len(merges) <= 5
        assert all(isinstance(m, tuple) and len(m) == 2 for m in merges)


class TestBPETokenizer:
    """Test BPE tokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = BPETokenizer(vocab_size=100)
        assert tokenizer.vocab_size == 100
        assert tokenizer.vocab.size() >= 7  # At least special tokens
        assert len(tokenizer.merges) == 0
    
    def test_tokenizer_training_basic(self):
        """Test basic tokenizer training."""
        texts = [
            "hello world",
            "hello there",
            "world peace",
            "the quick brown fox",
            "the lazy dog",
        ]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts, vocab_size=50)
        
        assert tokenizer.vocab.size() > 0
        assert len(tokenizer.merges) > 0
    
    def test_tokenizer_encode_basic(self):
        """Test encoding text."""
        texts = [
            "hello world",
            "hello there",
            "world peace",
        ]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)
        
        tokens = tokenizer.encode("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0
    
    def test_tokenizer_encode_with_special_tokens(self):
        """Test encoding with special tokens."""
        texts = [
            "hello world",
            "hello there",
        ]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)
        
        tokens = tokenizer.encode("hello", add_special_tokens=True)
        # Should have BOS at start and EOS at end
        assert tokens[0] == tokenizer.vocab.get_id("<BOS>")
        assert tokens[-1] == tokenizer.vocab.get_id("<EOS>")
    
    def test_tokenizer_decode_basic(self):
        """Test decoding token IDs."""
        texts = [
            "hello world",
            "hello there",
        ]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)
        
        original = "hello world"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        # Decoded should be similar to original (may have spacing changes)
        assert len(decoded) > 0
        assert "hello" in decoded.lower()
    
    def test_tokenizer_roundtrip(self):
        """Test encode-decode roundtrip."""
        texts = [
            "the quick brown fox",
            "jumps over the lazy dog",
            "quick fox",
        ]
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(texts)
        
        original = "quick fox"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        
        # Check that decoded contains the important words
        decoded_lower = decoded.lower()
        assert "quick" in decoded_lower or "fox" in decoded_lower
    
    def test_tokenizer_unknown_words(self):
        """Test handling of unknown words."""
        texts = ["hello world"]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)
        
        # Encode text with unknown words
        tokens = tokenizer.encode("hello xyzabc world")
        assert len(tokens) > 0
    
    def test_tokenizer_vocab_size(self):
        """Test vocabulary size."""
        texts = [
            "hello world",
            "hello there",
            "world peace",
        ]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)
        
        vocab_size = tokenizer.get_vocab_size()
        assert vocab_size > 0
        assert vocab_size <= 100 + 50  # Some tolerance
    
    def test_tokenizer_save_load(self):
        """Test saving and loading tokenizer."""
        texts = [
            "hello world",
            "hello there",
        ]
        tokenizer1 = BPETokenizer(vocab_size=100)
        tokenizer1.train(texts)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tokenizer.json")
            tokenizer1.save(filepath)
            
            tokenizer2 = BPETokenizer()
            tokenizer2.load(filepath)
            
            # Compare vocabularies
            assert tokenizer2.vocab.size() == tokenizer1.vocab.size()
            assert len(tokenizer2.merges) == len(tokenizer1.merges)
    
    def test_tokenizer_consistency(self):
        """Test that tokenizer produces consistent results."""
        texts = [
            "hello world",
            "hello there",
        ]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)
        
        text = "hello world"
        tokens1 = tokenizer.encode(text)
        tokens2 = tokenizer.encode(text)
        
        assert tokens1 == tokens2
    
    def test_tokenizer_empty_text(self):
        """Test encoding empty text."""
        texts = ["hello world"]
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)
        
        tokens = tokenizer.encode("")
        assert isinstance(tokens, list)
    
    def test_tokenizer_unicode(self):
        """Test handling of unicode characters."""
        texts = [
            "hello world",
            "café société",
            "你好 世界",
        ]
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(texts)
        
        # Should handle unicode without crashing
        tokens = tokenizer.encode("café")
        assert isinstance(tokens, list)
    
    def test_tokenizer_repr(self):
        """Test tokenizer string representation."""
        tokenizer = BPETokenizer(vocab_size=100)
        repr_str = repr(tokenizer)
        assert "BPETokenizer" in repr_str
        assert "vocab_size" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
