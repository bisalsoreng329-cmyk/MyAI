"""Special Token Definitions for BPE Tokenizer

Defines special tokens used in tokenization and model processing.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SpecialTokens:
    """Container for special token definitions.
    
    Attributes:
        pad_token (str): Padding token
        unk_token (str): Unknown token
        bos_token (str): Beginning of sequence token
        eos_token (str): End of sequence token
        sep_token (str): Separator token
        cls_token (str): Classification token
        mask_token (str): Masking token
    """
    
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    sep_token: str = "<SEP>"
    cls_token: str = "<CLS>"
    mask_token: str = "<MASK>"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, str]: Special tokens mapping
        """
        return {
            'pad': self.pad_token,
            'unk': self.unk_token,
            'bos': self.bos_token,
            'eos': self.eos_token,
            'sep': self.sep_token,
            'cls': self.cls_token,
            'mask': self.mask_token,
        }
    
    def to_list(self) -> list:
        """Get list of all special tokens.
        
        Returns:
            list: Special tokens
        """
        return [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.cls_token,
            self.mask_token,
        ]
    
    def __repr__(self) -> str:
        return f"SpecialTokens(pad={self.pad_token}, unk={self.unk_token}, bos={self.bos_token})"
