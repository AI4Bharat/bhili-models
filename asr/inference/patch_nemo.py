"""
Patch NeMo to support multilingual tokenizer.
Run this after installing NeMo.
"""

import os
import sys

def find_nemo_path():
    try:
        import nemo
        return os.path.dirname(nemo.__file__)
    except ImportError:
        if os.path.exists("NeMo/nemo"):
            return "NeMo/nemo"
        print("Error: NeMo not found. Install NeMo first.")
        sys.exit(1)

def patch_mixins(nemo_path):
    mixins_path = os.path.join(nemo_path, "collections/asr/parts/mixins/mixins.py")
    
    if not os.path.exists(mixins_path):
        print(f"Error: {mixins_path} not found")
        return False
    
    with open(mixins_path, "r") as f:
        content = f.read()
    
    if "multilingual" in content.lower():
        print("mixins.py already patched")
        return True
    
    old_setup = """    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tokenizer_type = tokenizer_cfg.get('type')
        if tokenizer_type is None:
            raise ValueError("`tokenizer.type` cannot be None")
        elif tokenizer_type.lower() == 'agg':
            self._setup_aggregate_tokenizer(tokenizer_cfg)
        else:
            self._setup_monolingual_tokenizer(tokenizer_cfg)"""
    
    new_setup = """    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tokenizer_type = tokenizer_cfg.get('type')
        if tokenizer_type is None:
            raise ValueError("`tokenizer.type` cannot be None")
        elif tokenizer_type.lower() == 'agg':
            self._setup_aggregate_tokenizer(tokenizer_cfg)
        elif tokenizer_type.lower() == 'multilingual':
            self._setup_aggregate_tokenizer(tokenizer_cfg)
        else:
            self._setup_monolingual_tokenizer(tokenizer_cfg)"""
    
    if old_setup in content:
        content = content.replace(old_setup, new_setup)
    else:
        print("Warning: Could not find exact _setup_tokenizer pattern")
        print("Attempting alternative patch...")
        if "elif tokenizer_type.lower() == 'agg':" in content:
            content = content.replace(
                "elif tokenizer_type.lower() == 'agg':\n            self._setup_aggregate_tokenizer(tokenizer_cfg)\n        else:",
                "elif tokenizer_type.lower() == 'agg':\n            self._setup_aggregate_tokenizer(tokenizer_cfg)\n        elif tokenizer_type.lower() == 'multilingual':\n            self._setup_aggregate_tokenizer(tokenizer_cfg)\n        else:"
            )
    
    old_agg = "self.tokenizer = tokenizers.AggregateTokenizer(tokenizers_dict)"
    new_agg = """if self.tokenizer_type == 'multilingual':
                self.tokenizer = tokenizers.MultilingualTokenizer(tokenizers_dict)
            else:
                self.tokenizer = tokenizers.AggregateTokenizer(tokenizers_dict)"""
    
    if old_agg in content and "MultilingualTokenizer" not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if old_agg in line and "if self.tokenizer_type" not in lines[i-1]:
                indent = len(line) - len(line.lstrip())
                new_lines = [
                    " " * indent + "if self.tokenizer_type == 'multilingual':",
                    " " * indent + "    self.tokenizer = tokenizers.MultilingualTokenizer(tokenizers_dict)",
                    " " * indent + "else:",
                    " " * indent + "    self.tokenizer = tokenizers.AggregateTokenizer(tokenizers_dict)"
                ]
                lines[i] = '\n'.join(new_lines)
                break
        content = '\n'.join(lines)
    
    with open(mixins_path, "w") as f:
        f.write(content)
    
    print(f"Patched: {mixins_path}")
    return True

def create_multilingual_tokenizer(nemo_path):
    tokenizers_path = os.path.join(nemo_path, "collections/common/tokenizers")
    ml_tokenizer_path = os.path.join(tokenizers_path, "multilingual_tokenizer.py")
    
    if os.path.exists(ml_tokenizer_path):
        print("multilingual_tokenizer.py already exists")
        return True
    
    content = '''"""
Multilingual Tokenizer for NeMo ASR.
"""

from typing import Dict, List, Optional, Union
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class MultilingualTokenizer(TokenizerSpec):
    """
    Multilingual tokenizer that wraps multiple language-specific tokenizers.
    """
    
    def __init__(self, tokenizers_dict: Dict[str, TokenizerSpec]):
        self.tokenizers_dict = tokenizers_dict
        self.langs = list(tokenizers_dict.keys())
        self.default_lang = self.langs[0] if self.langs else None
        
        if self.default_lang:
            self._tokenizer = tokenizers_dict[self.default_lang]
        else:
            self._tokenizer = None
    
    @property
    def vocab_size(self) -> int:
        if self._tokenizer:
            return self._tokenizer.vocab_size
        return 0
    
    def text_to_tokens(self, text: str, lang: Optional[str] = None) -> List[str]:
        tokenizer = self._get_tokenizer(lang)
        return tokenizer.text_to_tokens(text)
    
    def tokens_to_text(self, tokens: List[str], lang: Optional[str] = None) -> str:
        tokenizer = self._get_tokenizer(lang)
        return tokenizer.tokens_to_text(tokens)
    
    def text_to_ids(self, text: str, lang: Optional[str] = None) -> List[int]:
        tokenizer = self._get_tokenizer(lang)
        return tokenizer.text_to_ids(text)
    
    def ids_to_text(self, ids: List[int], lang: Optional[str] = None) -> str:
        tokenizer = self._get_tokenizer(lang)
        return tokenizer.ids_to_text(ids)
    
    def tokens_to_ids(self, tokens: List[str], lang: Optional[str] = None) -> List[int]:
        tokenizer = self._get_tokenizer(lang)
        return tokenizer.tokens_to_ids(tokens)
    
    def ids_to_tokens(self, ids: List[int], lang: Optional[str] = None) -> List[str]:
        tokenizer = self._get_tokenizer(lang)
        return tokenizer.ids_to_tokens(ids)
    
    def _get_tokenizer(self, lang: Optional[str] = None) -> TokenizerSpec:
        if lang and lang in self.tokenizers_dict:
            return self.tokenizers_dict[lang]
        return self._tokenizer
    
    def set_default_lang(self, lang: str):
        if lang in self.tokenizers_dict:
            self.default_lang = lang
            self._tokenizer = self.tokenizers_dict[lang]
    
    @property
    def pad_id(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, 'pad_id'):
            return self._tokenizer.pad_id
        return 0
    
    @property
    def bos_id(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, 'bos_id'):
            return self._tokenizer.bos_id
        return -1
    
    @property
    def eos_id(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, 'eos_id'):
            return self._tokenizer.eos_id
        return -1
    
    @property
    def unk_id(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, 'unk_id'):
            return self._tokenizer.unk_id
        return 0
'''
    
    with open(ml_tokenizer_path, "w") as f:
        f.write(content)
    
    print(f"Created: {ml_tokenizer_path}")
    
    init_path = os.path.join(tokenizers_path, "__init__.py")
    if os.path.exists(init_path):
        with open(init_path, "r") as f:
            init_content = f.read()
        
        if "MultilingualTokenizer" not in init_content:
            import_line = "from nemo.collections.common.tokenizers.multilingual_tokenizer import MultilingualTokenizer\n"
            
            if "from nemo.collections.common.tokenizers.aggregate_tokenizer" in init_content:
                init_content = init_content.replace(
                    "from nemo.collections.common.tokenizers.aggregate_tokenizer",
                    import_line + "from nemo.collections.common.tokenizers.aggregate_tokenizer"
                )
            else:
                init_content = import_line + init_content
            
            with open(init_path, "w") as f:
                f.write(init_content)
            
            print(f"Updated: {init_path}")
    
    return True

def main():
    print("Patching NeMo for multilingual tokenizer support...")
    print()
    
    nemo_path = find_nemo_path()
    print(f"NeMo path: {nemo_path}")
    print()
    
    success = True
    success = patch_mixins(nemo_path) and success
    success = create_multilingual_tokenizer(nemo_path) and success
    
    print()
    if success:
        print("Patch applied successfully!")
    else:
        print("Some patches failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()