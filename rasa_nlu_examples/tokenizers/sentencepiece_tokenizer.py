import os
from typing import Any, Dict, List, Text

import sentencepiece as spm

from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data import Message


class SentencePieceTokenizer(Tokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # specifies the path to a custom SentencePiece model file
        "model_file": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the SentencePiece framework."""

        super().__init__(component_config)

        self.case_sensitive = self.component_config["case_sensitive"]
        model_file = self.component_config["model_file"]
        if model_file:
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"SentencePiece model {model_file} not found. Please check config."
                )
        self.model = spm.SentencePieceProcessor(model_file=model_file)

    def _tokenize(self, sentence: Text) -> Any:

        return self.model.encode(sentence, out_type=str)
    
    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenize the text using the SentencePiece model.
        SentencePiece adds a special char in front of (some) words and splits words into
        sub-words. To ensure the entity start and end values matches the token values,
        tokenize the text first using the whitespace tokenizer. If individual tokens
        are split up into multiple tokens, add this information to the
        respected tokens.
        """

        # perform whitespace tokenization
        tokens_in = super().tokenize(message, attribute)

        tokens_out = []

        for token in tokens_in:
            # use ConveRT model to tokenize the text
            split_token_strings = self._tokenize(token.text)

            # clean tokens (remove special chars and empty tokens)
            split_token_strings = self._clean_tokens(split_token_strings)

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        return tokens_out

@staticmethod
def _clean_tokens(tokens: List[bytes]) -> List[Text]:
    """Encode tokens and remove special char added by ConveRT."""

    tokens = [string.decode("utf-8").replace("_", "") for string in tokens]
    return [string for string in tokens if string]
