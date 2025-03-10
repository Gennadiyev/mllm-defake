from swift.llm.train import SwiftSft


ADDED_TOKENS = ["<think>", "</think>", "<verdict>", "</verdict>", "<tag>", "</tag>"]


class SwiftSFTTrainer(SwiftSft):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hack_tokenizer()

    def _hack_tokenizer(self):
        self.tokenizer.add_tokens(ADDED_TOKENS)
