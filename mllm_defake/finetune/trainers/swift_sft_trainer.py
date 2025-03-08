from swift.llm.train import SwiftSft

from mllm_defake.finetune.utils import ADDED_TOKENS


class SwiftSFTTrainer(SwiftSft):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hack_tokenizer()
    
    def _hack_tokenizer(self):
        self.tokenizer.add_tokens(ADDED_TOKENS)


def sft_main(args = None):
    return SwiftSFTTrainer(args).main()


if __name__ == '__main__':
    sft_main()
