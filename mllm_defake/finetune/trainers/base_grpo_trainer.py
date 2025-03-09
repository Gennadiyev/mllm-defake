from abc import abstractmethod
from collections import defaultdict

import torch
from accelerate.utils import is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from peft import PeftConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, PreTrainedModel, Trainer, TrainerCallback
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig


class BaseGRPOTrainer(Trainer):
    """The base class for GRPO trainers.

    Args:
        model (str): The model name or path.
        reward_cls (type): The reward class.
        reward_config (dict): The reward configuration for `RewardVx` initialization.
        args (GRPOConfig): The GRPO configuration. Defaults to None.
        train_dataset (Dataset | IterableDataset): The training dataset. Defaults to None.
        test_dataset (Dataset | IterableDataset, optional): The test dataset. Defaults to None.
        callbacks (list[TrainerCallback], optional): The list of callbacks. Defaults to None.
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).
        optimizers (tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional):
            The tuple of optimizer and scheduler. Defaults to (None, None). If None, the default optimizer
            is `AdamW` and the default scheduler is given by `get_linear_schedule_with_warmup` controlled by `args`.
        peft_config (PeftConfig, optional): The PEFT configuration. Defaults to None.
        freeze_vision (bool): Whether to freeze the vision model. Defaults to False.
        torch_dtype (str): The torch dtype. Defaults to "bfloat16".
    """

    def __init__(
        self,
        model: str,
        reward_cls: type,
        reward_config: dict,
        args: GRPOConfig = None,
        train_dataset: Dataset | IterableDataset = None,
        test_dataset: Dataset | IterableDataset | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
        freeze_vision: bool = False,
        torch_dtype: str = "bfloat16",
    ):
        if args is None:
            model_name = model.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_name_or_path = model
        # trained model
        model_init_kwargs = self._build_model_init_kwargs(args, torch_dtype)
        model, vision_modules_keywords, processor, pad_token_id = self._build_model(
            model_name_or_path, model_init_kwargs
        )
        model = self._post_process_model(model, peft_config, freeze_vision, vision_modules_keywords)
        # ref model
        self.ref_model = self._build_ref_model(model, model_name_or_path, model_init_kwargs, peft_config)
        # reward
        self.reward_function = reward_cls(**reward_config)
        # train arguments
        self.max_prompt_length = None
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        self.epsilon = args.epsilon
        # multi step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        # copied
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True
        # initialize the metrics
        self._metrics = defaultdict(list)
        # super
        super().__init__(
            model=model,
            args=args,
            data_collator=lambda x: x,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=processor,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # copied
        # check if the per_device_train/test_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )
        # copied
        # ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        # copied
        # gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _build_model_init_kwargs(self, args: GRPOConfig, torch_dtype: str) -> dict:
        model_init_kwargs = args.model_init_kwargs or {}
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        # process torch_dtype
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        return model_init_kwargs

    @abstractmethod
    def _build_model(
        self, model_name_or_path: str, model_init_kwargs: dict
    ) -> tuple[PreTrainedModel, list[str], AutoProcessor | AutoTokenizer, int]:
        """Build the model, specify the vision modules, and return the processor or tokenizer and pad token id."""
        raise NotImplementedError

    def _post_process_model(
        self,
        model: PreTrainedModel,
        args: GRPOConfig,
        peft_config: PeftConfig,
        freeze_vision: bool,
        vision_modules_keywords: list[str],
    ) -> PreTrainedModel:
        # peft
        if peft_config is not None:

            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)

            target_modules = find_all_linear_names(model, vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)
        # freeze vision
        if freeze_vision:
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in vision_modules_keywords):
                    p.requires_grad = False
        # gradient checkpointing
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enable gradient checkpointing for the model."""
        # ensure use_cache is disabled
        model.config.use_cache = False
        # enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _build_ref_model(
        self, model: PreTrainedModel, model_name_or_path: str, model_init_kwargs: dict, peft_config: PeftConfig
    ) -> PreTrainedModel:
        """Build the reference model."""
        if is_deepspeed_zero3_enabled():
            model_type = type(model)
            ref_model = model_type.from_pretrained(model_name_or_path, **model_init_kwargs)
        elif peft_config is None:
            # if PEFT configuration is not provided, create a reference model based on the initial model
            ref_model = create_reference_model(model)
        else:
            # if PEFT is used, the reference model is not needed since the adapter can be disabled
            ref_model = None
        return ref_model

    def compute_loss(self, model: PreTrainedModel, inputs: dict, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            # generate new
            inputs = self._generate_and_score_completions(model, inputs)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            # use buffered
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1
        # TODO
    
    def _generate_and_score_completions(self, model: PreTrainedModel, inputs: dict) -> dict:
        device = self.accelerator.device
        prompt_inputs = self._process_input(inputs)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = self._model_generate(unwrapped_model, prompt_inputs, generation_config=self.generation_config)
        
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # mask everything after the first eos_token_id
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        return inputs

    @abstractmethod
    def _process_input(self, inputs: dict):
        """Process the input."""
        raise NotImplementedError
    
    @abstractmethod
    def _model_generate(self, model: PreTrainedModel, inputs: dict, generation_config: GenerationConfig):
        """Generate completions using the model."""
        raise NotImplementedError
    
    @abstractmethod
    def _model_forward(self, model: PreTrainedModel, inputs: dict):
        """Forward pass using the model."""
        raise NotImplementedError