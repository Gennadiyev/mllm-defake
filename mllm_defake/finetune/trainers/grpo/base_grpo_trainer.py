import os
import textwrap
from abc import abstractmethod
from collections import defaultdict

import torch
import transformers
from accelerate.utils import is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from peft import PeftConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, PreTrainedModel, Trainer, TrainerCallback
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import generate_model_card

from mllm_defake.finetune.trainers.grpo.grpo_config import VLGRPOConfig
from mllm_defake.finetune.trainers.grpo.sampler import RepeatRandomSampler


class BaseGRPOTrainer(Trainer):
    """The base class for GRPO trainers.

    Args:
        model (str): The model name or path.
        reward_cls (type): The reward class.
        reward_config (dict): The reward configuration for `RewardVx` initialization.
        args (VLGRPOConfig): The GRPO configuration. Defaults to None.
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
        args: VLGRPOConfig = None,
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
            args = VLGRPOConfig(f"{model_name}-GRPO")

        model_name_or_path = model
        # trained model
        model_init_kwargs = self._build_model_init_kwargs(args, torch_dtype)
        model, vision_modules_keywords, processor, pad_token_id, model_class = self._build_model(
            model_name_or_path, model_init_kwargs
        )
        self.model_class = model_class
        model = self._post_process_model(model, args, peft_config, freeze_vision, vision_modules_keywords)
        # ref model
        self.ref_model = self._build_ref_model(model, model_name_or_path, model_init_kwargs, peft_config)
        # reward
        self.reward_function = reward_cls(**reward_config)
        if not hasattr(self.reward_function, "num_functions"):
            raise ValueError("The reward class must have an attribute `num_functions`.")
        if not hasattr(self.reward_function, "reward_names"):
            raise ValueError("The reward class must have an attribute `reward_names`.")
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

    def _build_model_init_kwargs(self, args: VLGRPOConfig, torch_dtype: str) -> dict:
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
    ) -> tuple[PreTrainedModel, list[str], AutoProcessor | AutoTokenizer, int, type]:
        """Build the model, specify the vision modules, and return the processor or tokenizer, pad token id and the class for building model"""
        raise NotImplementedError

    def _post_process_model(
        self,
        model: PreTrainedModel,
        args: VLGRPOConfig,
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
        return model

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: VLGRPOConfig) -> PreTrainedModel:
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
            model_init_kwargs["trust_remote_code"] = True
            ref_model = self.model_class.from_pretrained(model_name_or_path, **model_init_kwargs)
        elif peft_config is None:
            # if PEFT configuration is not provided, create a reference model based on the initial model
            ref_model = create_reference_model(model)
        else:
            # if PEFT is used, the reference model is not needed since the adapter can be disabled
            ref_model = None
        return ref_model

    def compute_loss(self, model: PreTrainedModel, inputs: list[dict], return_outputs=False, num_items_in_batch=None):
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
        # get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        others = inputs["others"]
        # concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # get the current policy's log probabilities
        others.update({"input_ids": input_ids, "attention_mask": attention_mask})
        per_token_logps = self._get_per_token_logps(model, others)
        # get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1 :]

        # get the advantages
        advantages = inputs["advantages"]

        # when using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def _generate_and_score_completions(self, model: PreTrainedModel, inputs: list[dict]) -> dict:
        device = self.accelerator.device
        user_inputs = [x["user_input"] for x in inputs]
        assistant_outputs = [x["assistant_output"] for x in inputs]
        assert len(user_inputs) == len(assistant_outputs)
        prompt_inputs = self._process_input(inputs)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

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

        # compute logits
        logit_inputs = {k: v for k, v in prompt_inputs.items()}
        logit_inputs["input_ids"] = prompt_completion_ids
        logit_inputs["attention_mask"] = attention_mask

        with torch.inference_mode():
            # when using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(model, logit_inputs)
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1 :]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, logit_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, logit_inputs)
                    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # compute rewards
        rewards = torch.zeros(len(user_inputs), device=device)
        rewards_per_function = torch.zeros(len(user_inputs), self.reward_function.num_functions, device=device)
        for i, (user_input, assistant_output, completion) in enumerate(
            zip(user_inputs, assistant_outputs, completions, strict=False)
        ):
            reward = self.reward_function(user_input, assistant_output, completion)
            rewards[i] = torch.tensor(reward["all_reward"], dtype=torch.float32, device=device)
            rewards_per_function[i] = torch.tensor(reward["per_function_reward"], dtype=torch.float32, device=device)
        # gather the outputs
        rewards = self.accelerator.gather(rewards)

        # compute group-wise rewards
        # each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(user_inputs),
            (self.accelerator.process_index + 1) * len(user_inputs),
        )
        advantages = advantages[process_slice]

        # log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        rewards_per_function = self.accelerator.gather_for_metrics(rewards_per_function).mean(0)
        for i, reward_name in enumerate(self.reward_function.reward_names):
            self._metrics[reward_name].append(rewards_per_function[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "others": dict(),
        }
        for key in prompt_inputs.keys():
            if key == "input_ids" or key == "attention_mask":
                continue
            result["others"][key] = prompt_inputs[key]
        return result

    @abstractmethod
    def _process_input(self, inputs: list[dict]) -> dict:
        """Process the input."""
        raise NotImplementedError

    def _get_per_token_logps(self, model: PreTrainedModel, logit_inputs: dict):
        logits = model(**logit_inputs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = logit_inputs["input_ids"][
            :, 1:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids, strict=False):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # copied
    def create_model_card(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        tags: str | list[str] | None = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=None,
            comet_url=None,
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def _get_train_sampler(self):
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
