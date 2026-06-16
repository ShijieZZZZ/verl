# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_ids = None
        if self.dataset_config.get("use_raw_prompt_ids_for_generation", False):
            prompt_ids = self._coerce_prompt_ids(kwargs.get("raw_prompt_ids"))
        if prompt_ids is None:
            messages = list(kwargs["raw_prompt"])

            # 1. extract images and videos from messages
            multi_modal_data = await self.process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")

            # 2. apply chat template and tokenize
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
            )
        else:
            multi_modal_data = {}
            images = None
            videos = None

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output

    @staticmethod
    def _coerce_prompt_ids(raw_prompt_ids: Any) -> Optional[list[int]]:
        if raw_prompt_ids is None:
            return None

        if hasattr(raw_prompt_ids, "tolist"):
            raw_prompt_ids = raw_prompt_ids.tolist()

        if not isinstance(raw_prompt_ids, list | tuple):
            return None

        if len(raw_prompt_ids) == 1 and isinstance(raw_prompt_ids[0], list | tuple):
            raw_prompt_ids = raw_prompt_ids[0]

        prompt_ids: list[int] = []
        for token_id in raw_prompt_ids:
            if hasattr(token_id, "item"):
                token_id = token_id.item()
            if isinstance(token_id, bool):
                return None
            try:
                prompt_ids.append(int(token_id))
            except (TypeError, ValueError):
                return None

        return prompt_ids
