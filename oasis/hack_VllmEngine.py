# patch.py
from swift.llm import VllmEngine as OrigVllmEngine
# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import inspect
import os
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import torch
from packaging import version
from transformers import GenerationConfig

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer
from swift.plugin import Metric
from swift.utils import get_logger
from swift.llm.infer.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, random_uuid)
from swift.llm.infer.infer_engine.infer_engine import InferEngine
from swift.llm.infer.infer_engine.patch import patch_auto_config, patch_auto_tokenizer
from swift.llm.infer.infer_engine.utils import AdapterRequest, InferStreamer

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '3600'
    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, LLMEngine
except Exception:
    raise

logger = get_logger()
dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}



class VllmEngine(OrigVllmEngine):

    def _prepare_generation_config(self, request_config: RequestConfig) -> SamplingParams:
        kwargs = {'max_tokens': request_config.max_tokens}
        if hasattr(request_config, 'min_tokens') and request_config.min_tokens is not None:
            kwargs['min_tokens'] = request_config.min_tokens
        if hasattr(request_config, 'top_logprobs') and request_config.top_logprobs is not None:
            kwargs['logprobs'] = request_config.top_logprobs
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
            new_value = getattr(request_config, key)
            if new_value is None:
                kwargs[key] = getattr(self.generation_config, key)
            else:
                kwargs[key] = new_value

        if request_config.logprobs:
            kwargs['logprobs'] = 1
            if request_config.top_logprobs is not None:
                kwargs['logprobs'] = max(1, request_config.top_logprobs)

        # TODO: beam search
        for key in ['n', 'best_of', 'frequency_penalty', 'presence_penalty', 'seed']:
            kwargs[key] = getattr(request_config, key)

        res = SamplingParams(**kwargs)
        res.top_logprobs = request_config.top_logprobs
        return res


    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        
        request_config = deepcopy(request_config or RequestConfig())
        if template is None:
            template = self.default_template
        
        template.set_mode('vllm')
        loop = asyncio.get_running_loop()
        with torch.inference_mode():
            try:
                inputs = await loop.run_in_executor(None, template.encode, infer_request, False, request_config)
            except:
                inputs = await loop.run_in_executor(None, template.encode, infer_request)
        
        if hasattr(request_config, 'encode_hack') and request_config.encode_hack:
            assert hasattr(request_config, 'hack_idx')
            inputs['input_ids'] = inputs['input_ids'][:request_config.hack_idx]

        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        kwargs = {
            'template': template,
            'inputs': inputs,
            'generation_config': generation_config,
            'adapter_request': adapter_request
        }
        if pre_infer_hook:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        else:
            return await self._infer_full_async(**kwargs)
