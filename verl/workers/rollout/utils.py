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
import asyncio
import ipaddress
import logging
import os
import socket

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_hf_max_position_embeddings(hf_config) -> int:
    """Robustly resolve ``max_position_embeddings`` from a HF model config.

    Newer composite/multimodal configs (e.g. ``Qwen3_5MoeConfig``,
    ``Qwen2_5_VLConfig``, ``Qwen3OmniMoeConfig``) do not expose
    ``max_position_embeddings`` at the top level; the value lives under a text
    sub-config. This helper tries several common locations before giving up.
    """
    candidates = [hf_config]

    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        candidates.append(text_config)

    get_text_config = getattr(hf_config, "get_text_config", None)
    if callable(get_text_config):
        try:
            candidates.append(get_text_config())
        except Exception:
            pass

    for cfg in candidates:
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    raise AttributeError(
        f"Cannot determine max_position_embeddings from hf_config of type "
        f"{type(hf_config).__name__}; please set rollout.max_model_len explicitly."
    )


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server.should_exit = True
            await server.serve()
            server_task = asyncio.create_task(server.main_loop())
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
