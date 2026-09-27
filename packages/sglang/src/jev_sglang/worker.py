"""Import target for SGLang's spawned Uvicorn tokenizer workers.

Install the lifespan wrapper before entering the host lifespan. The host creates
that worker's TokenizerWorker during __aenter__; our wrapper then builds its own
runtime and canary. Parent process hooks alone are not inherited by spawn.
"""

from sglang.srt.entrypoints import http_server

from jev_sglang.plugin import configure_http_app

configure_http_app()
app = http_server.app
