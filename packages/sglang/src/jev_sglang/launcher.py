import runpy

from jev_sglang.plugin import configure_http_app


def main():
    configure_http_app()
    runpy.run_module("sglang.launch_server", run_name="__main__")
