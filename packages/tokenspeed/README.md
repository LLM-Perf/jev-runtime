# Jev Runtime for TokenSpeed

Experimental TokenSpeed adapter for
[Jev Runtime](https://github.com/LLM-Perf/jev-runtime), which turns model scores
into versioned typed decisions with probabilities.

```bash
pip install jev-tokenspeed
```

TokenSpeed support is experimental and is not yet GPU-certified. Read the
[TokenSpeed quick start](https://github.com/LLM-Perf/jev-runtime/blob/main/docs/quickstart-tokenspeed.md)
before deployment.

Includes source/hardware preflight, native Engine loop dispatch, durable completion
receipts and the shared typed-decision/bundle APIs. K labels currently require K
native requests; efficient joint readout and GPU certification remain open.
