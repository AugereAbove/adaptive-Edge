# Security and privacy

The 0.1.x line is the current supported development line. There are no earlier supported releases or promised response times.

AdaptiveEdge has no telemetry, analytics or automatic uploads. Databases and JSON exports contain prompts, references, outputs and hardware/software metadata in plaintext. Review them before sharing. Backend errors can contain sensitive information. CSV exports omit response text but still disclose task/model identifiers and measurements.

The default Ollama endpoint is local. OLLAMA_HOST can redirect prompts to another host; its operator receives those prompts. Protect unauthenticated services from untrusted networks. Model loading and backend runtimes have independent supply-chain and resource risks. AdaptiveEdge never downloads models automatically. Client hardware cannot describe a remote server.

## Generated-code boundary

code_test_success launches a trusted worker with Python -I -S, a temporary working directory, a minimal environment and a two-second subprocess deadline. The worker parses generated text and interprets a bounded, capability-restricted AST subset. Generated Python is never passed to exec, eval, a shell or an interpreter as executable source. Exactly one named function containing one return expression is accepted. Imports, attributes, comprehensions, loops, assignment, recursion and arbitrary calls are unsupported. Pure calls are limited to abs/min/max/len; source, AST, fixture, numeric and string sizes are bounded.

This is a controlled expression evaluator, not a general Python sandbox or full programming benchmark. Unsupported/malformed programs score zero even if correct in unrestricted Python. The worker has the user's OS privileges; safety depends on preserving its small capability boundary. Do not add arbitrary execution or capabilities without separate review. Repository modification and secret access are not evaluator capabilities. A timeout or worker error scores zero and does not establish a solved fixture.

YAML uses safe loading, SQL values are parameterized and typed records reject unexpected fields. These safeguards and dependency scans are not a security guarantee.

Use the repository host's private security advisory feature **if the owner enables it after publication**. Do not disclose credentials/private prompts publicly. No maintainer email or private reporting endpoint was provided; the owner must configure one. Private reporting is not claimed to be already enabled.
