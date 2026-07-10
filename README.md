# Agent skills

A small collection of reusable coding-agent skills.

## Skills

- [`tuning-local-llms`](skills/tuning-local-llms/): benchmarks and tunes GGUF / llama.cpp-family local LLM configurations against the actual hardware.

## Install

Pin a release tag when installing. `main` is for development.

### Pi

```sh
pi install git:github.com/ArtemStepanov/agent-skills@v0.1.0
```

### Hermes Agent

```sh
hermes plugins install ArtemStepanov/agent-skills --enable
```

### Claude Code

```text
/plugin marketplace add ArtemStepanov/agent-skills
/plugin install agent-skills@agent-skills
```

### Any Agent Skills-compatible harness

```sh
git clone --branch v0.1.0 https://github.com/ArtemStepanov/agent-skills.git \
  ~/.agents/skills/agent-skills
```

Each skill is self-contained. Read its `SKILL.md`; preserve its bundled license and supporting files when redistributing it.

## Releases

Releases are version tags. Upgrade by selecting a newer tag explicitly; this repo does not auto-update installed skills.

## Security

Do not commit API keys, private model paths, benchmark logs containing private prompts, or model weights. See [SECURITY.md](SECURITY.md).
