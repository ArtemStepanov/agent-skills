"""Smoke test for the minimal Hermes adapter."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agent_skills_plugin", ROOT / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(plugin)


class Context:
    def __init__(self):
        self.skills = []

    def register_skill(self, name, path):
        self.skills.append((name, Path(path)))


ctx = Context()
plugin.register(ctx)
assert ctx.skills == [("tuning-local-llms", ROOT / "skills/tuning-local-llms/SKILL.md")]

hermes_manifest = (ROOT / "plugin.yaml").read_text()
claude_plugin = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
marketplace = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
assert "name: agent-skills\n" in hermes_manifest
assert "version: 0.1.0\n" in hermes_manifest
assert "  - tuning-local-llms\n" in hermes_manifest
assert claude_plugin["name"] == marketplace["name"] == "agent-skills"
assert claude_plugin["version"] == "0.1.0"
