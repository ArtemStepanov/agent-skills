"""Hermes plugin for the bundled Agent Skills."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"


def register(ctx):
    """Register every bundled skill with Hermes."""
    for skill in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        ctx.register_skill(skill.parent.name, skill)
