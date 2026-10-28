"""A climate table's JSON as the repository writes it (match.py, world.py, tools/biome_lab): no numpy, so any tool can
import it."""
import json

# The world table world.py composes from the landform tables (relative to the repository root).
WORLD_TABLE = "adapter/godot/project/climate/world.json"


def dump_table(v, indent: int = 0) -> str:
    """JSON with one line per object or list that holds no object (a band, a multiplier, a curve point)."""
    flat = json.dumps(v)
    if not isinstance(v, (dict, list)) or len(flat) <= 110 and "{" not in flat[1:]:
        return flat
    pad = " " * (indent + 1)
    if isinstance(v, dict):
        items = [f"{pad}{json.dumps(k)}: {dump_table(x, indent + 1)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if all(not isinstance(x, (dict, list)) for x in v) or len(flat) <= 110 and all(isinstance(x, list) for x in v):
        return flat
    return "[\n" + ",\n".join(pad + dump_table(x, indent + 1) for x in v) + "\n" + " " * indent + "]"
