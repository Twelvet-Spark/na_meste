"""Which crossing directions an attendance camera counts; free of heavy imports."""

ROLE_DIRECTIONS = {"ENTRY": ("ENTRY",), "EXIT": ("EXIT",), "BOTH": ("ENTRY", "EXIT")}


def covered_directions(roles) -> frozenset[str]:
    return frozenset(direction for role in roles for direction in ROLE_DIRECTIONS[role])
