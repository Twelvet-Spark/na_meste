"""Fixed OWLv2 prompts and their exported query-vector identity (stdlib only)."""
import hashlib
import json

VOCABULARY: tuple[tuple[str, str, str], ...] = (
    ("a door", "DOOR", "Дверь"),
    ("an open doorway", "DOOR", "Дверной проём"),
    ("a glass door", "DOOR", "Стеклянная дверь"),
    ("a staircase", "PASSAGE", "Лестница"),
    ("a corridor", "PASSAGE", "Коридор"),
    ("a desk", "OBJECT", "Стол"),
    ("a table", "OBJECT", "Стол"),
    ("a chair", "OBJECT", "Стул"),
    ("an office chair", "OBJECT", "Офисное кресло"),
    ("a sofa", "OBJECT", "Диван"),
    ("a cabinet", "OBJECT", "Шкаф"),
    ("a shelf", "OBJECT", "Стеллаж"),
    ("a computer monitor", "OBJECT", "Монитор"),
    ("a reception counter", "OBJECT", "Стойка"),
    ("a refrigerator", "OBJECT", "Холодильник"),
    ("a potted plant", "OBJECT", "Растение"),
    ("a printer", "OBJECT", "Принтер"),
    ("a whiteboard", "OBJECT", "Доска"),
    ("a window", "OBJECT", "Окно"),
    ("a person", "PERSON", "Человек"),
)
VOCABULARY_ID = hashlib.sha256(json.dumps([prompt for prompt, _, _ in VOCABULARY]).encode()).hexdigest()
