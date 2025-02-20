import os
from pathlib import Path

def _sanitize_for_attr(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")

def _unsanitize_attr(attr: str) -> str:
    return attr.replace("_", ".")

class DotPath:
    def __init__(self, path: Path):
        self._path = path
        self._contents_map = None
    def __repr__(self):
        return f"<DotPath folder={self._path}>"
    def _scan(self):
        if self._contents_map is not None:
            return
        self._contents_map = {}
        if not self._path.is_dir():
            return
        for item in os.listdir(self._path):
            real_name = item
            item_path = self._path / real_name
            attr_name = _sanitize_for_attr(real_name)
            if item_path.is_dir():
                self._contents_map[attr_name] = ("dir", real_name)
            else:
                self._contents_map[attr_name] = ("file", real_name)
    def __getattr__(self, attr: str):
        self._scan()
        if not self._contents_map:
            raise AttributeError(f"No items in folder: {self._path}")
        entry = self._contents_map.get(attr)
        if not entry:
            raise AttributeError(f"'{attr}' not found under {self._path}.\nExisting items: {list(self._contents_map.keys())}")
        typ, real_name = entry
        real_path = self._path / real_name
        if typ == "dir":
            return DotPath(real_path)
        else:
            return str(real_path.resolve())

class DynamicPathResolver:
    def __init__(self, marker="README.md"):
        self.root = self._find_marker_root(marker)
        print(f"Project Root: {self.root}")
        self.path = DotPath(self.root)
    def _find_marker_root(self, marker: str) -> Path:
        current = Path.cwd()
        while True:
            if (current / marker).exists():
                return current
            if current.parent == current:
                print(f"Warning: Marker '{marker}' not found; using cwd: {Path.cwd()}")
                return Path.cwd()
            current = current.parent
    def __repr__(self):
        return f"<DynamicPathResolver root={self.root}>"
