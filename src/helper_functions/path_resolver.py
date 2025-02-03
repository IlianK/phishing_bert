import os
import types

class DynamicPathResolver:
    def __init__(self, root=None, marker=None):
        self.root = root or self._detect_project_root(marker)
        print(f"Project Root: {self.root}")
        self._generate_structure()

    def _detect_project_root(self, marker=None):
        current_dir = os.getcwd()
        marker = marker or ".git"
        while current_dir != os.path.dirname(current_dir):  
            if marker in os.listdir(current_dir): 
                return current_dir
            current_dir = os.path.dirname(current_dir)
        print(f"Warning: Marker '{marker}' not found. Using current working directory as root.")
        return os.getcwd()

    def _generate_structure(self):
        def create_subtree(path):
            subtree = types.SimpleNamespace()
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                key = self._sanitize_name(item)
                if os.path.isdir(item_path):
                    setattr(subtree, key, create_subtree(item_path))
                else:
                    setattr(subtree, key, item_path)
            return subtree
        self.structure = create_subtree(self.root)

    def _sanitize_name(self, name):
        return name.replace(".", "_").replace("-", "_")

    def __getattr__(self, item):
        current = getattr(self.structure, item, None)
        if current is None:
            raise AttributeError(f"'DynamicPathResolver' object has no attribute '{item}'")
        if isinstance(current, types.SimpleNamespace):
            return self._get_directory_path(current)
        return current

    def _get_directory_path(self, subtree):
        for key, value in subtree.__dict__.items():
            if isinstance(value, str) and os.path.isdir(value):
                return value
            elif isinstance(value, types.SimpleNamespace):
                return self._get_directory_path(value)
        return self.root

    def get_folder_path_from_namespace(self, namespace_obj):
        if isinstance(namespace_obj, types.SimpleNamespace):
            for key, value in namespace_obj.__dict__.items():
                if isinstance(value, str) and os.path.isfile(value):
                    return os.path.dirname(value)
        return None

def read_wordbag(file_path):
    try:
        with open(file_path, 'r') as file:
            wordbag = [line.strip().lower() for line in file.readlines()]
        return wordbag
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []