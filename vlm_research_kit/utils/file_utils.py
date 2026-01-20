"""Utility functions for common file operations like loading/saving
JSON, JSONL, Pickle, and YAML files. Supports both string paths and
pathlib.Path objects."""

import json
import os
import pickle
import yaml
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

from vlm_research_kit.settings import EXPERIMENTS_DIR, PROJECT_ROOT, CXRFESCORE_CACHE_DIR

# Define a type alias for paths for cleaner type hints
PathLike = Union[str, Path]

def get_timestamp() -> str:
    """
    Returns the current timestamp as a formatted string.
    Format: YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_dirs_in_filepath(filepath: PathLike) -> None:
    """
    Creates all necessary parent directories for a given file path if they
    do not already exist.

    Args:
        filepath: The full path (string or pathlib.Path) to the file whose
                  directories need creating.
    """
    # Convert to Path object for consistent handling
    filepath_obj = Path(filepath)
    parent_dir = filepath_obj.parent
    # Only create directories if parent_dir is not the current directory '.'
    # or empty (though Path.parent is unlikely to be empty)
    # os.makedirs handles exist_ok=True correctly even for '.'
    if parent_dir and str(parent_dir) != ".":
        # os.makedirs accepts Path objects since Python 3.5
        os.makedirs(parent_dir, exist_ok=True)


def load_pickle(path: PathLike) -> Any:
    """
    Loads a Python object from a pickle file.

    Args:
        path: The path (string or pathlib.Path) to the pickle file.

    Returns:
        The Python object loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If the file is corrupted or not a valid pickle.
        EOFError: If the file is empty or truncated.
        Exception: Other file system related errors.
    """
    # open() accepts Path objects since Python 3.6
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: PathLike) -> None:
    """
    Saves a Python object to a pickle file. Creates parent directories
    if they don't exist.

    Args:
        obj: The Python object to save.
        path: The path (string or pathlib.Path) where the pickle file
              will be saved.

    Raises:
        pickle.PicklingError: If the object cannot be pickled.
        Exception: File system related errors (e.g., permissions).
    """
    make_dirs_in_filepath(path)
    # open() accepts Path objects since Python 3.6
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_json(path: PathLike) -> Any:
    """
    Loads data from a JSON file.

    Args:
        path: The path (string or pathlib.Path) to the JSON file.

    Returns:
        The Python object (dict, list, etc.) loaded from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        Exception: Other file system related errors.
    """
    # open() accepts Path objects since Python 3.6
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    obj: Any,
    path: PathLike,
    indent: Optional[int] = None,
    ensure_ascii: bool = True,
) -> None:
    """
    Saves a Python object to a JSON file. Creates parent directories
    if they don't exist.

    Args:
        obj: The Python object to save (must be JSON serializable).
        path: The path (string or pathlib.Path) where the JSON file
              will be saved.
        indent: If not None, specifies the indentation level for pretty-printing.
        ensure_ascii: If True (default), non-ASCII characters are escaped.
                      If False, they are written as-is (requires UTF-8).

    Raises:
        TypeError: If the object is not JSON serializable.
        Exception: File system related errors (e.g., permissions).
    """
    make_dirs_in_filepath(path)
    # open() accepts Path objects since Python 3.6
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def load_jsonl(path: PathLike) -> List[Any]:
    """
    Loads data from a JSON Lines (.jsonl) file. Each line should be a valid
    JSON object.

    Args:
        path: The path (string or pathlib.Path) to the JSON Lines file.

    Returns:
        A list of Python objects loaded from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a directory.
        ValueError: If the file does not end with '.jsonl'.
        json.JSONDecodeError: If any line contains invalid JSON.
        Exception: Other file system related errors.
    """
    # Convert to Path object for consistent checks
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"No such file or directory: '{path}'")
    if not path_obj.is_file():
        raise IsADirectoryError(f"Expected a file, but got a directory: '{path}'")
    # Use Path.suffix for extension checking
    if path_obj.suffix.lower() != ".jsonl":
        raise ValueError(f"File extension must be .jsonl: '{path}'")

    data = []
    # open() accepts Path objects since Python 3.6
    with open(path_obj, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Strip potential trailing whitespace/newlines before decoding
                stripped_line = line.strip()
                if stripped_line: # Avoid errors on empty lines
                    data.append(json.loads(stripped_line))
            except json.JSONDecodeError as e:
                # Add line number context to the error
                raise json.JSONDecodeError(
                    f"Error decoding JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
    return data


def save_jsonl(
    obj_iterable: Iterable[Any], path: PathLike, append: bool = False
) -> None:
    """
    Saves an iterable of Python objects to a JSON Lines (.jsonl) file.
    Each object is serialized as JSON on its own line. Creates parent
    directories if they don't exist.

    Args:
        obj_iterable: An iterable (e.g., list, tuple, generator) of Python
                      objects to save (each must be JSON serializable).
        path: The path (string or pathlib.Path) where the JSON Lines file
              will be saved.
        append: If True, append to the file if it exists. Otherwise, overwrite.

    Raises:
        TypeError: If any object in the iterable is not JSON serializable.
        Exception: File system related errors (e.g., permissions).
    """
    make_dirs_in_filepath(path)
    mode = "a" if append else "w"
    # open() accepts Path objects since Python 3.6
    with open(path, mode, encoding="utf-8") as f:
        for obj in obj_iterable:
            json_string = json.dumps(obj, ensure_ascii=False) # Often preferred for jsonl
            f.write(json_string + "\n")


def load_yaml(path: PathLike) -> Any:
    """
    Loads data from a YAML file using safe_load. Requires PyYAML to be installed.

    Args:
        path: The path (string or pathlib.Path) to the YAML file.

    Returns:
        The Python object loaded from the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        Exception: Other file system related errors.
    """
    # open() accepts Path objects since Python 3.6
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)    

def load_config_yaml(path: PathLike) -> Dict[str, Any]:
    # Check if the path is relative
    if not os.path.isabs(path):
        # Convert to absolute path using PROJECT_ROOT
        path = os.path.join(PROJECT_ROOT, path)
    # Now load the YAML file
    return load_yaml(path)

def save_yaml(
    obj: Any,
    path: PathLike,
    default_flow_style: Optional[bool] = None,
    sort_keys: bool = False,
) -> None:
    """
    Saves a Python object to a YAML file using safe_dump. Creates parent
    directories if they don't exist. Requires PyYAML to be installed.

    Args:
        obj: The Python object to save.
        path: The path (string or pathlib.Path) where the YAML file
              will be saved.
        default_flow_style: Controls the output style (block vs flow).
                            None lets PyYAML decide, False prefers block style.
        sort_keys: If True, sort dictionary keys alphabetically in the output.

    Raises:
        yaml.YAMLError: If the object cannot be serialized to YAML.
        Exception: File system related errors (e.g., permissions).
    """
    make_dirs_in_filepath(path)
    # open() accepts Path objects since Python 3.6
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            allow_unicode=True,  # Generally preferred with UTF-8 encoding
        )

def read_txt(path: PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Define standard ignore list globally or pass it in
DEFAULT_IGNORE_LIST = ['.git', '__pycache__', '.DS_Store', '.ipynb_checkpoints']

def print_directory_tree(
    start_path: Union[str, Path],
    level: int = -1, # Internal tracker for recursion depth
    prefix: str = '',
    ignore_list: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    include_files: bool = True,
    print_root: bool = True,
    max_files_per_extension: Optional[int] = None, # New argument
):
    """
    Prints a directory tree structure similar to the 'tree' command.

    Args:
        start_path (Union[str, Path]): The directory path to start from.
        level (int): Current recursion depth (internal use).
        prefix (str): String prefix for lines (internal use).
        ignore_list (Optional[List[str]]): List of directory/file names to ignore.
                                            Defaults to DEFAULT_IGNORE_LIST.
        max_depth (Optional[int]): Maximum depth to traverse. None means infinite.
        include_files (bool): Whether to include files in the output. Defaults to True.
        print_root (bool): Whether to print the root directory name. Defaults to True.
        max_files_per_extension (Optional[int]): Max files of the same extension to print.
                                                 None means no limit. If 0, all files
                                                 of that extension are omitted and a
                                                 message is shown.
    """
    if ignore_list is None:
        effective_ignore_list = list(DEFAULT_IGNORE_LIST)
    else:
        effective_ignore_list = list(ignore_list)

    current_path_obj = Path(start_path)
    if not current_path_obj.is_dir():
        print(f"Error: '{current_path_obj}' is not a valid directory.")
        return

    current_recursion_level = 0
    if level == -1: # First call
        if print_root:
            print(f"{current_path_obj.name}/")
        # current_recursion_level is already 0 for the first set of children
    else:
        current_recursion_level = level

    if max_depth is not None and current_recursion_level >= max_depth:
        return

    try:
        all_items_in_dir = list(current_path_obj.iterdir())
    except PermissionError:
        print(f"{prefix}├── [Permission Denied]")
        return
    except OSError as e:
        print(f"{prefix}├── [Error Reading Directory: {e}]")
        return

    unignored_items = [
        item
        for item in all_items_in_dir
        if item.name not in effective_ignore_list
    ]

    items_to_render_as_paths = []
    omission_message_strings = []

    if include_files and max_files_per_extension is not None:
        potential_dirs = []
        potential_files = []
        for item in unignored_items:
            if item.is_dir():
                potential_dirs.append(item)
            elif item.is_file():
                potential_files.append(item)
        
        potential_files.sort(key=lambda x: x.name)

        files_by_ext = defaultdict(list)
        for f_item in potential_files:
            ext = f_item.suffix.lower() # e.g., ".txt", ".tar.gz" -> ".gz", "" for no ext
            files_by_ext[ext].append(f_item)

        allowed_files_for_display = []
        files_omitted_counts_by_ext = defaultdict(int)

        for ext_key, files_in_ext_group in sorted(files_by_ext.items()):
            # files_in_ext_group is already sorted by name
            # Handle max_files_per_extension = 0 correctly (take none)
            num_to_take = max(0, max_files_per_extension)
            allowed_files_for_display.extend(files_in_ext_group[:num_to_take])
            
            omitted_count = len(files_in_ext_group) - num_to_take
            if omitted_count > 0:
                files_omitted_counts_by_ext[ext_key] = omitted_count
        
        items_to_render_as_paths.extend(allowed_files_for_display)
        items_to_render_as_paths.extend(potential_dirs)
        items_to_render_as_paths.sort(key=lambda x: x.name)

        for ext_key, count in sorted(files_omitted_counts_by_ext.items()):
            ext_display_name = ext_key if ext_key else "files with no extension"
            message = f"[... {count} other {ext_display_name} omitted]"
            omission_message_strings.append(message)
            
    else: # No file limit per extension, or not including files
        for item in unignored_items:
            if item.is_dir():
                items_to_render_as_paths.append(item)
            elif include_files and item.is_file():
                items_to_render_as_paths.append(item)
        items_to_render_as_paths.sort(key=lambda x: x.name)

    final_display_list_for_level = items_to_render_as_paths + omission_message_strings
    
    if not final_display_list_for_level:
        return

    pointers = ['├── '] * (len(final_display_list_for_level) - 1) + ['└── ']

    for pointer_char, display_entry_item in zip(pointers, final_display_list_for_level):
        if isinstance(display_entry_item, Path):
            path_object_to_print = display_entry_item
            if path_object_to_print.is_dir():
                print(f"{prefix}{pointer_char}{path_object_to_print.name}/")
                child_prefix_extension = '│   ' if pointer_char == '├── ' else '    '
                print_directory_tree(
                    start_path=path_object_to_print,
                    level=current_recursion_level + 1,
                    prefix=prefix + child_prefix_extension,
                    ignore_list=effective_ignore_list,
                    max_depth=max_depth,
                    include_files=include_files,
                    print_root=False,
                    max_files_per_extension=max_files_per_extension
                )
            else: # File Path object
                print(f"{prefix}{pointer_char}{path_object_to_print.name}")
        else: # Omission message string
            print(f"{prefix}{pointer_char}{display_entry_item}")


def list_filepaths_with_prefix_and_timestamps(path_prefix: str, must_contain: Optional[Union[str, List[str]]] = None) -> List[Tuple[str, str]]:
    """
    Lists filepaths with a prefix and timestamps.
    Args:
        path_prefix: The prefix of the filepaths to list.
        must_contain: The string or list of strings that must be contained in the filepaths.
    Returns:
        A list of tuples containing the filepath and timestamp.
    """
    if isinstance(must_contain, str):
        must_contain = [must_contain]
    matching_files = []
    directory = os.path.dirname(path_prefix)
    for root, _, files in os.walk(directory):
        for filename in files:
            full_path = os.path.join(root, filename)
            if full_path.startswith(path_prefix):
                if must_contain is not None:
                    if not all(s in full_path for s in must_contain):
                        continue
                creation_timestamp = os.path.getctime(full_path)
                timestamp_human_readable = datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                matching_files.append((full_path, timestamp_human_readable))
    matching_files.sort(key=lambda x:x[1], reverse=True)
    return matching_files


def get_safe_filename(model_name_or_path: str) -> str:
    """
    Converts a model name or path to a safe filename by replacing
    non-alphanumeric characters (except '-', '_', '.') with underscores.

    Args:
        model_name_or_path (str): The model name or path to convert.

    Returns:
        str: A safe filename derived from the input.
    """
    # Replace non-alphanumeric characters (except '-', '_', '.') with underscores
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in model_name_or_path)
    

def _get_nested_value(config: Dict, keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
    """
    Helper function to safely access nested dictionary values.
    Args:
        config: The dictionary to search.
        keys: A list of keys representing the path to the desired value.
        default: The default value to return if the key is not found.
    Returns:
        The value at the specified path, or the default value if not found.
    """
    value = config
    try:
        for key in keys:        
            value = value[key]
    except KeyError:
        return default
    return value

def get_eval_folder_name(eval_config: Dict[str, Any]) -> str:
    """
    Generates a descriptive folder name for an evaluation run based on its config.

    Args:
        eval_config: The dictionary containing evaluation configuration.
                     Expected keys (examples):
                     - dataset.name (e.g., "padchest")
                     - dataset.split (e.g., "test")
                     - task.name (e.g., "report_generation")
                     - evaluation.run_tag (optional, e.g., "beam5_run")
                     - evaluation.decoding.method (optional, e.g., "beam_search")
                     - evaluation.decoding.num_beams (optional, if method is beam)

    Returns:
        A sanitized, descriptive string suitable for use as a folder name.
    """
    parts = []

    # --- Essential Info ---
    dataset_name = _get_nested_value(eval_config, ['dataset', 'name'])
    dataset_split = _get_nested_value(eval_config, ['dataset', 'split'])
    task_name = _get_nested_value(eval_config, ['task', 'name'])
    assert dataset_name, "Dataset name is required."
    assert dataset_split, "Dataset split is required."
    assert task_name, "Task name is required."

    parts.append(get_safe_filename(dataset_name))
    parts.append(get_safe_filename(task_name))
    parts.append(get_safe_filename(dataset_split))

    # --- Optional Info ---
    run_tag = _get_nested_value(eval_config, ['run_tag'])
    if run_tag:
        parts.append(get_safe_filename(run_tag))

    decoding_method = _get_nested_value(eval_config, ['decoding', 'method'])
    if decoding_method:
        decoding_part = get_safe_filename(decoding_method)
        if decoding_method.lower() == "beam_search":
            num_beams = _get_nested_value(eval_config, ['decoding', 'num_beams'])
            if num_beams:
                decoding_part += f"_k{num_beams}"
        # Add conditions for other methods like nucleus sampling (e.g., _p0.9) if needed
        parts.append(decoding_part)

    # Add more optional parts based on your specific config structure
    # e.g., metric sets, specific filters

    # --- Timestamp for Uniqueness ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)

    # --- Join Parts ---
    # Filter out any potential None or empty strings just in case
    valid_parts = [part for part in parts if part]
    folder_name = "_".join(valid_parts)

    # Optional: Limit overall length if necessary, though underscores help
    # max_len = 100
    # if len(folder_name) > max_len:
    #     folder_name = folder_name[:max_len] + "..." # Or use hashing

    return folder_name

def setup_experiment_dir(experiment_dir: Optional[str] = None,
                         run_name: Optional[str] = None,
                         base_dir: str = EXPERIMENTS_DIR) -> Path:
    """
    Determines and creates the experiment directory.

    Args:
        experiment_dir (str or None): User-specified experiment directory.
        run_name (str or None): Name of the run/experiment.
        base_dir (str): Base directory for experiments.

    Returns:
        str: Path to the experiment directory.
    """
    if experiment_dir is None:
        if run_name is None:
            raise ValueError("run_name must be provided if experiment_dir is None.")
        experiment_dir = os.path.join(base_dir, run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return Path(experiment_dir)


def get_file_size(size_in_bytes):
    """Formats file size into KB, MB, etc."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} Bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_in_bytes / 1024**3:.2f} GB"


def inspect_single_cache_file(file_path: str, name: str):
    """Helper to inspect one cache file."""
    print("-" * 50)
    print(f"Inspecting: {name}")
    print(f"Path: {file_path}")
    
    if not os.path.exists(file_path):
        print("Status: File does not exist.")
        return

    try:
        # Get file metadata
        file_stat = os.stat(file_path)
        print(f"File Size: {get_file_size(file_stat.st_size)}")
        mod_time = datetime.fromtimestamp(file_stat.st_mtime)
        print(f"Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Try to load and inspect the content
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"Number of Entries: {len(data)}")

        if isinstance(data, dict) and data:
            # Show a few sample entries to verify content
            print("Sample Entries (first 3):")
            for i, (key, value) in enumerate(data.items()):
                if i >= 3:
                    break
                # Truncate long values for cleaner display
                value_repr = repr(value)
                if len(value_repr) > 100:
                    value_repr = value_repr[:100] + "..."
                print(f"  - Key: '{key}'")
                print(f"    Value: {value_repr}")
        
    except FileNotFoundError:
        print("Status: File not found (race condition, was deleted).")
    except (pickle.UnpicklingError, EOFError):
        print("\nERROR: Could not load pickle file. It may be CORRUPTED.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")


def inspect_cxrfescore_cache(cache_dir: str = CXRFESCORE_CACHE_DIR):
    """
    Loads and prints the state of CXRFEScore cache files.
    
    Args:
        cache_dir: The directory where the cache files are stored.
    """
    print("=" * 50)
    print(f"Inspecting CXRFEScore Cache in: {cache_dir}")
    print("=" * 50)

    facts_cache_path = os.path.join(cache_dir, "sent_to_facts.pkl")
    embed_cache_path = os.path.join(cache_dir, "fact_to_embedding.pkl")
    
    inspect_single_cache_file(facts_cache_path, "Sentence-to-Facts Cache")
    inspect_single_cache_file(embed_cache_path, "Fact-to-Embedding Cache")
    
    print("-" * 50)