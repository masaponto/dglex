import argparse
import os
import sys
from typing import Any, Dict


def load_config(config_path: str = "dglex.yaml") -> Dict[str, Any]:
    """設定ファイルから設定を読み込む。

    Args:
        config_path (str): YAML 設定ファイルのパス。

    Returns:
        Dict[str, Any]: view セクションの設定辞書。ファイルが存在しない場合は空辞書。
    """
    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and isinstance(config, dict) and "view" in config:
                    return config["view"]
        except Exception as e:
            print(
                f"Warning: Failed to load configuration from {config_path}: {e}",
                file=sys.stderr,
            )
    return {}


def merge_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """コマンドライン引数と設定ファイルの設定をマージする。

    Args:
        args (argparse.Namespace): コマンドライン引数。
        config (Dict[str, Any]): 設定ファイルから読み込んだ設定。

    Returns:
        argparse.Namespace: マージ後の引数。
    """
    import json

    args_dict = vars(args)

    settings_to_merge = {
        "index": "index",
        "title": "title",
        "edge_weight": "edge_weight",
        "node_palette": "node_palette",
        "edge_palette": "edge_palette",
        "output": "output",
        "figsize": "figsize",
        "reverse_etypes": "reverse_etypes",
    }

    defaults = {
        "index": 0,
        "title": None,
        "edge_weight": None,
        "node_palette": "tab10",
        "edge_palette": "tab10",
        "output": None,
        "figsize": None,
        "reverse_etypes": None,
    }

    for arg_name, config_key in settings_to_merge.items():
        if arg_name in args_dict and config_key in config:
            if args_dict[arg_name] == defaults.get(arg_name):
                if config_key == "figsize" and isinstance(config[config_key], list):
                    args_dict[arg_name] = tuple(config[config_key])
                else:
                    args_dict[arg_name] = config[config_key]

    if args.reverse_etypes and isinstance(args.reverse_etypes, str):
        try:
            args_dict["reverse_etypes"] = json.loads(args.reverse_etypes)
        except json.JSONDecodeError as e:
            print(f"Error parsing --reverse-etypes: {e}", file=sys.stderr)
            args_dict["reverse_etypes"] = None

    # Merge limits
    default_limits: Dict[str, Any] = {
        "max_nodes": 500,
        "max_edges": 2000,
        "sample_size": 5,
        "sample_fanouts": [10],
    }

    args_dict["limits"] = default_limits.copy()
    if "limits" in config and isinstance(config["limits"], dict):
        for key, value in config["limits"].items():
            if key in default_limits:
                args_dict["limits"][key] = value

    return args
