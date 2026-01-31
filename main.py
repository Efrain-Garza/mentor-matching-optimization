"""
Project entry point for mentor matching optimization.
"""

from src.model import run_optimization
from src.utils import load_config


def main():
    cfg = load_config("config/default.yaml")
    run_optimization(cfg)


if __name__ == "__main__":
    main()
