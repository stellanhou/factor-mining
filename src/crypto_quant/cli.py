"""Command entry point for the independent factor-mining repository."""
import argparse
import json
from crypto_quant.research.factor_mining.cli import add_arguments, execute
from crypto_quant.research.data_policy import policy_catalog


def build_parser():
    parser = argparse.ArgumentParser(prog="factor-mining")
    commands = parser.add_subparsers(dest="command", required=True)
    add_arguments(commands.add_parser("factor-mine"))
    commands.add_parser("research-data-policy")
    return parser


def main():
    args = build_parser().parse_args()
    result = policy_catalog() if args.command == "research-data-policy" else execute(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
