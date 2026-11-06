"""Create a linked child research run from an explicitly approved local change."""

from __future__ import annotations

import argparse
import json

from crypto_quant.research.factor_mining.native_changes import record_human_change


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a human-approved FM-v6 change and prepare its child A ticket."
    )
    parser.add_argument("--binding", required=True, help="Trusted parent native-chat binding JSON")
    parser.add_argument("--approval", required=True, help="Human-authored change approval JSON")
    args = parser.parse_args()
    print(json.dumps(record_human_change(args.binding, args.approval),
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
