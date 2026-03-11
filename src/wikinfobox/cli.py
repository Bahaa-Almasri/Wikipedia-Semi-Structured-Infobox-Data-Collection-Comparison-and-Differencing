from __future__ import annotations

import argparse

from .country_list import fetch_un_member_states
from .pipeline import collect_all_countries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Wikipedia infobox data for all UN member states."
    )
    parser.add_argument(
        "--no-save-html",
        action="store_true",
        help="Do not store raw infobox HTML snapshots.",
    )
    args = parser.parse_args()

    countries = fetch_un_member_states()
    print(f"Discovered {len(countries)} UN member states.")

    slugs = collect_all_countries(countries, save_html=not args.no_save_html)
    print(f"Successfully wrote JSON documents for {len(slugs)} countries.")


if __name__ == "__main__":
    main()

