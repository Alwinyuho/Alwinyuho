"""Download data for all benchmark languages. Usage: python scripts/build_data.py"""

import json

from bhasharag import config
from bhasharag.data import build_language

if __name__ == "__main__":
    for lang in config.LANGUAGES:
        print(json.dumps(build_language(lang)), flush=True)
