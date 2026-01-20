from pathlib import Path
from io import IOBase, BufferedReader
from model.common import TextLoader
import json
import zstandard

class PileLoader(TextLoader):
    def open_file(self, path: Path) -> IOBase:
        return BufferedReader(zstandard.open(path, 'rb'))

    def next(self) -> str:
        if self.current_file is None:
            self.switch_file_by_index(self.current_index)

        while True:
            line = self.current_file.readline()
            if len(line) == 0:
                # eof
                self.next_file()
                continue

            line = line.strip()
            if len(line) == 0:
                continue

            parsed = json.loads(line)
            text = parsed['text']
            if len(text) == 0:
                continue

            set_name = parsed['meta']['pile_set_name']
            if set_name not in (
                'BookCorpus2', 'Books3', 'Enron Emails', 'Gutenberg (PG-19)',
                'HackerNews', 'OpenWebText2', 'Ubuntu IRC', 'Wikipedia (en)',
                'StackExchange', 'Pile-CC', 'USPTO Backgrounds', 'OpenSubtitles',
            ):
                continue

            return text
