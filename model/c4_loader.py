from io import IOBase
from model.common import TextLoader
import json
import gzip
from pathlib import Path

READ_CHUNK_SIZE = 4096

class C4Loader(TextLoader):
    def open_file(self, path: Path) -> IOBase:
        return gzip.open(path, 'rb')

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
            if len(text) > 0:
                return text
