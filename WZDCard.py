import os
import glob
import re
import tempfile
import subprocess
import shutil
import numpy as np

class WZDCard:

    _sub_pattern = re.compile(r'\$SUB:([a-zA-Z0-9_]*)')

    def __init__(self, path):
        self.path = path
        with open(self.path) as file:
            self.content = file.read()
        self.vars = self._find_vars()

    def _find_vars(self):
        return sorted(set(self._sub_pattern.findall(self.content)))

    def sub(self, vals):
        content = self.content
        for var in self.vars:
            content = content.replace(f'$SUB:{var}', str(vals[var]))
        return content

    def run(self, vals, capture_output=False):
        workdir = vals.pop('workdir')
        content = self.sub(vals)
        os.makedirs(workdir)
        sinpath = os.path.join(workdir, 'input.sin')
        with open(sinpath, 'w') as sinfile: sinfile.write(content)
        proc = subprocess.run(['whizard', 'input.sin'], capture_output=capture_output, cwd=workdir)
        return proc

def load_cards(basedir='cards'):
    cards = { }
    for path in glob.glob(os.path.join(basedir, '*.sin')):
        cards[os.path.basename(path)] = WZDCard(path)
    return cards

if __name__ == '__main__':
    for path, card in load_cards().items():
        print(path, card.vars, card.content, sep='\n' + '-' * 40 + '\n', flush=True)
