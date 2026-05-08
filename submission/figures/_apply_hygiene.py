"""Bulk LaTeX hygiene pass on paper_revised.tex."""
import re

P = r'D:/Quantum_kernel/quantum-sat-classification/submission/paper_revised.tex'
s = open(P, encoding='utf-8').read()
orig_len = len(s)

# 1) Remove \!=\!, \!<\!, \!>\! thin-space pairs around relations.
s = s.replace(r'\!=\!', '=')
s = s.replace(r'\!<\!', '<')
s = s.replace(r'\!>\!', '>')
s = s.replace(r'\!\le\!', r'\le ')
s = s.replace(r'\!\geq\!', r'\geq ')
s = s.replace(r'\!\ge\!', r'\ge ')
s = s.replace(r'\!\to\!', r'\to ')
s = s.replace(r'\!\in\!', r'\in ')
s = s.replace(r'\!\not\!', r'\not')
s = s.replace(r'\!\!', '')   # any leftover doubled thin-spaces

# 2) Thousand separators: 1800, 2000, 10000 (4-digit) -> 1{,}800 / 10{,}000
def thsep(m):
    n = m.group(0)
    if len(n) == 4:
        return n[0] + '{,}' + n[1:]
    if len(n) == 5:
        return n[:2] + '{,}' + n[2:]
    return n

# Only standalone 4-5 digit numbers, not part of larger ID/path
s = re.sub(r'(?<![\w{,])\d{4,5}(?![\w])', thsep, s)
# Also normalise occurrences like 1,800 (real comma) to 1{,}800 in math
s = re.sub(r'(?<![\w{,])(\d{1,2}),(\d{3})(?![\d])', r'\1{,}\2', s)

# 3) Standardise "macro-F1" capitalisation
s = s.replace('Macro-F1', 'macro-F1')
s = s.replace('Macro-F1', 'macro-F1')

open(P, 'w', encoding='utf-8').write(s)
print(f'wrote {P}: {orig_len} -> {len(s)} chars')
