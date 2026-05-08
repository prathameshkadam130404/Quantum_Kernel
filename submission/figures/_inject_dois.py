"""Inject DOI URLs into bibitems in paper_revised.tex."""
import re

P = r'D:/Quantum_kernel/quantum-sat-classification/submission/paper_revised.tex'
s = open(P, encoding='utf-8').read()

doi_map = {
    'huang2021':      'https://doi.org/10.1038/s41467-021-22539-9',
    'havlicek2019':   'https://doi.org/10.1038/s41586-019-0980-2',
    'schuld2019':     'https://doi.org/10.1103/PhysRevLett.122.040504',
    'liu2021':        'https://doi.org/10.1038/s41567-021-01287-z',
    'thanasilp2024':  'https://doi.org/10.1038/s41467-024-49287-w',
    'shaydulin2022':  'https://doi.org/10.1103/PhysRevA.106.042407',
    'hubregtsen2022': 'https://doi.org/10.1103/PhysRevA.106.042431',
    'suzuki2020':     'https://doi.org/10.1007/s42484-020-00020-y',
    'helber2019':     'https://doi.org/10.1109/JSTARS.2019.2918242',
    'qiu2020':        'https://doi.org/10.1016/j.isprsjprs.2019.05.004',
    'zhu2020':        'https://doi.org/10.1109/MGRS.2020.2964708',
    'rodriguez2025':  'https://doi.org/10.1088/2632-2153/ad9c80',
    'schnabel2025':   'https://doi.org/10.1007/s42484-025-00273-5',
    'bowles2024':     'https://arxiv.org/abs/2403.07059',
    'schuld2021':     'https://arxiv.org/abs/2101.11020',
}

added = 0
for key, url in doi_map.items():
    needle = '\\bibitem{' + key + '}'
    idx = s.find(needle)
    if idx < 0:
        print('MISS', key); continue
    # find end of this bibitem (next \bibitem or \end{thebibliography})
    nxt = min(
        x for x in [s.find('\\bibitem{', idx + 1),
                    s.find('\\end{thebibliography}', idx + 1)]
        if x >= 0
    )
    block = s[idx:nxt]
    if 'doi.org' in block or 'arxiv.org' in block.lower():
        continue  # already has URL
    # insert url before the trailing whitespace
    rstripped = block.rstrip() + '\n\\url{' + url + '}.\n\n'
    s = s[:idx] + rstripped + s[nxt:]
    added += 1
    print('+', key)

print(f'\nadded {added} URLs')
open(P, 'w', encoding='utf-8').write(s)
