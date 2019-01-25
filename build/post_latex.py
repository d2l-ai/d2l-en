from os import remove
from shutil import move
from tempfile import mkstemp


def unnumber_sections(source_file):
    _, target_file = mkstemp()
    preface_reached = False
    ch2_reached = False
    with open(target_file, 'w') as target_f, open(source_file, 'r') as source_f:
        for l in source_f:
            if l.rstrip() == '\\chapter*{Preface}\\addcontentsline{toc}{chapter}{Preface}':
                preface_reached = True
            if preface_reached and l.rstrip() == '\\chapter{A Taste of Deep Learning}':
                ch2_reached = True
            if not ch2_reached and preface_reached:
                if l.startswith('\section'):
                    target_f.write(l.replace('\\section', '\section*'))
                if l.startswith('\subsection'):
                    target_f.write(l.replace('\\subsection', '\subsection*'))
                if l.startswith('\subsubsection'):
                    target_f.write(l.replace('\\subsubsection', '\subsubsection*'))
            else:
                target_f.write(l)
    remove(source_file)
    move(target_file, source_file)


tex_file = 'build/_build/latex/d2l-en.tex'
unnumber_sections(tex_file)
