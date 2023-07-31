import os
import re
import regex
import sys

def _unnumber_chaps_and_secs(lines):

    # Preface, Installation, and Notation are unnumbered chapters
    NUM_UNNUMBERED_CHAPS = 3
    # Prelimilaries
    TOC2_START_CHAP_NO = 5

    preface_reached = False
    ch2_reached = False
    num_chaps = 0
    for i, l in enumerate(lines):
        if l.startswith('\\chapter{'):
            num_chaps += 1
            # Unnumber unnumbered chapters
            if num_chaps <= NUM_UNNUMBERED_CHAPS:
                chap_name = re.split('{|}', l)[1]
                lines[i] = ('\\chapter*{' + chap_name
                            + '}\\addcontentsline{toc}{chapter}{'
                            + chap_name + '}\n')
            # Set tocdepth to 2 after Chap 1
            elif num_chaps == TOC2_START_CHAP_NO:
                lines[i] = ('\\addtocontents{toc}{\\protect\\setcounter{tocdepth}{2}}\n'
                            + lines[i])
        # Unnumber all sections in unnumbered chapters
        elif 1 <= num_chaps <= NUM_UNNUMBERED_CHAPS:
            if (l.startswith('\\section') or l.startswith('\\subsection')
                    or l.startswith('\\subsubsection')):
                lines[i] = l.replace('section{', 'section*{')

    # Since we inserted '\n' in some lines[i], re-build the list
    lines = '\n'.join(lines).split('\n')


# If label is of chap*/index.md title, its numref is Chapter X instead of Section X
def _sec_to_chap(lines):
    for i, l in enumerate(lines):
        # e.g., {Section \ref{\detokenize{chapter_dlc/index:chap-dlc}}} matches
        # {Section \ref{\detokenize{chapter_prelim/nd:sec-nd}}} does not match
        # Note that there can be multiple {Section } in one line

        longest_balanced_braces = regex.findall('\{(?>[^{}]|(?R))*\}', l)
        for src in longest_balanced_braces:
            if src.startswith('{Section \\ref') and 'index:' in src:
                tgt = src.replace('Section \\ref', 'Chapter \\ref')
                lines[i] = lines[i].replace(src, tgt)

# Apple roman numbers for front matters and page number 1 starts from the Introduction chapter
def _pagenumbering(lines):
    BEGINDOC = '\\begin{document}'
    FRONTNUMS = ['\\pagenumbering{roman}',
    '\\pagestyle{empty}',
    '\\halftitle',
    '\\cleardoublepage']
    INTRONUMS = ['\\mainmatter', '\\pagenumbering{arabic}', '\\setcounter{page}{1}']
    CHAPINTRO = '\\chapter{Introduction}'
    chapintro_i = -1
    for i, l in enumerate(lines):
        if l.startswith(BEGINDOC):
            frontnums_i = i + 1
        elif l.startswith(CHAPINTRO):
            chapintro_i = i
            break
    for i, v in enumerate(FRONTNUMS):
        lines.insert(frontnums_i + i, v)
    for i, v in enumerate(INTRONUMS):
        if chapintro_i > 0:
            lines.insert(chapintro_i + len(FRONTNUMS) + i, v)

# E.g., \chapter{Builders’ Guide} -> \chapter{Builders' Guide}
def _replace_quote_in_chapter_title(lines):
    for i, l in enumerate(lines):
        if l.startswith('\\chapter{'):
            lines[i] = lines[i].replace('’', '\'')


# Remove date
def _edit_titlepage(pdf_dir):
    smanual = os.path.join(pdf_dir, 'sphinxmanual.cls')
    with open(smanual, 'r') as f:
        lines = f.read().split('\n')

    for i, l in enumerate(lines):
        lines[i] = lines[i].replace('\\@date', '')

    with open(smanual, 'w') as f:
        f.write('\n'.join(lines))


def delete_lines(lines, deletes):
    return [line for i, line in enumerate(lines) if i not in deletes]


def _delete_discussions_title(lines):
    deletes = []
    to_delete = False
    for i, l in enumerate(lines):
        if 'section*{Discussion' in l or 'section{Discussion' in l:
            to_delete = True
        elif to_delete and '\\sphinxincludegraphics' in l:
            to_delete = False
        if to_delete:
            deletes.append(i)
    return delete_lines(lines, deletes)

# Within \caption{} or \sphinxcaption{}: \hyperlink -> \protect\hyperlink
def _protect_hyperlink_in_caption(lines):
    def _get_num_extra_left_braces(l, num_extra_left_braces):
        num = num_extra_left_braces
        for char in l:
            if char == '{':
                num += 1
            elif char == '}':
                num -= 1
                if num == 0:
                    return 0
        return num

    i = 0
    while i < len(lines):
        if lines[i].startswith('\\caption{') or lines[i].startswith('\\sphinxcaption{'):
            num_extra_left_braces = _get_num_extra_left_braces(lines[i], 0)
            if num_extra_left_braces == 0:
                j = i
            else:
                j = i + 1
                while j < len(lines):
                    num_extra_left_braces = _get_num_extra_left_braces(
                            lines[j], num_extra_left_braces)
                    if num_extra_left_braces == 0:
                        break
                    j += 1
            # lines[i] -- lines[j] are within \caption{} or \sphinxcaption{}
            for index in range(i, j + 1):
                lines[index] = lines[index].replace('\\hyperlink', '\\protect\\hyperlink')
            i = j + 1
        else:
            i += 1


def main():
    tex_file = sys.argv[1]
    with open(tex_file, 'r') as f:
        lines = f.read().split('\n')

    _unnumber_chaps_and_secs(lines)
    _sec_to_chap(lines)
    #lines = _delete_discussions_title(lines)
    _protect_hyperlink_in_caption(lines)
    _pagenumbering(lines)
    _replace_quote_in_chapter_title(lines)

    with open(tex_file, 'w') as f:
        f.write('\n'.join(lines))

    pdf_dir = os.path.dirname(tex_file)
    #_edit_titlepage(pdf_dir)

if __name__ == "__main__":
    main()
