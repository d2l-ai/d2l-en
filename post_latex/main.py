import re
import regex
import sys

def _unnumber_chaps_and_secs(lines):
    def _startswith_unnumbered(l):
        UNNUMBERED = {'\\section{Summary',
                      '\\section{Exercise',
                      '\\section{Exercises',
                      '\\section{Discussions',
                      '\\subsection{Summary',
                      '\\subsection{Exercise',
                      '\\subsection{Exercises',
                      '\\subsection{Discussions'}
        for unnum in UNNUMBERED:
            if l.startswith(unnum):
                return True
        return False

    # Preface and Installation are numbered chapters
    NUM_UNNUMBERED_CHAPS = 2
    # Prelimilaries
    TOC2_START_CHAP_NO = 4

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
        # Unnumber summary, references, exercises, qr code in numbered chapters
        elif _startswith_unnumbered(l):
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


def main():
    tex_file = sys.argv[1]
    with open(tex_file, 'r') as f:
        lines = f.read().split('\n')

    _unnumber_chaps_and_secs(lines)
    _sec_to_chap(lines)

    with open(tex_file, 'w') as f:
        f.write('\n'.join(lines))

if __name__ == "__main__":
    main()
