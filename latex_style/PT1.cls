%%
%% This is file 'PT1.cls'
%% PT1 LaTeX2e document class
%%
%% copyright (c) 2010 ali woollatt
%%   generated from cambridge7A.cls 2009/09/16, v2.00
%%
%% Bugs (in the case of unchanged files) should be reported to
%% texline@cambridge.org
%%
%% 2010/10/10, v1.10 includes the following changes:
%%   01 correction to long titles in appendices
%%   02 \evensidemargin changed to 61pt
%%   03 \marginparwidth changed to 54pt
%%   04 inclusion of artwork PT1header.eps, PT1chrule.eps and PT1secrule.eps
%%
%% 2010/12/02, v1.20 includes the following changes:
%%   01 inclusion of artwork PT1box.eps
%%   02 changed chapter titles in appendices to sf
%%   03 chapter authors u/l case
%%   04 footnotes justified
%%   05 figure captions flush left
%%   06 endnote indents cleared For 10
%%   07 summary environment added
%%   08 unnumbered list indents Changed
%%   09 \authoraffil added
%%   10 chapterquote source added
%%   11 example environment improved
%%
%% This software may only be used in the preparation of journal articles
%% or books or parts of books to be published by Cambridge University Press.
%% Any other use constitutes an infringement of copyright.
%%
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%

\NeedsTeXFormat{LaTeX2e}

\newcommand\version{2010/12/02, v1.20 }
\ProvidesClass{PT1}[\version LaTeX2e document class]

% font sizes for PT1, 246 x 189mm
% \tiny                        5/6
% \scriptsize                  7/8
% \indexsize                   8/10
%   \let\footnotesize\indexsize
% \affiliationsize             8/11 (chapter heads)
% \smallish                    9/11
%   \let\enotesize\smallish
% \abstractsize                9/12
%   \let\tablesize\abstractsize
% \headandfootsize             9/13
% \extractsize               9.5/12
% \fignumbersize             9.5/13
% \normalsmall                10/12
% \let\listsize\normalsize    10/13
% \let\figcaptionsize\normalsize
% \tablecaptionsize         10.5/11
% \yheadsize                10.5/13
% \floatingboxsize          10.5/14 (summaries)
% \xheadsize                  11/13
% \large                      11/14
% \Large                      12/17 (chapter authors)
% \parttitlesintoc            12/13 (part titles in toc)
% \Bheadsize                  13/14 (B,C,D heads)
% \Aheadsize                  15/16 (A heads)
% \LARGE                      16/18
% \huge                       20/22 (chapter numbers/titles)
% \quitehuge                  22/22 (part numbers)
% \superhuge                  30/34 (part titles)


\newif\ifAJW@multisty
%\newif\ifAJW@largesty
\newif\ifAJW@indentsty % false
\newif\ifAJW@landscape
\newif\if@openright
\newif\ifAJW@truesize
\newif\ifchapterrule
\newdimen\AJW@indent
\newdimen\AJW@floatindent
\AJW@indent\z@        % zero
\newif\ifsp@nningrule

\usepackage{color,framed}%,graphics

\DeclareOption{multi}{\AJW@multistytrue}

%\DeclareOption{cup5a}{\AJW@largestytrue\AJW@indentstytrue\AJW@indent 3pc}
%\DeclareOption{CUP5A}{\@nameuse{ds@cup5a}}
%\DeclareOption{cup5b}{\AJW@multistytrue\AJW@largestytrue
%  \AJW@indentstytrue\AJW@indent 3pc}
%\DeclareOption{CUP5B}{\@nameuse{ds@cup5b}}
%\DeclareOption{cup6a}{\AJW@largestytrue}
%\DeclareOption{CUP6A}{\@nameuse{ds@cup6a}}
%\DeclareOption{cup6b}{\AJW@multistytrue\AJW@largestytrue}
%\DeclareOption{CUP6B}{\@nameuse{ds@cup6b}}
%\DeclareOption{cup7a}{\relax}
%\DeclareOption{CUP7A}{\relax}
%\DeclareOption{cup7b}{\AJW@multistytrue}
%\DeclareOption{CUP7B}{\@nameuse{ds@cup7b}}
%\DeclareOption{cup8a}{\AJW@indentstytrue\AJW@indent 3pc}
%\DeclareOption{CUP8A}{\@nameuse{ds@cup8a}}
%\DeclareOption{cup8b}{\AJW@multistytrue\AJW@indentstytrue\AJW@indent 3pc}
%\DeclareOption{CUP8B}{\@nameuse{ds@cup8b}}
%\DeclareOption{cup9a}{\AJW@indentstytrue\AJW@indent=\z@}
%\DeclareOption{CUP9A}{\@nameuse{ds@cup9a}}
%\DeclareOption{cup9b}{\AJW@multistytrue\AJW@indentstytrue}
%\DeclareOption{CUP9B}{\@nameuse{ds@cup9b}}

\DeclareOption{oneside}{\relax}
\DeclareOption{twoside}{\@twosidetrue \@mparswitchtrue}
\DeclareOption{onecolumn}{\@twocolumnfalse}
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{draft}{\setlength\overfullrule{5\p@}}
\DeclareOption{final}{\setlength\overfullrule{\z@}}
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\DeclareOption{a4paper}
  {\setlength\paperheight {297mm}%
   \setlength\paperwidth  {210mm}}
\DeclareOption{landscape}
  {\setlength\@tempdima   {\paperheight}%
   \setlength\paperheight {\paperwidth}%
   \setlength\paperwidth  {\@tempdima}%
   \AJW@landscapetrue}
\DeclareOption{truesize}{\AJW@truesizetrue}
\DeclareOption{spanningrule}{\sp@nningruletrue}

\newif\ifprodtf
\DeclareOption{prodtf}{\prodtftrue}

\ExecuteOptions{a4paper,twoside,onecolumn,final,openright}
\ProcessOptions\relax

\ifprodtf
\AtBeginDocument{
  \usepackage{myriad-pt1}%
  %
  \newcommand{\seriffont}{\MyriadProCond}
  \newcommand{\semiboldseriffont}{\MyriadProSemiBoldCond}
%
% take the kerning out of the \LaTeXe macro for condensed font (title)
  \newcommand\LaTeXetitle{\protect{{L\kern-.12em\protect
    \raisebox{0.15em}{\LARGE A}\kern-.05emT\kern-.12em\lower.4ex\hbox{E}\kern-.02emX}}}
  \newcommand\LaTeXeintitle{\LaTeXetitle{}\kern-.00em2$_{\textstyle\varepsilon}$}
%
% take the kerning out of the \LaTeXe macro for condensed font (section heads)
  \newcommand\LaTeXesection{\protect{{L\kern-.12em\protect
    \raisebox{0.14em}{\Large A}\kern-.05emT\kern-.12em\lower.4ex\hbox{E}\kern-.02emX}}}
  \newcommand\LaTeXeinsectionhead{\LaTeXesection{}\kern-.00em2$_{\textstyle\varepsilon}$}
%
% take the kerning out of the \TeX macro for condensed font (section heads)
  \newcommand\TeXinsectionhead{\protect{{T\kern-.12em\lower.4ex\hbox{E}\kern-.02emX}}}
  }%
\else
  \newcommand{\seriffont}{\sf}
  \newcommand{\semiboldseriffont}{\sf\bfseries}
  \let\LaTeXeintitle\LaTeXe
  \let\LaTeXeinsectionhead\LaTeXe
  \let\TeXinsectionhead\TeX
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtEndOfPackage{\AtBeginDocument{%

% 1. to bring amsthm.sty    into line with cambridge style
% 2. to bring multind.sty   into line with cambridge style
% 3. to bring index.sty     into line with cambridge style
% 4. to bring natbib.sty    into line with cambridge style
% 5. to bring rotating.sty  into line with cambridge style
% 6. to reset values for letterspacing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 1. to bring amsthm.sty into line with cambridge style
%
% replace period after heading with \enskip and increase space above proofs to 23pt
%
\@ifpackageloaded{amsthm}{%
\providecommand\proof{}
\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}%
  \normalfont \topsep 10pt plus 3pt minus 1pt\relax
  \trivlist
  \item[\hskip\labelsep
%   \itshape
    \xheadsize\semiboldseriffont
%   #1\@addpunct{.}]\ignorespaces
    #1\enskip]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}
%

% increase space above theorems to 23pt
\def\thm@space@setup{%
  \thm@preskip=10pt plus 3pt minus 1pt%\topsep
    \thm@postskip=\thm@preskip
}

% increase space above remarks to 23pt
\def\th@remark{%
  \thm@headfont{\itshape}%
  \normalfont % body font
  \thm@preskip=10pt plus 3pt minus 1pt%\topsep
  \thm@postskip\thm@preskip
}

% replace period after heading with \enskip
%
\def\@thm#1#2#3{%
  \ifhmode\unskip\unskip\par\fi
  \normalfont
  \trivlist
  \let\thmheadnl\relax
  \let\thm@swap\@gobble
% \thm@headpunct{.}% add period after heading
  \thm@headpunct{\enskip}%
  \thm@headsep 5\p@ plus\p@ minus\p@\relax
  \thm@space@setup
  #1% style overrides
  \@topsep\thm@preskip               % used by thm head
  \@topsepadd \thm@postskip           % used by \@endparenv
  \def\@tempa{#2}\ifx\@empty\@tempa
    \def\@tempa{\@oparg{\@begintheorem{\xheadsize\semiboldseriffont #3}{}}[]}%
  \else
    \refstepcounter{#2}%
    \def\@tempa{\@oparg{\@begintheorem{\xheadsize\semiboldseriffont #3}%
      {\csname the#2\endcsname}}[]}%
  \fi
  \@tempa
}}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 2. to bring multind.sty into line with cambridge style
%      note that an error in multind.sty will stop verbatim environment
%      working; also everything will come out \indexsize
%      to correct this, we've added an extra set of parens
%
% change chapter opening
%
\@ifpackageloaded{multind}{%
  \def\printindex#1#2{{\@restonecoltrue
    \if@twocolumn\@restonecolfalse\fi
    \columnseprule\z@
    \columnsep 1pc%
    \twocolumn[\vspace*{10\p@}\mbox{}\par\@makeschapterhead{#2}]%
    \@mkboth{#2}{#2}%\markright{\uppercase{#2}}
    \typeout{#2}%
    \addcontentsline{toc}{schapter}{#2}%
    \thispagestyle{plain}%
    \indexsize\raggedright
    \parindent\z@
    \parskip\z@ \@plus.3\p@\relax
    \let\item=\@idxitem
    \@input{#1.ind}}}
%
  \def\@idxitem{\par\hangindent 2em}
  \def\subitem{\par\hangindent 3em \hspace*{1em}}
  \def\subsubitem{\par\hangindent 4em \hspace*{2em}}
  \def\indexspace{\par \vskip 3\p@ \@plus 1\p@ \@minus 1\p@\relax}}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 3. to bring index.sty into line with cambridge style
%
\@ifpackageloaded{index}{%
  \renewenvironment{theindex}{%
    \edef\indexname{\the\@nameuse{idxtitle@\@indextype}}%
    \if@twocolumn
      \@restonecolfalse
    \else
      \@restonecoltrue
    \fi
    \columnseprule \z@
    \columnsep 1pc
    %\twocolumn[\@makeschapterhead{\indexname}%
    \twocolumn[\vspace*{10\p@}\mbox{}\par\@makeschapterhead{\indexname}
    \ifx\index@prologue\@empty\else
      \index@prologue
      \bigskip
    \fi
  ]%
    % \@mkboth{\MakeUppercase\indexname}%
    % {\MakeUppercase\indexname}%
  \@mkboth{\indexname}%
  {\indexname}%
  \ifAJW@indentsty
    \ifdim\AJW@indent>\z@
      \addcontentsline{toc}{schapter}{\protect\numberline{\mbox{}}\indexname}
    \else
      \addcontentsline{toc}{schapter}{\indexname}
    \fi
  \else
    \addcontentsline{toc}{schapter}{\indexname}
  \fi
  \thispagestyle{plain}%
  \indexsize\raggedright
  \parindent\z@
  \parskip\z@ \@plus .3\p@\relax
  \let\item\@idxitem
  }{%
  \if@restonecol
    \onecolumn
  \else
    \clearpage
  \fi
  }}%
{}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 4. to bring natbib.sty into line with cambridge style
%    and make 'References' the default
%
\@ifpackageloaded{natbib}{%
\providecommand\refname{References}
\providecommand\bibname{Bibliography}

\ifAJW@multisty
  \def\NAT@sectionbib{on}% natbib will use \section* headings
\fi

\setlength\bibhang{1em}
\renewenvironment{thebibliography}[1]{%
  \bibsection\parindent \z@\bibpreamble\normalsize\list
   {\@biblabel{\arabic{NAT@ctr}}}{\@bibsetup{#1}%
    \setcounter{NAT@ctr}{0}}%
    \ifNAT@openbib
      \renewcommand\newblock{\par}
    \else
      \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
    \fi
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax
    \let\citeN\cite \let\shortcite\cite
    \let\citeasnoun\cite
 }{\def\@noitemerr{%
  \PackageWarning{natbib}
     {Empty `thebibliography' environment}}%
  \endlist\vskip-\lastskip}

\ifAJW@multisty
  \renewcommand\bibsection{\section*{\protect\refname
    %\ifx\@mkboth\@gobbletwo\else\markright{\refname}\fi
    }}%
\else
  \renewcommand\bibsection{\chapter*{\refname
    %\@mkboth{\refname}{\refname}
    }}%
\fi}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 5. to bring rotating.sty into line with cambridge style
%
%
% \rotFPtop        \rotFPbot          Effect
% 0pt plus 1fil    0pt plus 1fil      figure/table appears in middle (default value)
% 0pt              0pt plus 1fil      figure/table appears with its top nearest the edge of the page
% 0pt plus 1fil    0pt plus 2fil      figure/tables bottom appears twice as far from the edge as the top does

\@ifpackageloaded{rotating}{%
%
\def\sidewaysfigure{\@rotfloat{figure}%
  \let\@makecaption=\make@rotatedfigure@caption}
%
\def\sidewaystable{\@rotfloat{table}%
  \let\@makecaption\@tablecaption}
%
  \rotFPtop 0pt plus 1fil%
  \rotFPbot 0pt plus 1fil%
%
  }%
  {}%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 6. to reset values for letterspacing
%    soul.sty removed for v1.00
%
% default values:
%\sodef\textso{}{.25em}{.65em\@plus.08em\@minus.06em}%
%        {.55em\@plus.275em\@minus.183em}%

\ifprodtf
\@ifpackageloaded{soul}{%
  \sodef\lettersp{}{0em}{0em}{0em}%
  }%
  {}%
\else
\@ifpackageloaded{soul}{%
  \sodef\lettersp{}{.01em}{.35em\@plus.08em\@minus.06em}%
    {.35em\@plus.275em\@minus.183em}%
  }%
  {}%
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}}% end of \AtEndOfPackage section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  \newcommand\cls@pttxt[1]{\typeout{cambridge7A.cls, [multi] option}}
% \newcommand\CUP@five{\ifAJW@multisty \cls@pttxt{5b}\else \cls@pttxt{5a}\fi}
% \newcommand\CUP@six{\ifAJW@multisty \cls@pttxt{6b}\else \cls@pttxt{6a}\fi}
  \newcommand\CUP@seven{\ifAJW@multisty \cls@pttxt{cambridge7B}\fi}
% \newcommand\CUP@eight{\ifAJW@multisty \cls@pttxt{8b}\else \cls@pttxt{8a}\fi}
% \newcommand\CUP@nine{\ifAJW@multisty \cls@pttxt{9b}\else \cls@pttxt{9a}\fi}

% \ifAJW@largesty
% \ifAJW@indentsty \CUP@five\else \CUP@six\fi
% \else
% \ifAJW@indentsty {\ifdim\AJW@indent=\z@ \CUP@nine\else \CUP@eight\fi}%
% \else
    \CUP@seven
% \fi
% \fi

\newif\if@mainmatter \@mainmattertrue
\newcommand\frontmatter{\cleardoublepage
            \@mainmatterfalse\pagenumbering{roman}}
\newcommand\mainmatter{\cleardoublepage
       \@mainmattertrue\pagenumbering{arabic}}
\newcommand\backmatter{\if@openright\cleardoublepage\else\clearpage\fi
      \@mainmatterfalse}

%  ****************************************
%  *              SHADING                 *
%  ****************************************

 \definecolor{tencolor}{gray}{.9}       %box text
 \definecolor{fifteencolor}{gray}{.85}  %lower half of tables
 \definecolor{fortycolor}{gray}{.6}     %table captions, running head, section head rule
 \definecolor{sixtycolor}{gray}{.4}     %box titles
%\definecolor{fortycolor}{cmyk}{0,0,0,0.4}% 40 per cent

%  ****************************************
%  *               FONTS                  *
%  ****************************************
%
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\newskip\AJW@baseskip
\newskip\AJW@theskip
\def\AJW@setskips#1{\AJW@theskip #1\relax%
  \abovedisplayskip      0.50\AJW@theskip \@plus 0.25\AJW@theskip \@minus 1\p@%
  \belowdisplayskip      \abovedisplayskip
  \abovedisplayshortskip 0.25\AJW@theskip \@plus 0.25\AJW@theskip
  \belowdisplayshortskip 0.50\AJW@theskip \@plus 0.25\AJW@theskip \@minus 1\p@%
}
\def\AJW@setlists#1{\AJW@theskip #1\relax%
 \def\@listi{\leftmargin\leftmargini
  \topsep  0.5\AJW@theskip \@plus 2\p@ \@minus 1\p@%
  \parsep  \z@
  \itemsep \z@}}
%
%
% common sizes
\newcommand{\@ixptvpt}{9.5}%                  9.5pt
  \DeclareMathSizes{\@ixptvpt}{\@ixptvpt}{\@viipt}{\@vpt}
\newcommand{\@xptvpt}{10.5}%                  10.5pt
  \DeclareMathSizes{\@xptvpt}{\@xptvpt}{\@viipt}{\@vpt}
\newcommand{\@xiiipt}{13}%                    13pt
  \DeclareMathSizes{\@xiiipt}{\@xiiipt}{\@xpt}{\@ixpt}
\newcommand{\@xvpt}{15}%                      15pt
  \DeclareMathSizes{\@xvpt}{\@xvpt}{\@xiipt}{\@xpt}
\newcommand{\@xvipt}{16}%                     16pt
  \DeclareMathSizes{\@xvipt}{\@xvipt}{\@xiipt}{\@xpt}
\newcommand{\@xviiipt}{18}%                   18pt
  \DeclareMathSizes{\@xviiipt}{\@xviiipt}{\@xiipt}{\@xpt}
\newcommand{\@xxiipt}{22}%                    22pt
  \DeclareMathSizes{\@xxiipt}{\@xxiipt}{\@xxpt}{\@xviipt}
\newcommand{\@xxxpt}{30}%                     30pt
  \DeclareMathSizes{\@xxxpt}{\@xxxpt}{\@xxvpt}{\@xxpt}
\newcommand{\@xxxvipt}{36}%                   36pt
  \DeclareMathSizes{\@xxxvipt}{\@xxxvipt}{\@xxvpt}{\@xxpt}

\newcommand\tiny{\@setfontsize\tiny\@vpt{6}\AJW@setskips\AJW@setlists}
\newcommand\indexsize{\AJW@baseskip 10pt%
  \@setfontsize\indexsize\@viiipt{10}\AJW@setskips\AJW@setlists}
\let\footnotesize\indexsize

\newcommand\abstractsize{\AJW@baseskip 12pt%
  \@setfontsize\abstractsize\@ixpt{12}\AJW@setskips\AJW@setlists}
\let\tablesize\abstractsize

\newcommand\affiliationsize{\AJW@baseskip 11pt%
  \@setfontsize\affiliationsize\@viiipt{11}\AJW@setskips\AJW@setlists}
\newcommand\headandfootsize{\AJW@baseskip 13pt%
  \@setfontsize\headandfootsize\@ixpt{13}\AJW@setskips\AJW@setlists}
\newcommand\extractsize{\AJW@baseskip 12pt%
  \@setfontsize\extractsize\@ixptvpt{12}\AJW@setskips\AJW@setlists}
\newcommand\fignumbersize{\AJW@baseskip 13pt%
  \@setfontsize\fignumbersize\@ixptvpt{13}\AJW@setskips\AJW@setlists}
\newcommand\tablecaptionsize{\AJW@baseskip 11pt%
  \@setfontsize\tablecaptionsize\@xptvpt{11}\AJW@setskips\AJW@setlists}
\newcommand\yheadsize{\AJW@baseskip 13pt%
  \@setfontsize\yheadsize\@xptvpt{13}\AJW@setskips\AJW@setlists}
\newcommand\floatingboxsize{\AJW@baseskip 14pt%
  \@setfontsize\floatingboxsize\@xptvpt{14}\AJW@setskips\AJW@setlists}
\newcommand\xheadsize{\AJW@baseskip 13pt%
  \@setfontsize\xheadsize\@xipt{13}\AJW@setskips\AJW@setlists}
\newcommand\large{\@setfontsize\large\@xipt{14}\AJW@setskips\AJW@setlists}
\newcommand\parttitlesintoc{\@setfontsize\parttitlesintoc\@xiipt{13}\AJW@setskips\AJW@setlists}
\newcommand\Large{\@setfontsize\Large\@xiipt{17}\AJW@setskips\AJW@setlists}
\newcommand\Bheadsize{\@setfontsize\Bheadsize\@xiiipt{14}\AJW@setskips\AJW@setlists}
\newcommand\Aheadsize{\@setfontsize\Aheadsize\@xvpt{16}\AJW@setskips\AJW@setlists}
\newcommand\LARGE{\@setfontsize\LARGE\@xvipt{18}\AJW@setskips\AJW@setlists}
\newcommand\huge{\@setfontsize\huge\@xxpt{22}\AJW@setskips\AJW@setlists}
\newcommand\quitehuge{\@setfontsize\quitehuge\@xxiipt{22}\AJW@setskips\AJW@setlists}
\newcommand\superhuge{\@setfontsize\superhuge\@xxxpt{34}\AJW@setskips\AJW@setlists}%


%\ifAJW@largesty
%  \newcommand\scriptsize{%
%    \@setfontsize\scriptsize\@viiipt{9}\AJW@setskips\AJW@setlists}
%  \newcommand\enotesize{\AJW@baseskip 10pt%
%    \@setfontsize\enotesize\@ixpt{10}\AJW@setskips\AJW@setlists}
%  \newcommand\small{\AJW@baseskip 11pt%
%    \@setfontsize\small\@xpt{11}\AJW@setskips\AJW@setlists}
%  \let\tablesize\small
%  \newcommand\normalsmall{\AJW@baseskip 12pt%
%    \@setfontsize\normalsmall\@xpt{12}\AJW@setskips\AJW@setlists}
%  \let\smallish\normalsmall
%  \newcommand\listsize{\AJW@baseskip 13pt%
%    \@setfontsize\listsize\@xpt{13}\AJW@setskips\AJW@setlists}
%  \renewcommand\normalsize{\AJW@baseskip 14pt%
%    \@setfontsize\normalsize\@xipt{14}\AJW@setskips{\AJW@baseskip}%
%    \let\@listi\@listI}
%\else
  \newcommand\scriptsize{%
    \@setfontsize\scriptsize\@viipt{8}\AJW@setskips\AJW@setlists}
  \newcommand\small{\AJW@baseskip 10pt%
    \@setfontsize\small\@ixpt{10}\AJW@setskips\AJW@setlists}
  \newcommand\smallish{\AJW@baseskip 11pt%
    \@setfontsize\smallish\@ixpt{11}\AJW@setskips\AJW@setlists}
  \let\enotesize\smallish
  \newcommand\normalsmall{\AJW@baseskip 12pt%
    \@setfontsize\normalsmall\@xpt{12}\AJW@setskips\AJW@setlists}
  \renewcommand\normalsize{\AJW@baseskip 13pt%
    \@setfontsize\normalsize\@xpt{13}\AJW@setskips{\AJW@baseskip}%
    \let\@listi\@listI}
  \let\listsize\normalsize
  \let\figcaptionsize\normalsize
%\fi
\normalsize

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch{\relax}{\mathcal}}
\DeclareRobustCommand*\mit{\@fontswitch{\relax}{\mathnormal}}

% define a smallverbatim 8/10

\begingroup \catcode `|=0 \catcode `[= 1
\catcode`]=2 \catcode `\{=12 \catcode `\}=12
\catcode`\\=12 |gdef|@xsmallverbatim#1\end{smallverbatim}[#1|end[smallverbatim]]
|gdef|@sxsmallverbatim#1\end{smallverbatim*}[#1|end[smallverbatim*]]
|endgroup

\def\@smallverbatim{\trivlist \item\relax
  \if@minipage\else\vskip\parskip\fi
  \leftskip\@totalleftmargin\rightskip\z@skip
  \parindent\z@\parfillskip\@flushglue\parskip\z@skip
  \@@par
  \@tempswafalse
  \def\par{%
    \if@tempswa
      \leavevmode \null \@@par\penalty\interlinepenalty
    \else
      \@tempswatrue
      \ifhmode\@@par\penalty\interlinepenalty\fi
    \fi}%
  \let\do\@makeother \dospecials
  \obeylines \smallverbatim@font \@noligs
  \hyphenchar\font\m@ne
  \everypar \expandafter{\the\everypar \unpenalty}%
}
\def\smallverbatim{\@smallverbatim \frenchspacing\@vobeyspaces \@xsmallverbatim}
\def\endsmallverbatim{\if@newlist \leavevmode\fi\endtrivlist}

\def\smallverbatim@font{\normalfont\indexsize\ttfamily}


%  ****************************************
%  *            PAGE LAYOUT               *
%  ****************************************
%
% All margin dimensions measured from a point one inch from top and side
% of page.
%
% SIDE MARGINS:
%
% \leftskip set in ps@headings
  \oddsidemargin=4.5pc
  \evensidemargin=61pt
  \marginparwidth=54pt%to line up with running head
  \marginparsep=6pt
  \marginparpush 0.5pc% space allowed between marginal notes
  \@mparswitchfalse   % puts all margin notes on one side of page
  \reversemarginpar   % for notes on inside margins
% \normalmarginpar    % for notes on outside margins


%\ifAJW@largesty
%  \setlength\topmargin{34\p@}
%  \setlength\headheight{14\p@}
% %% \setlength\footheight{14\p@}
%  \setlength\footskip{28\p@}
%  \setlength\topskip{11\p@}
%\else
  \setlength\topmargin{29\p@}
  \setlength\headheight{13\p@}
% \setlength\footheight{13\p@}
  \setlength\footskip{26\p@}
  \setlength\topskip{10\p@}
%\fi
  \setlength\headsep{26\p@}
%
% DIMENSION OF TEXT:
%
\newdimen\AJW@measure
\setlength\textheight{43\baselineskip}
\addtolength\textheight{\topskip}

%\ifAJW@largesty
   \AJW@measure 30pc
%\else
%  \AJW@measure 26pc
%\fi
\textwidth\AJW@measure
\ifAJW@landscape
  \textwidth\textheight
  \textheight\AJW@measure
\fi
\setlength\columnsep{1pc}
\setlength\columnseprule{0\p@}
%
% FOOTNOTES
%
\setlength\footnotesep{6.5\p@}
\setlength{\skip\footins}{9\p@ \@plus 4\p@ \@minus 2\p@}% top of first footnote.
%
% FLOATS
%
\setlength\floatsep{13\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\intextsep{13\p@ \@plus 2\p@ \@minus 2\p@}
%\setlength\@maxsep{20\p@}
\setlength\dblfloatsep{13\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
%\setlength\@dblmaxsep{20\p@}
%
% for floats on a separate float page
\setlength\@fptop{0\p@ \@plus 1fil}
\setlength\@fpsep{13\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}

%
% wide floats
\setlength\@dblfptop{0\p@ \@plus 0fil}
\setlength\@dblfpsep{13\p@ \@plus 0fil}
\setlength\@dblfpbot{0\p@ \@plus 3fil}
%
%
% MARGINAL NOTES:
%
\setlength\marginparpush{5\p@}

%  ****************************************
%  *           PARAGRAPHING               *
%  ****************************************
%
\setlength\parskip{0\p@ \@plus .25\p@}
\setlength\parindent{1em}
\listparindent\parindent
\setlength\partopsep{0\p@ \@plus 1\p@}
%
% The following page-breaking penalties are defined
%
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
%
  \clubpenalty=0       % 'Club line'  at bottom of page.
  \widowpenalty=10000  % 'Widow line' at top of page.
% \displaywidowpenalty % Math display widow line.
% \predisplaypenalty   % Breaking before a math display.
% \postdisplaypenalty  % Breaking after a math display.
% \interlinepenalty    % Breaking at a line within a paragraph.
% \brokenpenalty       % Breaking after a hyphenated line.


%    ****************************************
%    *        CHAPTERS AND SECTIONS         *
%    ****************************************
%
% Heading for the \chapter command -- fixed depth chapter headings...
%
\newdimen\AJW@headwidth
\AJW@headwidth \textwidth
\addtolength{\AJW@headwidth}{-\AJW@indent}

\newif\ifAJW@appendix
\newif\ifAJW@oneappendix


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% macros from cup-naz.cls
\newlength\fullwidth
\setlength{\fullwidth}{35pc}

\newdimen\AJW@headwidth
\AJW@headwidth \textwidth
\addtolength{\AJW@headwidth}{-\AJW@indent}
\newbox\chapbox
\newdimen\chapwidth

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% single-contributor books \chapterauthor and \chapteraffil

\newif\if@chapterauthor
  \edef\@chapterauthor{\relax}
\newcommand\chapterauthor[1]{\@chapterauthortrue%
  \gdef\@chapterauthor{\uppercase{#1}}}

\newif\if@chapteraffil
  \edef\@chapteraffil{\relax}
\newcommand\chapteraffil[1]{\@chapteraffiltrue%
  \gdef\@chapteraffil{\\*[4pt]\seriffont \affiliationsize #1\par}}

% multi-contributor books \author and \authoraffil

% this coding only affects multi-contributor Option
\renewcommand\author{%
  \@ifnextchar[{\@authortwo}{\@authorone}%
}

\def\@authortwo[#1]#2{%
  \gdef\@author{\uppercase{#2}}\gdef\@shortauthor{#1}}

\def\@authorone#1{%
  \gdef\@author{\uppercase{#1}}\gdef\@shortauthor{}}

\newif\if@authoraffil
  \edef\@authoraffil{\relax}
\newcommand\authoraffil[1]{\@authoraffiltrue%
  \gdef\@authoraffil{\\*[4pt]\seriffont \affiliationsize #1\par}}

\newcommand\details[1]{\gdef\@details{#1}}
\gdef\@details{}


\def\@makechapterhead#1{%
  {\normalsize
  \parindent 0\p@
  \vspace*{-0.5\baselineskip}%
  \ifnum \c@secnumdepth >\m@ne
   \raisebox{0.5\baselineskip}[0\p@][0\p@]{\mbox{\hspace*{\AJW@indent}}%
    \begin{minipage}[t]{\AJW@headwidth}
    %
    \ifAJW@appendix
      \setbox\chapbox=\hbox{\huge\seriffont Appendix A\enskip
       \bfseries\textcolor{white}{\strut}#1\hspace*{1pc}}%
    \else
      \setbox\chapbox=\hbox{\huge\seriffont
        \bfseries\textcolor{white}\strut #1\hspace*{1pc}}%
    \fi
    %
    \global\chapwidth\wd\chapbox
    \ifdim\wd\chapbox<\AJW@headwidth
% single line chapter titles
    \vspace*{0pt}%
    {\hspace*{-5pc}\begin{tabular*}{\fullwidth}{@{\extracolsep{\fill}}p{5pc}@{}p{30pc}}
    \multicolumn{2}{@{}l@{}}{\includegraphics[width=35pc]{PT1chrule.eps}}\\[-38pt]
    \multicolumn{1}{@{}p{5pc}@{}}{\centering\huge\seriffont\bfseries
      \ifAJW@oneappendix
        \raisebox{0pt}{\textcolor{white}{}}%
      \else
        \raisebox{0pt}{\textcolor{white}{\thechapter}}%
      \fi
      }&\multicolumn{1}%
      {@{\hspace{0.5pc}}p{29pc}@{\hspace{0.5pc}}}%half pica space either side
        {\centering\huge\seriffont\bfseries
      \textcolor{white}{%
      \ifAJW@oneappendix
        {{\huge\seriffont Appendix}\enskip #1}%
      \else
        \ifAJW@appendix
        {{\huge\seriffont Appendix \thechapter}\enskip #1}%
        \else
          {\huge\seriffont #1}%
        \fi
      \fi
      \vphantom{\huge\seriffont gh}}}\\
   \end{tabular*}}
% two line chapter titles
  \else
    \vspace*{0pt}%
    {\hspace*{-5pc}\begin{tabular*}{\fullwidth}{@{\extracolsep{\fill}}p{5pc}@{}p{30pc}}
    \multicolumn{2}{@{}l@{}}{\includegraphics[width=35pc]{PT1chrule.eps}}\\[-51pt]
    \multicolumn{1}{@{}p{5pc}@{}}{\centering\raisebox{-11.8pt}{%
    \huge\seriffont\bfseries\textcolor{white}{\thechapter}}}&
    \multicolumn{1}{@{\hspace{0.5pc}}p{29pc}@{\hspace{0.5pc}}}%half pica space either side
      {\centering\huge\seriffont\bfseries
    \textcolor{white}{%
 %
    \ifAJW@oneappendix
      {{\huge\seriffont\strut Appendix}\enskip\bfseries #1}%
    \else
       \ifAJW@appendix
        {{\huge\seriffont\strut Appendix \thechapter\enskip\seriffont\bfseries #1}}%------------
       \else
          {\huge\seriffont\bfseries\strut #1}%
       \fi
     \fi
     }}%
 %
     \end{tabular*}}
  \fi
     \nobreak
     \par % End of code from cup-naz.cls
     \ifAJW@indentsty \raggedright \else \centering \fi
     \par
     %\ifAJW@largesty \vskip 9\p@ \else
     %\vskip 7\p@
        \ifsp@nningrule
          \vskip 0\p@
          \rule{\textwidth}{0.4pt}\par
          \vskip 7pt
        \else
          \vskip 7\p@
        \fi
     %\fi
%ajw   \huge\rm #1%
     \nobreak
     \par
     \end{minipage}%
    }
  \else
   \raisebox{0.5\baselineskip}[0\p@][0\p@]{\mbox{\hspace*{\AJW@indent}}%
    \begin{minipage}[t]{\AJW@headwidth}
     \ifAJW@indentsty \raggedright \else \centering \fi
     {\huge\bfseries \mbox{}}
     \par
     %\ifAJW@largesty \vskip 9\p@ \else
     %\vskip 7\p@
        \ifsp@nningrule
          \vskip 0\p@
          \rule{\textwidth}{0.4pt}\par
          \vskip 7pt
        \else
          \vskip 7\p@
        \fi
     %\fi
     \huge\rm #1%
     \nobreak
     \par
    \end{minipage}}
  \fi
  \par \nobreak
    \vskip 6.5\baselineskip
  }%
%
  \ifAJW@multisty
    \noindent
      \vspace*{-25pt}% first line of text on line 9
      \vbox to 0pt{%
      \begin{minipage}[t]{28.5pc}%
        \centering{\seriffont\Large
        \vspace*{-41pt}\@author}% author to be 18pt LF below bottom of CT box
      \if@authoraffil\@authoraffil\vspace{42pt}\fi
      \end{minipage}%
      }%
  \@authoraffilfalse
  \else
    \noindent
      \vspace*{-25pt}% first line of text on line 9
      \vbox to 0pt{%
      \begin{minipage}[t]{28.5pc}%
        \centering{\seriffont\Large
        \vspace*{-41pt}%
      \if@chapterauthor\@chapterauthor\fi}% author to be 18pt LF below bottom of CT box
      \if@chapteraffil\@chapteraffil\vspace{47.5pt}\fi
      \end{minipage}%
      }%
  \@chapteraffilfalse\@chapterauthorfalse
  \fi
}

\def\@makeschapterhead#1{%
  {\normalsize
  \parindent 0\p@
  \vspace*{-0.5\baselineskip}%
  \raisebox{0.5\baselineskip}[0\p@][0\p@]{\mbox{\hspace*{\AJW@indent}}%
   \begin{minipage}[t]{\AJW@headwidth}
   \vspace*{0pt}%
   \hspace*{-5pc}\begin{tabular*}{\fullwidth}{@{\extracolsep{\fill}}p{5pc}@{}p{30pc}}
   \multicolumn{2}{@{}l@{}}{\includegraphics[width=35pc]{PT1chrule.eps}}\\[-38pt]
   \multicolumn{1}{@{}p{5pc}@{}}{\centering\huge\seriffont\bfseries
     \raisebox{0pt}{\textcolor{white}{\phantom{\huge 0}}}%
   } &
   \multicolumn{1}{@{\hspace{0.5pc}}p{29pc}@{\hspace{0.5pc}}}%
     {\centering\huge\seriffont\bfseries
   \textcolor{white}{{#1}\vphantom{\huge gh}}}\\
   \end{tabular*}%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%finish here
    \ifAJW@indentsty \raggedright \else \centering \fi
    {\huge\bfseries \mbox{}}
    \par
    %\ifAJW@largesty \vskip 9\p@ \else \vskip 7\p@
       \ifsp@nningrule
         \vskip 0\p@
         \rule{\textwidth}{0.4pt}\par
         \vskip 7pt
       \else
         \vskip 7\p@
       \fi
    %\fi
%ajw \huge\rm #1%
    \nobreak
    \par
%   \@ch@pterauthor
   \end{minipage}}
  \par \nobreak
  \vskip 6.5\baselineskip%
  }%\def\@chapterauthor{}%
}
%
% \secdef{UNSTARCMDS}{STARCMDS} :
%

\newcount\apptmp
\def\chapter{%
   %\c@chapter=\apptmp
   %\stepcounter{chapter}
   \endappendix% in case previous chapter contained appendices
   \clearpage
   \thispagestyle{plain}
   \global\@topnum\z@
   \@afterindentfalse
   \secdef\@chapter\@schapter}

\def\@chapter[#1]#2{%
  \ifnum \c@secnumdepth >\m@ne
    \refstepcounter{chapter}
    \typeout{\@chapapp\space\thechapter.}
    \ifAJW@multisty
      \ifx\@shortauthor\@empty
%
% this code just added
      \ifAJW@oneappendix
        \addcontentsline{toc}{chapter}{\protect\AJW@nonumberline{}#1}
      \else
        \ifAJW@appendix
          \addcontentsline{toc}{chapter}{\protect\AJW@numberline{\thechapter}#1}
        \else
          \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}#1}
        \fi
      \fi
%
%
%%%%%%%%%\addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}#1}
      \else
        \addcontentsline{toc}{chapter}{\protect
        \numberline{\thechapter}#2\hspace{1em}{\itshape \@shortauthor\/}}% #1 changed to #2
                                                                         % so long version of chapter title
                                                                         % appears in toc
      \fi
    \else %if single-contributor
      \ifAJW@oneappendix
        \addcontentsline{toc}{chapter}{\protect\AJW@nonumberline{}#1}
      \else
        \ifAJW@appendix
          \addcontentsline{toc}{chapter}{\protect\AJW@numberline{\thechapter}#1}
        \else
          \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}#1}
        \fi
      \fi
    \fi
  \else
   \addcontentsline{toc}{chapter}{#1}
  \fi
  %\addtocontents{toc}{\mbox{}\protect\pagebreak[0]}%
  \ifAJW@multisty
   \markboth{\@shortauthor}{#1}
   \gdef\AJW@endnotehead{\relax}
  \else
   \chaptermark{#1}
   \gdef\AJW@endnotehead{\@chapapp~\thechapter}%
  \fi
  \if@twocolumn
    \@topnewpage[\@makechapterhead{#2}]
   \else
    \@makechapterhead{#2}%
    \@afterheading
  \fi
  \ifAJW@multisty
   \gdef\@author{}
   \gdef\@shortauthor{}
  \fi
}
\def\@schapter#1{%
 \ifAJW@addtoc
  \ifAJW@indentsty
   \ifdim\AJW@indent>\z@
    \addcontentsline{toc}{schapter}{\protect\numberline{\mbox{}}#1}
   \else
    \addcontentsline{toc}{schapter}{#1}
   \fi
  \else
   \addcontentsline{toc}{schapter}{#1}
  \fi
 \fi
 \ifAJW@multisty
  \markboth{#1}{#1}
  \gdef\AJW@endnotehead{\relax}
 \else
  \chaptermark{#1}
  \gdef\AJW@endnotehead{#1}%
 \fi
 \if@twocolumn
  \@topnewpage[\@makeschapterhead{#1}]
 \else
  \@makeschapterhead{#1}
  \@afterheading
 \fi
 \normalsize
}


%
% \@startsection {NAME}{LEVEL}{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}
%            optional * [ALTHEADING]{HEADING}
%
\ifAJW@indentsty
 \def\section{\@startsection {section}{1}{\AJW@indent}
  {-30\p@ \@plus-7\p@ \@minus-3.5\p@}{8\p@}{\Aheadsize\bfseries\AJW@hangraggedright}}
 \def\subsection{\@startsection{subsection}{2}{\AJW@indent}
  {-20\p@ \@plus-7\p@ \@minus-3.5\p@}{5\p@}{\large\bfseries\AJW@hangraggedright}}
 \def\subsubsection{\@startsection{subsubsection}{3}{\AJW@indent}
  {-13\p@ \@plus-7\p@ \@minus-3.5\p@}{.00001\p@}{\normalsize\bfseries\AJW@hangraggedright}}
 \def\paragraph{\@startsection{paragraph}{4}{\AJW@indent}
  {13\p@ \@plus7\p@ \@minus3.5\p@}{-0.5em}{\normalsize\bfseries\AJW@raggedright}}
 \def\subparagraph{\@startsection{subparagraph}{5}{\AJW@indent}
  {13\p@ \@plus7\p@ \@minus3.5\p@}{-0.5em}{\normalsize\itshape\AJW@raggedright}}
\else
 %A
 \newcommand\section{\@startsection {section}{1}{\z@}
   {23pt plus6pt minus2pt}{12pt}%
   {\Aheadsize\seriffont\bfseries\centering
   }}
 %B
 \newcommand\subsection{\@startsection{subsection}{2}{\z@}
   {12pt plus6pt minus2pt}{6pt}%
   {\Bheadsize\semiboldseriffont\centering
   }}
 %C
 \newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
   {18.5pt plus6pt minus2pt}{6.5pt}%
   {\Bheadsize\semiboldseriffont\centering
   }}
 %D
 \newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
   {18.5pt plus6pt minus2pt}{6.5pt}%
   {\Bheadsize\seriffont\centering
   }}
 %E
 \newcommand\subparagraph{\@startsection{subparagraph}{5}{\z@}%
   {-13pt plus-3pt minus-2pt}{.00001\p@}%
   {\normalsize\seriffont\bfseries\AJW@raggedright
   }}
 %X
 \newcommand\xhead{\@startsection{xhead}{6}{\z@}%
   {-10pt plus-3pt minus-2pt}{3\p@}%
   {\xheadsize\semiboldseriffont\AJW@raggedright
   }}
 \newcounter{xhead}
 \newcommand\xheadmark[1]{}
 \newcommand\l@xhead{\@dottedtocline{1}{1.5em}{2.3em}}
%
 %Y
 \newcommand\yhead{\@startsection{yhead}{6}{\z@}%
   {-10pt plus-3pt minus-2pt}{3\p@}%
   {\yheadsize\seriffont\AJW@raggedright
   }}
 \newcounter{yhead}
 \newcommand\yheadmark[1]{}
 \newcommand\l@yhead{\@dottedtocline{1}{1.5em}{2.3em}}
 \fi

\def\AJW@hangraggedright{\rightskip\@flushglue \let\\=\newline}
\def\AJW@raggedright{\rightskip\@flushglue \let\\=\@centercr
 \parindent=\AJW@indent}
%
% Default initializations of \...mark commands
%
%\def\chaptermark#1{}
\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}

%
% APPENDIX
%
\ifAJW@multisty
%
  \def\appendix{%
    \apptmp=\c@chapter %save chapter number
    \par
    \AJW@appendixtrue
    \renewcommand\section{\@startsection {section}{1}{\z@}
     {23pt plus6pt minus2pt}{12pt}%
     {\Aheadsize\seriffont\bfseries\centering\appendixname~
    }}
    \setcounter{section}{0}
    \@addtoreset{figure}{subsection}
    \setcounter{figure}{0}
    \setcounter{table}{0}
% set tocdepth to chapter heads only whether multi-contributor or single-contributor
    %\addtocontents{toc}{\protect\setcounter{tocdepth}{1}}
    \def\@chapapp{\appendixname}
    \def\thechapter {\Alph{chapter}}
    \def\thesection {\Alph{section}}
    \def\thesubsection {\Alph{section}.\arabic{subsection}}
    \def\thetable   {\Alph{section}.\@arabic\c@table}
    \def\thefigure  {\Alph{section}.\@arabic\c@figure}
    \def\theequation{\Alph{section}.\arabic{equation}}
}

% and only one appendix
  \def\oneappendix{\AJW@oneappendixtrue\appendix
    \def\thesection {}% remove the 'A' from Appendix heading
  }
\else
%
% single version
\def\appendix{\par\clearpage
 \AJW@appendixtrue
 \setcounter{chapter}{0}
 \setcounter{section}{0}
% set tocdepth to chapter heads only whether multi-contributor or single-contributor
 \addtocontents{toc}{\protect\setcounter{tocdepth}{0}}
 \def\@chapapp{\appendixname}
 \def\thechapter {\Alph{chapter}}
 \def\thesection {\thechapter.\arabic{section}}
 \def\thetable   {\thechapter.\@arabic\c@table}
 \def\thefigure  {\thechapter.\@arabic\c@figure}
 \def\theequation{\thechapter.\arabic{equation}}
}
%
% and only one appendix
    \def\oneappendix{\AJW@oneappendixtrue\appendix}
\fi


%%%%%%%%%%%%%%%%%%%%%%
\ifAJW@multisty
  \def\endappendix{\AJW@appendixfalse
  \renewcommand\section{\@startsection {section}{1}{\z@}
    {23pt plus6pt minus2pt}{12pt}%
    {\Aheadsize\seriffont\bfseries\centering
    }}
    \def\thechapter {\arabic{chapter}}
    \def\thesection {\thechapter.\arabic{section}}
    \def\thesubsection {\thesection.\arabic{subsection}}
    \def\thetable{\thechapter.\@arabic\c@table}
    \def\thefigure{\thechapter.\@arabic\c@figure}
    \def\theequation{\arabic{chapter}.\arabic{equation}}%
  }
\else
  \def\endappendix{\relax}
\fi

%    ****************************************
%    *                LISTS                 *
%    ****************************************
%
\ifAJW@indentsty
 \ifdim \AJW@indent=\z@ \setlength\leftmargini{1em}
  \else \setlength\leftmargini{\AJW@indent}
 \fi \else \setlength\leftmargini{1em}
\fi
\setlength\leftmarginii{1em}
\setlength\leftmarginiii{1em}
\setlength\leftmarginiv{1em}
\setlength\leftmarginv{1em}
\setlength\leftmarginvi{1em}
\leftmargin\leftmargini
\setlength\labelsep{0.5em}
\labelwidth\leftmargini \advance\labelwidth-\labelsep
\def\@listI{\leftmargin\leftmargini
 \parsep    \z@ \@plus 2\p@
 \partopsep \z@ \@plus 1\p@
 \topsep  6.5\p@ \@plus 3.25\p@ \@minus 1\p@%
 \itemsep   \z@ \@plus.1\p@
 \listparindent \parindent
}
\let\@listi\@listI
\@listi
\def\@listii{\leftmargin\leftmarginii
 \labelwidth\leftmarginii \advance\labelwidth-\labelsep
 \parsep    \z@ \@plus 2\p@
 \partopsep \z@ \@plus 1\p@
 \topsep 3.25\p@ \@plus .5\p@ \@minus 1\p@%
 \itemsep   \z@ \@plus.1\p@
}
\def\@listiii{\leftmargin\leftmarginiii
 \labelwidth\leftmarginiii \advance\labelwidth-\labelsep
 \parsep    \z@ \@plus 2\p@
 \partopsep \z@ \@plus 1\p@
 \topsep 3.25\p@ \@plus .5\p@ \@minus 1\p@%
 \itemsep   \z@ \@plus.1\p@
}
\def\@listiv{\leftmargin\leftmarginiv
 \labelwidth\leftmarginiv \advance\labelwidth-\labelsep}
\def\@listv{\leftmargin\leftmarginv
 \labelwidth\leftmarginv  \advance\labelwidth-\labelsep}
\def\@listvi{\leftmargin\leftmarginvi
 \labelwidth\leftmarginvi \advance\labelwidth-\labelsep}

%    ****************************************
%    *           LIST ENVIRONMENTS          *
%    ****************************************
%
%
% ENUMERATE -- with optional argument to set left margin
% DEFAULT is arabic numbering with full point

\def\makeRRlabel#1{\hss\llap{#1}}
\def\makeRRlabeldot#1{\hss\llap{#1.}}

\newbox\@dotbox
\setbox\@dotbox\hbox{.}%
\def\enumerate{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
 \fi
 \@ifnextchar [{\@enumeratetwo}{\@enumerateone}%
}
\def\@enumeratetwo[#1]{%
 \list{\csname label\@enumctr\endcsname}%
  {\settowidth\labelwidth{#1}%
   \leftmargin\labelwidth \advance\leftmargin\labelsep
   \usecounter{\@enumctr}%
   \let\makelabel\makeRRlabel}%
}
\def\@enumerateone{%\listsize
 \list{\csname label\@enumctr\endcsname}%
  {\usecounter{\@enumctr}%
   \ifAJW@indentsty
    \ifdim \AJW@indent=\z@ \leftmargin 1em % overrides \leftmargini,ii,iii...
                 \fi \else %\leftmargin 1em
                           \leftmargin 0.5em% width of a digit
                           \advance\leftmargin\labelsep
                           \advance\leftmargin\wd\@dotbox
   \fi
   \let\makelabel\makeRRlabeldot}}

\newcommand\labelenumi{\theenumi}
\newcommand\labelenumii{\theenumii}
\newcommand\labelenumiii{\theenumiii}
\newcommand\labelenumiv{\theenumiv}

\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
%
% ITEMIZE
%
\def\itemize{\ifnum \@itemdepth >3 \@toodeep \else
  \advance\@itemdepth \@ne
  \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
%  \listsize
  \list{\csname\@itemitem\endcsname}%
   {%
   \itemindent -1em%
   \leftmargin 2em%
   \def\makelabel##1{\hss\llap{##1}}}%
  \fi
}
\newcommand\labelitemi{$\bullet$}
\newcommand\labelitemii{\bfseries --}
\newcommand\labelitemiii{$\circ$}
\newcommand\labelitemiv{$\diamond$}
%
% DESCRIPTION
%
\newcommand*\descriptionlabel[1]{\hspace\labelsep\bfseries #1}
\def\description{%\listsize
 \list{}{\labelwidth\z@
  \leftmargin 3pc
  \itemindent-\leftmargin
  \let\makelabel\descriptionlabel
 }}
\let\enddescription\endlist
%

%    ****************************************
%    *         OTHER ENVIRONMENTS           *
%    ****************************************
%
% VERSE
%
\def\verse{\let\\=\@centercr
  \list{}{\itemsep\z@ \itemindent -1em\listparindent \itemindent
          \rightmargin\z@
          \leftmargin\AJW@indent\advance\leftmargin 1em}\item[]\smallish}
\let\endverse=\endlist
%
% QUOTATION
%
\def\quotation{\list{}{\leftmargin\AJW@indent\rightmargin\z@}\item[]\smallish}
\let\endquotation=\endlist
%
% QUOTE
%
\def\quote{\list{}{\leftmargin\AJW@indent \rightmargin\z@}\item[]\smallish}
\let\endquote=\endlist
%
% INDENTED (for use in title pages)
%
\def\indented{\list{}{\leftmargin\AJW@indent
 \listparindent\z@\rightmargin\z@}\raggedright\item[]}
\let\endindented=\endlist
%
% ARRAY AND TABULAR
%
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.5\p@}
\setlength\doublerulesep{2\p@}
%
% TABBING
%
\tabbingsep\labelsep
%
% FRAMEBOX
%
\setlength\fboxsep{5\p@}
\setlength\fboxrule{.5\p@}
%
% THEOREMS
%
% let theorems be handled by amsthm.sty
%
%
% PROOFs
%
% let proofs be handled by amsthm.sty

\newenvironment{proof*}[1][\proofname]{\par
  \pushQED{}%{\qed}%
  \normalfont \topsep 10pt plus 3pt minus 1pt\relax
  \trivlist
  \item[\hskip\labelsep
%   \itshape
    \xheadsize\semiboldseriffont
%   #1\@addpunct{.}]\ignorespaces
    #1\enskip]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}

%    ****************************************
%    *        CHAPTERS AND SECTIONS         *
%    ****************************************
%
% DEFINE COUNTERS:
%
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]

\def\thepart          {\Roman{part}}
\def\thechapter       {\arabic{chapter}}
\def\thesection       {\thechapter.\arabic{section}}
\def\thesubsection    {\thesection.\arabic{subsection}}
\def\thesubsubsection {\thesubsection .\arabic{subsubsection}}
\def\theparagraph     {\thesubsubsection.\arabic{paragraph}}
\def\thesubparagraph  {\theparagraph.\arabic{subparagraph}}
\def\@chapapp{\chaptername}

%    ****************************************
%    *                 PARTS                *
%    ****************************************
%

\newcommand\ls{\kern.05em} % for letterspacing; .15em is a hair less than thinspace
\newcommand\ns{\kern.35em} % for name-spacing

\newif\ifp@rtquote
\newcommand\partquote[2]{%
  \p@rtquotetrue\def\p@rtquote{%
  \vspace{35pt}\trivlist\leftskip 2pc\rightskip\leftskip
  \item\normalsize\normalfont #1\\
  \noindent\hbox to 26pc{\hfill #2}\endlist}}

\def\part{\cleardoublepage
 \thispagestyle{empty}
 \if@twocolumn
   \onecolumn
   \@tempswatrue
  \else \@tempswafalse
 \fi
 \secdef\@part\@spart
}

\def\@part[#1]#2{\ifnum \c@secnumdepth >-2\relax
  \refstepcounter{part}
  \ifAJW@indentsty
   \ifdim\AJW@indent>\z@
    \ifAJW@multisty
    \addcontentsline{toc}{part}{{Part~\thepart}%
    \hspace*{6pt}{#1}}
    \else
    \addcontentsline{toc}{part}{{Part~\thepart}%
    \hspace*{6pt}{#1}}
    \fi
   \else
    \addcontentsline{toc}{part}{{Part~\thepart}%
    \hspace*{6pt}{#1}}
   \fi
  \else
    \addcontentsline{toc}{part}{{Part~\thepart}%
    \hspace*{6pt}{#1}}%
  \fi
  \else
    \addcontentsline{toc}{part}{#1}
  \fi
  \@mkboth{}{}
  {\ifAJW@indentsty \raggedright \else \centering \fi
  \vspace*{-\baselineskip}
   \ifnum \c@secnumdepth >-2\relax
   \vspace*{12pt}%top of box positioned on first line of text
   \hspace*{-5pc}%
   \begin{tabular*}{\fullwidth}{@{\extracolsep{\fill}}p{5pc}@{}p{30pc}}
   \multicolumn{2}{@{}l@{}}{\includegraphics[width=35pc]{PT1chrule.eps}}\\[-39pt]%
     &\multicolumn{1}{@{\hspace{0.5pc}}p{29pc}@{\hspace{0.5pc}}}{%
       \centering\quitehuge\seriffont\bfseries\textcolor{white}%
       {\centering\partname\ \thepart}}\\[49pt]
   \end{tabular*}
   \nobreak
   %{
   \centering\superhuge\seriffont\textcolor{fortycolor}{%
   \MakeUppercase{#1}\vphantom{gh}}%}\\
    \par
    \ifp@rtquote\p@rtquote\fi
    %\ifAJW@largesty \vskip 9\p@ \else
      %\ifsp@nningrule
%ajw     \vskip 0\p@
%ajw     \rule{\textwidth}{0.4pt}\par
%ajw     \vskip 7pt
      %\else
      %  \vskip 20\p@% part title on 4th line
      %\fi
    %\fi
   \else
    \par
    \vskip 2.5\baselineskip
  \fi
%ajw \huge \rm \hspace*{\AJW@indent}\MakeUppercase{#2}\par
  }%
  \p@rtquotefalse\@endpart
}

% rule appears in spart title whether [multi] option is used or not
\def\@spart#1{{\ifAJW@indentsty \raggedright \else \centering \fi
  \vspace*{-\baselineskip}
  \par
% \vskip 19.5\p@
  \vskip 13\p@
      %\ifsp@nningrule
         \vskip 0\p@
         \rule{\textwidth}{0.4pt}\par
         \vskip 7pt
      %\else
      %  \vskip 20\p@% part title on 4th line
      %\fi
  \huge \rm \MakeUppercase{#1}\par}
\@endpart}
%
% \@endpart finishes the part page
%
\def\@endpart{\cleardoublepage}

%    ****************************************
%    *                EXAMPLES              *
%    ****************************************
%
%
%\newenvironment{example}%
%  {\par\addvspace{2pt}\noindent\rule{\textwidth}{1pt}\vspace*{-6pt}\begin{example-norules}}%
%  {\end{example-norules}\vspace*{-10pt}\noindent\rule{\textwidth}{1pt}\par\addvspace{5pt}}
%
% coding from EngC, modified
%
% Examples numbered by chapter
\newcounter{example@bean}[chapter]
\def\p@example@bean{\thechapter.}% add section number to cross-references

\newcount\tmp
\def\examplelist{\@ifnextchar [{\@examplelisttwo}{\@examplelistone}}
% use \@exampleliston - heading 'Example'

%\newcommand\solution{\vspace{6.5pt}\par\noindent{\seriffont Solution}\\*}
 \newcommand\solution[1]{\vspace{10pt}\par\noindent{\seriffont #1}\\*[3pt]}

\def\@examplelistone{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\examplelabel##1{\xheadsize\semiboldseriffont Example \thechapter.##1\hss}
    \vspace{\baselineskip}%
    \par\noindent\rule{\textwidth}{1pt}%
    \par\nobreak
    \@afterheading
     \list{%
       \arabic{example@bean}}%
       {\tmp=\c@example@bean
        \usecounter{example@bean}%
        \c@example@bean=\tmp
        \leftmargin 0pc%
        \labelwidth \leftmargin
        \labelsep 1em%ajw
        \addtolength{\labelwidth}{-\labelsep}%
        \itemsep\baselineskip
        \partopsep=0pt plus0pt minus0pt%
        \let\makelabel\examplelabel}%
  \fi}

\def\endexamplelist{\\*[-4pt]\noindent\rule{\textwidth}{1pt}\endlist\addvspace{17pt}}

\def\@examplelisttwo[#1]{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\examplelabel##1{\xheadsize\semiboldseriffont #1 \thechapter.##1\hss}
    \vspace{\baselineskip}%
    \par\noindent\rule{\textwidth}{1pt}%
    \par\nobreak
    \@afterheading
     \list{\arabic{example@bean}}%
       {\tmp=\c@example@bean%
        \usecounter{example@bean}%
        \c@example@bean=\tmp%
        \leftmargin 0pc%
        \labelwidth \leftmargin
        \labelsep 1em%ajw
        \addtolength{\labelwidth}{-\labelsep}%
        \itemsep\baselineskip
        \partopsep=0pt plus0pt minus0pt%
        \let\makelabel\examplelabel}%
\fi}

%\let\endexamplelist=\endlist


% Examples numbered by chapter
%\newcounter{example@beanbychapter}[chapter]
%\def\p@example@beanbychapter{\thechapter.}% add chapter number to cross-references

%\def\examplelistbychapter{\@ifnextchar [{\@examplelisttwobychapter}{\@examplelistonebychapter}}
%\def\@examplelistonebychapter{\ifnum \@enumdepth >3 \@toodeep \else
%  \advance\@enumdepth \@ne
%  \def\examplelabel##1{\rm \thechapter.##1\hss}
%  \list{\arabic{example@beanbychapter}}%
%       {\usecounter{example@beanbychapter}%
%        \leftmargin 2pc
%        \labelwidth \leftmargin
%        \addtolength{\labelwidth}{-\labelsep}
%  \let\makelabel\examplelabel}%
%\fi}
%\def\@examplelisttwobychapter[#1]{\ifnum \@enumdepth >3 \@toodeep \else
%  \advance\@enumdepth \@ne
%  \def\examplelabel##1{\bf #1 \thechapter.##1\hss}
%  \list{\arabic{example@beanbychapter}}%
%      {\tmp=\c@example@beanbychapter%new line
%       \usecounter{example@beanbychapter}%
%        \c@example@beanbychapter=\tmp%new line
%        \leftmargin 2pc
%        \labelwidth \leftmargin
%        \addtolength{\labelwidth}{-\labelsep}
%  \let\makelabel\examplelabel}%
%\fi}

%\let\endexamplelistbychapter=\endlist

%\def\examples{\section*{Examples}%
%  \addcontentsline{toc}{section}{Examples}
%  \ifAJW@multisty\else\markright{Examples}\fi
%  \examplelistbychapter
%  %\setcounter{equation}{0}
%  %\def\theequation{E\thechapter.\arabic{equation}}
%}
%\let\endexamples=\endexamplelistbychapter
%\let\example=\item

%    ****************************************
%    *               EXERCISES              *
%    ****************************************

% Exercises numbered by section
\newcounter{ex@bean}[section]
\def\p@ex@bean{\thesection.}% add section number to cross-references

\newcount\tmp
\def\exerciselist{\@ifnextchar [{\@exerciselisttwo}{\@exerciselistone}}
\def\@exerciselistone{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\exerciselabel##1{\floatingboxsize\semiboldseriffont \thesection.##1\hss}
  \list{\arabic{ex@bean}}%
       {\usecounter{ex@bean}%
        \leftmargin 2pc
        \labelwidth \leftmargin
        \labelsep 1em%ajw
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\exerciselabel}%
\fi}
\def\@exerciselisttwo[#1]{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\exerciselabel##1{\floatingboxsize\semiboldseriffont #1 \thesection.##1\hss}
  \list{\arabic{ex@bean}}%
       {\tmp=\c@ex@bean%new line
        \usecounter{ex@bean}%
        \c@ex@bean=\tmp%new line
        \leftmargin 2pc
        \labelwidth \leftmargin
        \labelsep 1em%ajw
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\exerciselabel}%
\fi}
\let\endexerciselist=\endlist

% Exercises numbered by chapter
\newcounter{ex@beanbychapter}[chapter]
\def\p@ex@beanbychapter{\thechapter.}% add chapter number to cross-references

\def\exerciselistbychapter{\@ifnextchar [{\@exerciselisttwobychapter}{\@exerciselistonebychapter}}
\def\@exerciselistonebychapter{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\exerciselabel##1{\floatingboxsize\semiboldseriffont\thechapter.##1\hss}
  \list{\arabic{ex@beanbychapter}}%
       {\usecounter{ex@beanbychapter}%
        \leftmargin 2pc
        \labelwidth \leftmargin
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\exerciselabel}%
\fi}
\def\@exerciselisttwobychapter[#1]{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\exerciselabel##1{\floatingboxsize\semiboldseriffont #1 \thechapter.##1\hss}
  \list{\arabic{ex@beanbychapter}}%
       {\tmp=\c@ex@beanbychapter%new line
        \usecounter{ex@beanbychapter}%
        \c@ex@beanbychapter=\tmp%new line
        \leftmargin 2pc
        \labelwidth \leftmargin
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\exerciselabel}%
\fi}

\let\endexerciselistbychapter=\endlist


\def\exercises{\section*{Exercises}%
  %\addcontentsline{toc}{section}{Exercises}
  \ifAJW@multisty\else\markright{Exercises}\fi
  \exerciselistbychapter
  %\setcounter{equation}{0}
  %\def\theequation{E\thechapter.\arabic{equation}}
}
\let\endexercises=\endexerciselistbychapter
\let\exercise=\item


%    ****************************************
%    *               PROBLEMS               *
%    ****************************************

% this is an exact copy of exercises
% Problems numbered by section
\newcounter{pr@bean}[section]
\def\p@pr@bean{\thesection.}% add section number to cross-references

%\newcount\tmp
\def\problemlist{\@ifnextchar [{\@problemlisttwo}{\@problemlistone}}
\def\@problemlistone{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\problemlabel##1{\floatingboxsize\semiboldseriffont \thesection.##1\hss}
  \list{\arabic{pr@bean}}%
       {\usecounter{pr@bean}%
        \leftmargin 2pc
        \labelwidth \leftmargin
        \labelsep 1em%ajw
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\problemlabel}%
\fi}
\def\@problemlisttwo[#1]{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\problemlabel##1{\floatingboxsize\semiboldseriffont #1 \thesection.##1\hss}
  \list{\arabic{pr@bean}}%
       {\tmp=\c@pr@bean%new line
        \usecounter{pr@bean}%
        \c@pr@bean=\tmp%new line
        \leftmargin 2pc
        \labelwidth \leftmargin
        \labelsep 1em%ajw
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\problemlabel}%
\fi}
\let\endproblemlist=\endlist

% Problems numbered by chapter
\newcounter{pr@beanbychapter}[chapter]
\def\p@pr@beanbychapter{\thechapter.}% add chapter number to cross-references

\def\problemlistbychapter{\@ifnextchar [{\@problemlisttwobychapter}{\@problemlistonebychapter}}
\def\@problemlistonebychapter{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\problemlabel##1{\floatingboxsize\semiboldseriffont\thechapter.##1\hss}
  \list{\arabic{pr@beanbychapter}}%
       {\usecounter{pr@beanbychapter}%
        \leftmargin 2pc
        \labelwidth \leftmargin
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\problemlabel}%
\fi}
\def\@problemlisttwobychapter[#1]{\ifnum \@enumdepth >3 \@toodeep \else
  \advance\@enumdepth \@ne
  \def\problemlabel##1{\floatingboxsize\semiboldseriffont #1 \thechapter.##1\hss}
  \list{\arabic{pr@beanbychapter}}%
       {\tmp=\c@pr@beanbychapter%new line
        \usecounter{pr@beanbychapter}%
        \c@pr@beanbychapter=\tmp%new line
        \leftmargin 2pc
        \labelwidth \leftmargin
        \addtolength{\labelwidth}{-\labelsep}
  \let\makelabel\problemlabel}%
\fi}

\let\endproblemlistbychapter=\endlist


\def\problems{\section*{Problems}%
  %\addcontentsline{toc}{section}{Problems}
  \ifAJW@multisty\else\markright{Problems}\fi
  \problemlistbychapter
  %\setcounter{equation}{0}
  %\def\theequation{E\thechapter.\arabic{equation}}
}
\let\endproblems=\endproblemlistbychapter
\let\problem=\item

%    ****************************************
%    *                SUMMARY               *
%    ****************************************

\newenvironment{summary}{%
  \section{Summary}
  \floatingboxsize\seriffont
 }{}



%    ****************************************
%    *         TABLE OF CONTENTS, ETC.      *
%    ****************************************
%
\newdimen\AJW@tocindent
%\ifAJW@largesty \AJW@tocindent=65\p@ \else
  \AJW@tocindent=60\p@
%\fi
% \def\numberline#1{\hbox to\@tempdima{#1\hfil}}
\def\AJW@numberline#1{\hbox to 1.15\AJW@tocindent {{\appendixname\ #1\hfil}}}%1.15
\def\AJW@nonumberline#1{\hbox to 56pt{{\appendixname\hfil}}}%56pt matches section heads
\ifAJW@indentsty \ifdim\AJW@indent>\z@
 \def\numberline#1{\hbox to\@tempdima{\hfil#1\hspace*{0.5em}}}
 \def\AJW@numberline#1{\hspace*{\@tempdima}%
  \hbox to 1.15\AJW@tocindent {{\itshape \appendixname\ #1\hfil}}}
\fi \fi
%
\newif\ifAJW@addtoc        \AJW@addtocfalse
\newif\ifAJW@firstTOCentry \AJW@firstTOCentrytrue
\newdimen\AJW@toclmarg
%\newdimen\AJW@tocfloatlmarg
\ifAJW@multisty
 \setcounter{tocdepth}{0}
 \AJW@toclmarg 16\p@
% \AJW@tocfloatlmarg 16\p@
\else
 \setcounter{tocdepth}{1}
 \AJW@toclmarg 13\p@
% \AJW@tocfloatlmarg 13\p@
\fi
\ifAJW@indentsty %and
  \ifdim\AJW@indent>\z@
    \AJW@toclmarg \AJW@indent
%    \AJW@tocfloatlmarg \AJW@indent
 \fi
\fi
\def\@pnumwidth{2.55em}%{1.55em}
\def\@tocrmarg {3.55em}%{2.55em}
\def\@dotsep{4.5}
%
% TABLEOFCONTENTS
%
\newcommand\tableofcontents{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \AJW@addtocfalse
 \chapter*{\contentsname}
 \@starttoc{toc}
 \AJW@addtoctrue
 \if@restonecol\twocolumn\fi
}

\newcommand*\l@part[2]{\pagebreak[3]%
 \vspace{2\baselineskip}%
 \addpenalty{-\@highpenalty}%
 \begingroup
  \parindent \z@ \rightskip \@pnumwidth
  \parfillskip -\@pnumwidth
  \leavevmode
  %\ifAJW@multisty \else \hspace*{\AJW@toclmarg}\fi
  \ifAJW@firstTOCentry
     \hfill{\parttitlesintoc\seriffont\bfseries #1}\hfill
       \llap{\hbox to\@pnumwidth{\hss {\normalfont\itshape page\/} #2}}\par
    \global\AJW@firstTOCentryfalse
  \else
    \hfill{\parttitlesintoc\seriffont\bfseries #1}\hfill
    \llap{\hbox to\@pnumwidth{\hss #2}}\par%
  \fi
  \nobreak
 \endgroup
}

\newcommand*\l@chapter[2]{\pagebreak[3]%
 \vspace{\baselineskip}%
 \@tempdima \AJW@toclmarg
 \begingroup
  \parindent \z@  \rightskip \@pnumwidth  \parfillskip -\@pnumwidth
  \leavevmode
  \advance \leftskip \@tempdima  \hskip -\leftskip
  \ifAJW@firstTOCentry
   %\hskip -20pt%
   {\semiboldseriffont #1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss {\normalfont\itshape page\/} #2}\par
   \global\AJW@firstTOCentryfalse
  \else
   {\semiboldseriffont #1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
  \fi
 \endgroup
}

\newcommand*\l@schapter[2]{\pagebreak[3]% now indented
 \@tempdima \AJW@toclmarg
 \begingroup
  \parindent \z@  \rightskip \@pnumwidth  \parfillskip -\@pnumwidth
  \leavevmode
  \advance \leftskip \@tempdima \hskip -\leftskip
  \ifAJW@firstTOCentry
   %\hskip -20pt%
   {\itshape #1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss {\itshape page\/} #2}\par
   \global\AJW@firstTOCentryfalse
  \else
   {\itshape #1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
  \fi
 \endgroup
}


% remove page number from contributor page
\newcommand*\l@contrib[2]{\pagebreak[3]
 \@tempdima \AJW@toclmarg
 \begingroup
% \parindent \z@  \rightskip \@pnumwidth  \parfillskip -\@pnumwidth
  \parindent \z@  \rightskip \z@
  \leavevmode
  \advance \leftskip \@tempdima  \hskip -\leftskip
  \ifAJW@firstTOCentry
   {#1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss {\itshape page\/} #2}\par
   \global\AJW@firstTOCentryfalse
  \else
%  {#1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
   {#1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss }\par
  \fi
 \endgroup
}

% instead of using \AJW@toclmarg, revert to 13pt throughout
% (sections not listed in multi-contributor designs)

%\newcommand*\l@section      {\@dottedtocline{1}{28pt}{28pt}}
%\newcommand*\l@subsection   {\@dottedtocline{2}{56pt}{34pt}}
\newcommand*\l@section      {\@dottedtocline{1}{13pt}{25pt}}
\newcommand*\l@subsection   {\@dottedtocline{2}{38pt}{31pt}}

% these are not used
\newcommand*\l@subsubsection{\@dottedtocline{3}{\z@}{\AJW@toclmarg}}
\newcommand*\l@paragraph    {\@dottedtocline{4}{\z@}{\AJW@toclmarg}}
\newcommand*\l@subparagraph {\@dottedtocline{5}{\z@}{\AJW@toclmarg}}
%
% LIST OF FIGURES
%
\newcommand\listoffigures{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \bgroup
 \AJW@addtocfalse% different chapter title and toc entries
   \chapter*{\listfigurename}
   \addcontentsline{toc}{schapter}{List of illustrations}
 \normalsize
 \setcounter{tocdepth}{1}
 \@starttoc{lof}
 \egroup
 \if@restonecol\twocolumn\fi
}

%%%\newcommand*\l@figure{\@dottedtocline{1}{0pt}{28pt}}% same as sections

% listing for figures (repeat of schapter)
\newcommand*\l@figure[2]{\pagebreak[3]
 \@tempdima 28pt%\AJW@toclmarg
 \begingroup
  \parindent \z@  \rightskip \@pnumwidth  \parfillskip -\@pnumwidth
  \leavevmode
  \advance \leftskip \@tempdima  \hskip -\leftskip
%
  \ifAJW@firstTOCentry
    {#1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss {\itshape page\/} #2}\par
    \global\AJW@firstTOCentryfalse
  \else
    {#1}\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
 \fi
%
 \endgroup
}

% LIST OF TABLES
%
\newcommand\listoftables{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \bgroup
 \AJW@addtocfalse% different chapter title and toc entries
   \chapter*{\listtablename}
   \addcontentsline{toc}{schapter}{List of tables}
   \vspace*{3pt}%
 \normalsize
 \setcounter{tocdepth}{1}
 \@starttoc{lot}
 \egroup
 \if@restonecol\twocolumn\fi
}

\let\l@table\l@figure

% LIST OF FLOATINGBOXES
%
\newcommand\listoffloatingboxes{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \bgroup
 \AJW@addtocfalse% different chapter title and toc entries
   \chapter*{\listfloatingboxname}
   \addcontentsline{toc}{schapter}{List of boxes}
   \vspace*{3pt}%
 \normalsize\baselineskip 10pt% correction factor - 3pt added after captions
 \setcounter{tocdepth}{1}
 \@starttoc{lofb}
 \egroup
 \if@restonecol\twocolumn\fi
}

\let\l@floatingbox\l@figure

% LIST OF CONTRIBUTORS
\newcommand\contributor[1]{\addcontentsline{loc}{contrib}{#1}}
\newcommand\affiliation[1]{\quad {\em #1}}

\newcommand\listofcontributors{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \bgroup
 \AJW@addtocfalse% different chapter title and toc entries
   \chapter*{\listcontribname}
   \addcontentsline{toc}{schapter}{List of contributors}
 \normalsize
 \setcounter{tocdepth}{1}
 \@starttoc{loc}
 \egroup
 \if@restonecol\twocolumn\fi
}

\newcommand\editedlistofcontributors{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn\fi
 \bgroup
 \AJW@addtocfalse% different chapter title and toc entries
   \chapter*{\listcontribname}
   \addcontentsline{toc}{schapter}{List of contributors}
 \normalsize
 \setcounter{tocdepth}{1}
 %\@starttoc{loc}
 \input{editedloc}
 \egroup
 \if@restonecol\twocolumn\fi
}

%
% \@dottedtocline{LEVEL}{INDENT}{NUMWIDTH}{TITLE}{PAGE} without the dots
%

\def\@dottedtocline#1#2#3#4#5{% For toc alone, all designs assume \AJW@indent=0pc
 \ifnum #1>\c@tocdepth
 \else
  \vskip \z@ \@plus .2\p@
  {\leftskip #2\relax
   \rightskip \@tocrmarg \@plus2em % for ragged right entries
   \parfillskip -\rightskip
   \parindent #2\relax
   \@afterindenttrue
   \interlinepenalty\@M
   \leavevmode
   \@tempdima #3\relax \advance\leftskip \@tempdima \hbox{}\hskip -\leftskip
   #4\nobreak
   \hfill \nobreak
   \ifAJW@firstTOCentry
    \hbox to\@pnumwidth{\hfil\rm {\itshape page\/} #5}\par
    \global\AJW@firstTOCentryfalse
   \else
    \hbox to\@pnumwidth{\hfil\rm #5}\par
   \fi}%
 \fi
}



%    ****************************************
%    *             BIBLIOGRAPHY             *
%    ****************************************
%

% from book.cls
\newdimen\bibindent
\setlength\bibindent{1em}
\newenvironment{thebibliography}[1]
     {\chapter*{\refname}%
      \@mkboth{\refname}{\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\normalsize\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty


%    ****************************************
%    *              THE INDEX               *
%    ****************************************
%
% THE THEINDEX ENVIRONMENT
%

\def\index@prologue{}
\def\@indextype{}

\newif\if@restonecol
\newenvironment{theindex}{\the@index{\indexname}}{\par\endthe@index}
%\newenvironment{theauthorindex}{\the@index{Author index}}{\par\endthe@index}
%\newenvironment{thesubjectindex}{\the@index{Subject index}}{\par\endthe@index}

\newcommand\the@index[1]{\@restonecoltrue
 \if@twocolumn\@restonecolfalse\fi
 \columnseprule \z@
 \columnsep 1pc
 \twocolumn[\vspace*{10\p@}\mbox{}\par\@makeschapterhead{#1}]
 \@mkboth{#1}{#1}
 \typeout{#1.}
 \ifAJW@indentsty
  \ifdim\AJW@indent>\z@
   \addcontentsline{toc}{schapter}{\protect\numberline{\mbox{}}#1}
  \else
   \addcontentsline{toc}{schapter}{#1}
  \fi
 \else
  \addcontentsline{toc}{schapter}{#1}
 \fi
 \thispagestyle{plain}
 \indexsize\raggedright
 \parindent\z@
 \parskip\z@ \@plus.3\p@\relax
 \let\item=\@idxitem
}
\def\endthe@index{\if@restonecol\onecolumn\else\clearpage\fi}

\newcommand\@idxitem{\par\hangindent 2em}
\newcommand\subitem{\par\hangindent 3em \hspace*{1em}}
\newcommand\subsubitem{\par\hangindent 4em \hspace*{2em}}
%\newcommand\indexspace{\par \vskip 9\p@ \@plus 5\p@ \@minus 3\p@\relax}
\newcommand\indexspace{\par \vskip 3\p@ \@plus 1\p@ \@minus 1\p@\relax}


%    ****************************************
%    *          NUMBERED FOOTNOTES          *
%    ****************************************
%

\newskip\@footindent
\@footindent=1em

\renewcommand\footnoterule{\kern-3\p@ \hrule width 0\columnwidth \kern 2.6\p@}

% reset on chapters
\@addtoreset{footnote}{chapter}
\long\def\@makefntext#1{\@setpar{\@@par\@tempdima \hsize
 \advance\@tempdima-\@footindent
 \parshape \@ne \@footindent \@tempdima}\par
 \noindent \hbox to \z@{\hss\@thefnmark\enskip}#1}

\renewcommand\thefootnote{\mbox{$\fnsymbol{footnote}$}}
\def\@makefnmark{\hbox{\@thefnmark}}

% numbers preferred
%\def\fnsymbol#1{\expandafter\@fnsymbol\csname c@#1\endcsname}
 \def\@fnsymbol#1{{{^{\scriptstyle\@arabic{#1}}}\relax}}% correct


% redfine footnotemark and footnotetext to work on chapter openings
\newcommand\alphafootnotes{%
 \def\@fnsymbol##1{{{\scriptstyle\@alph{##1}}\relax}}%
 \def\@xfootnotenext[##1]{%
  \begingroup
     \csname c@\@mpfn\endcsname ##1\relax
     \unrestored@protected@xdef\@thefnmark{$^{\thempfn}$}%
  \endgroup
  \@footnotetext}%
}

\newcommand\arabicfootnotes{%
  \setcounter{footnote}{0}%
  \def\@fnsymbol##1{{{^{\scriptstyle\@arabic{##1}}}\relax}}%
}

%\def\@fnsymbol#1{\ifcase#1\or \dagger\or \ddagger\or
%   \S \or \P \or \|\or \dagger\dagger
%   \or \ddagger\ddagger\or \S\S\or \P\P\or \|\|\else *\fi\relax}
%
% MINIPAGE
%
\skip\@mpfootins = 4\p@ \@plus 1\p@ \@minus .1\p@  % top of first minipage footnote.
\def\thempfootnote{\alph{mpfootnote}}
\def\@makempfnmark{\hbox{$^{\@thefnmark}$}}
%
\long\def\@makempfntext#1{\@setpar{\@@par\@tempdima \hsize
 \leftskip 0.5pc\rightskip\leftskip
 \advance\@tempdima-\@footindent
 \parshape \@ne \@footindent \@tempdima}\par
 \noindent \hbox to \z@{\hss$^{\@thefnmark}$\enskip}#1}
%
% redefine minipage to reset footnotemark as well as text and counters...
%
%\def\@iminipage[#1]#2{\leavevmode \@pboxswfalse
% \if #1b\vbox
%  \else \if #1t\vtop
%         \else \ifmmode \vcenter
%                \else \@pboxswtrue $\vcenter
%               \fi
%        \fi
% \fi\bgroup
% \hsize #2\textwidth\hsize \columnwidth\hsize
% \@parboxrestore
% \def\@mpfn{mpfootnote}
% \def\thempfn{\thempfootnote}\c@mpfootnote\z@
% \let\@makefnmark\@makempfnmark % v.01.03 AJW
% \let\@footnotetext\@mpfootnotetext
% \let\@listdepth\@mplistdepth \@mplistdepth\z@
% \@minipagerestore\global\@minipagetrue
% \everypar{\global\@minipagefalse\everypar{}}}

\def\@iiiminipage#1#2[#3]#4{% from latex.ltx
  \leavevmode
  \@pboxswfalse
  \setlength\@tempdima{#4}%
  \def\@mpargs{{#1}{#2}[#3]{#4}}%
  \setbox\@tempboxa\vbox\bgroup
    \color@begingroup
      \hsize\@tempdima
      \textwidth\hsize \columnwidth\hsize
      \@parboxrestore
      \def\@mpfn{mpfootnote}\def\thempfn{\thempfootnote}\c@mpfootnote\z@
      \let\@makefnmark\@makempfnmark % from v1.2 ajw
      \let\@footnotetext\@mpfootnotetext
      \let\@listdepth\@mplistdepth \@mplistdepth\z@
      \@minipagerestore\global\@minipagetrue %% \global added 24 May 89
      \everypar{\global\@minipagefalse\everypar{}}}

\long\def\@mpfootnotetext#1{\global\setbox\@mpfootins
 \vbox{\unvbox\@mpfootins
  \footnotesize
  \hsize\columnwidth \@parboxrestore
  \edef\@currentlabel{\csname p@mpfootnote\endcsname\@thefnmark}
  \@makempfntext{\rule{\z@}{\footnotesep}\ignorespaces #1\strut}}}

%       ****************************************
%       *               ENDNOTES               *
%       ****************************************
%
\newcounter{endnote}[chapter] \def\theendnote{\arabic{endnote}}
\newdimen\endnotesep
\newwrite\@enotes
\newif\if@enotesopen          \global\@enotesopenfalse
\newif\ifAJW@firstendnote     \global\AJW@firstendnotefalse

% in some designs, notes headings are different from section heads
% this is not the case for PT1

%\ifAJW@indentsty
% \def\endnoteheading{\@startsection {section}{1}{\AJW@indent}
%  {-30\p@ \@plus-7\p@ \@minus-3.5\p@}{8\p@}{\Aheadsize\bfseries\AJW@hangraggedright}}
%\else
% \def\endnoteheading{\@startsection {section}{1}{\z@}
%   {25pt plus6pt minus2pt}{22pt}%
%   {\fontsize{12}{14}\selectfont\seriffont\bfseries\centering}}
% \fi

\def\@makeenmark{\hbox{$^{\@theenmark}$}}

\def\endnote{%
 \ifAJW@multisty \AJW@firstendnotefalse
        \else \ifnum \c@endnote = \z@ \AJW@firstendnotetrue
                                \else \AJW@firstendnotefalse \fi
 \fi
 \@ifnextchar[{\@xendnote
  }{\stepcounter{endnote}%
    \xdef\@theenmark{\theendnote}%
   }%
 \@endnotemark
 \@endnotetext
}
\def\@xendnote[#1]{\begingroup \c@endnote=#1\relax
  \xdef\@theenmark{\theendnote}\endgroup
}
\let\@doanenote=0
\let\@endanenote=0
\def\@openenotes{\immediate\openout\@enotes=\jobname.ent\relax
 \global\@enotesopentrue
}
\gdef\AJW@endnotehead{\relax}
\long\def\@endnotetext#1{%
 \if@enotesopen \else \@openenotes \fi
 \ifAJW@firstendnote
   \addtoendnotes{\protect\subsection*{\AJW@endnotehead}}%\endnoteheading*
 \fi
   \addtoendnotes{\protect\noindent}%
   \ifAJW@multisty\else
  \addtoendnotes{\protect\mark{\thepage}}%
 \fi
 \immediate\write\@enotes{\@doanenote{\@theenmark}}%
 \begingroup
 \def\next{#1}%
  \newlinechar='40
  \immediate\write\@enotes{\meaning\next}%
 \endgroup
 \immediate\write\@enotes{\@endanenote}%
 \edef\@currentlabel{\csname p@endnote\endcsname\@theenmark}%
}
\long\def\addtoendnotes#1{%
 \if@enotesopen \else \@openenotes \fi
 \begingroup
  \newlinechar='40
  \let\protect\string
  \immediate\write\@enotes{#1}%
 \endgroup
}
\def\endnotemark{\@ifnextchar [{\@xendnotemark
 }{\stepcounter{endnote}%
   \xdef\@theenmark{\theendnote}}%
 \@endnotemark
}
\def\@xendnotemark[#1]{\begingroup
  \c@endnote #1\relax
  \xdef\@theenmark{\theendnote}\endgroup
}
\def\@endnotemark{\leavevmode
 \ifhmode \edef\@x@sf{\the\spacefactor}\fi
 \@makeenmark
 \ifhmode\spacefactor\@x@sf\fi\relax
}
\def\endnotetext{\@ifnextchar [{\@xendnotenext
  }{\xdef\@theenmark{\theendnote}}%
 \@endnotetext
}
\def\@xendnotenext[#1]{\begingroup
  \c@endnote=#1\relax \xdef\@theenmark{\theendnote}\endgroup
}

\ifAJW@multisty
 \def\enoteheading{\section*{Notes}}
 %\def\enoteheading{\endnoteheading*{Notes}}
\else
 \def\enoteheading{\chapter*{Notes}%\leavevmode\par\vskip-\baselineskip
 }
\fi

\def\enoteformat{\raggedright
 \ifAJW@indentsty
  \@tempdima = 3pc
 \else
  \@tempdima = 2em
 \fi
 \leftskip \@tempdima
 \leavevmode
 \hangindent 3.1pt
 \hskip-\@tempdima
 \hbox to \@tempdima {\hfil\@theenmark\hspace*{0.5em}}%
}


\ifAJW@multisty\else
 \def\AJW@enoteheadline{Notes to pp. \firstmark--\botmark}
\fi
\def\theendnotes{\immediate\closeout\@enotes
 \global\@enotesopenfalse
 \begingroup
  \makeatletter
  \def\@doanenote##1##2>{\def\@theenmark{##1}%\par removed
    \begingroup\enoteformat}%
  \def\@endanenote{\par\endgroup}%
  \def\ETC.{\@warning{[CUPBOOK] Some long endnotes will be truncated}
   \def\ETC.{\relax}}
  \enoteheading
  \ifAJW@multisty\else % redefine marking for running heads
   \def\chaptermark##1{}
   \def\leftmark{\AJW@enoteheadline}
   \def\rightmark{\AJW@enoteheadline}
  \fi
  \enotesize
  %\noindent
  \@input{\jobname.ent}%
  \ifAJW@multisty\else\clearpage\fi
 \endgroup
}



%    ****************************************
%    *         FIGURES AND TABLES           *
%    ****************************************
%
% Float placement parameters.
%
\setcounter{topnumber}{2}
\renewcommand\topfraction{0.9}
\setcounter{bottomnumber}{2}
\renewcommand\bottomfraction{0.5}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{0.05}
\renewcommand\floatpagefraction{0.95}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{0.9}
\renewcommand\dblfloatpagefraction{0.8}
%


% \@makecaption{NUMBER}{TEXT} : Macro to make a figure or table caption.
%
\newlength\figcapindent
\setlength\figcapindent{1.5pc}% 1.5pc either side

\newlength\figcapwidth
\setlength\figcapwidth\textwidth
\addtolength\figcapwidth{-2\figcapindent}

\newbox\@figbox

\font\figboxcurve=lcircle10 at 25pt%

\def\multilinefigcaption{\vskip -8.5pt% multi-line captions
  \noindent\fboxsep5pt\colorbox{fortycolor}{\vbox to 0pt{%
    \kern1.24pt%
    \hbox{\kern-6pt{\textcolor{fortycolor}{\figboxcurve\char"074}}}%
  }%
  \textcolor{white}{%
    \seriffont\hspace*{6.5pt}{}%
    {\vrule depth2.5pt height0pt width0pt}%
  }\hspace*{24pt}}}

\def\singlelinefigcaption{\vskip-1.7pt% one-line captions%
  \noindent\fboxsep5pt\colorbox{fortycolor}{\vbox to 0pt{%
    \kern1.24pt%
    \hbox{\kern-5pt{\textcolor{fortycolor}{\figboxcurve\char"074}}}%kerned -5 to get smooth curve
  }%
  \textcolor{white}{%
    \seriffont\hspace*{6.5pt}{}%
    {\vrule depth2.5pt height0pt width0pt}%
  }\hspace*{24pt}}}

% single lines to be flush-left, so use the same coding
\long\def\make@figure@caption#1#2{\vskip11\p@
  \setbox\@tempboxa\hbox{%
    %{\fignumbersize\seriffont\bfseries #1}\enskip% caption is outside measure
    {\figcaptionsize\seriffont\selectfont #2}%
  }%
%Y  \ifdim \wd\@tempboxa >\hsize% multi-line captions
    \vspace*{-8pt}%
    \hspace*{-53pt}%
    {\parbox[t]{5pc}{\multilinefigcaption\vspace*{-11.5pt}\par
      \fignumbersize\seriffont\bfseries
      \textcolor{white}{\hspace*{-11.7pt}\hfil #1%dot removed
      \hfil}%
      }%
    }%
    \hspace*{-7.1pt}\parbox[t]{30pc}{\figcaptionsize\seriffont%
      \raggedright #2}\par
%Y  \else% one-line captions to be flush-left
    %\ifSFB@indentsty
    %  \hbox to\hsize{\box\@tempboxa\hfill}\par
    %\else
%Y      \vspace*{-6pt}%
%Y      \hspace*{-54pt}% move 1pt to the left due to kerning
%Y      {\parbox{5pc}{\singlelinefigcaption\vspace*{-11.5pt}\par
%Y        \fignumbersize\seriffont\bfseries
%Y        \textcolor{white}{\hspace*{-10.7pt}\hfil #1%dot removed
%Y        \hfil}%
%Y        }%
%Y        \figcaptionsize\seriffont%
%Y        \hskip-6pt%
%Y        \hbox to\hsize{\hfill #2\hfill}\par
%Y      }%
    %\fi
%Y  \fi
}


% same as above, except for lines marked %Changed
% depth of page roughly 48pc excl running head
% allow 4.5pc + 0.5pc for Fig #

% single lines to be flush-left, so use the same coding
\long\def\make@rotatedfigure@caption#1#2{\vskip11\p@
  \setbox\@tempboxa\hbox{%
    {\fignumbersize\seriffont\bfseries\selectfont #1}\enskip
    {\figcaptionsize\seriffont\selectfont #2%
    }%
  }%
%Y  \ifdim \wd\@tempboxa >38pc% multi-line captions
    \vspace*{6pt}%
    \hspace*{6pt}%changed
    {\parbox[t]{5.5pc}{\multilinefigcaption\vspace*{-11.5pt}\par
      \fignumbersize\seriffont\bfseries
      \textcolor{white}{\hspace*{-11.7pt}\hfil#1%dot removed
      \hfil
      }}%
    }%
    \hspace*{-7.1pt}\parbox[t]{40pc}{\figcaptionsize\seriffont
      \raggedright#2}\par
%Y  \else% one-line captions to be centred
    %\ifSFB@indentsty
    %  \hbox to\hsize{\box\@tempboxa\hfill}\par
    %\else
%Y      \vspace*{6pt}%
%Y      \hspace*{6pt}%Changed
%Y      \centering
%Y      {\parbox{5.5pc}{\singlelinefigcaption\vspace*{-11.5pt}\par
%Y        \fignumbersize\seriffont\bfseries
%Y        \textcolor{white}{\hspace*{-11.7pt}\hfil#1%dot removed
%Y        \hfil
%Y        }%
%Y        }%
%Y        \figcaptionsize\seriffont
%Y        \hskip-6pt%
        %\hbox to\hsize{\hfill #2\hfill\hspace*{10pc}}\par%Changed
%Y        {#2}\par%changed
%Y      }%
    %\fi
%Y  \fi
}

% wide figures
\newcommand\widefigure{\hspace*{-5pc}}

%
% Float control
%
\newdimen\AJW@floatwidth

\def\AJW@float{\AJW@floatwidth \textwidth
 \addtolength{\AJW@floatwidth}{-\AJW@floatindent}
 \hspace*{\AJW@floatindent}\minipage{\AJW@floatwidth}%
 \ifAJW@indentsty\raggedright\else\centering\fi\small}
\def\endAJW@float{\endminipage}
\def\AJW@widefloat{\minipage{\textwidth}\raggedright\small}
\let\endAJW@widefloat=\endAJW@float
%
% FIGURE
%
\newcounter{figure}[chapter]
\def\thefigure{\thechapter.\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\ \thefigure}

\def\figure{\let\@makecaption=\make@figure@caption
 \AJW@floatindent=\AJW@indent
 \@ifnextchar[{\AJW@figone}{\AJW@figzero}}
\def\AJW@figone[#1]{\@xfloat{figure}[#1]\AJW@float}
\edef\AJW@figzero{\noexpand\@xfloat{figure}[\fps@figure]\noexpand\AJW@float}
\def\endfigure{\endAJW@float\end@float}

\@namedef{figure*}{\let\@makecaption=\make@figure@caption
 \@ifnextchar[{\AJW@widefigone}{\AJW@widefigzero}}
\def\AJW@widefigone[#1]{\@xfloat{figure}[#1]\AJW@widefloat}
%\edef\AJW@widefigzero{\noexpand\@xfloat{figure}[p]\noexpand\AJW@widefloat}
\edef\AJW@widefigzero{\noexpand\@xfloat{figure}[\fps@figure]\noexpand
  \AJW@widefloat}
\@namedef{endfigure*}{\endAJW@widefloat\end@float}
%
% TABLE
%
\newcounter{table}[chapter]
\def\thetable{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\ \thetable}

\def\AJW@tabone[#1]{\@xfloat{table}[#1]\AJW@float}
\edef\AJW@tabzero{%
  \noexpand\@xfloat{table}[\fps@table]\noexpand\AJW@float}

\def\AJW@widetabone[#1]{\@xfloat{table}[#1]\AJW@widefloat}
%\edef\AJW@widetabzero{\noexpand\@xfloat{table}[p]\noexpand\AJW@widefloat}
\edef\AJW@widetabzero{\noexpand\@xfloat{table}[\fps@table]\noexpand\AJW@widefloat}

% TABLE AND ARRAY PARAMETERS

\setlength\arraycolsep{.5em}
\setlength\tabcolsep{.5em}
\setlength\arrayrulewidth{.5pt}
\setlength\doublerulesep{2.5pt}
\renewcommand\arraystretch{1}

\def\HD#1#2{\vrule height #1pt depth #2pt width 0pt\relax}

\newenvironment{table}
  {\let\@makecaption\@tablecaption\@float{table}}
  {\end@float}

\newenvironment{table*}
  {\@dblfloat{table}}
  {\end@dblfloat}

\def\sidewaystable{\@rotfloat{table}\let\@makecaption\@tablecaption}
\let\endsidewaystable\end@rotfloat

\newcommand\tabfont{\seriffont\tablecaptionsize\centering}

\long\def\@tablecaption#1#2{\vskip4\p@
  \setbox\@tempboxa\hbox{\seriffont\bfseries\tablecaptionsize #1\enskip\rmfamily #2}%
  \fboxsep5pt%
  \ifdim \wd\@tempboxa >\hsize
    \vspace*{-4pt}%
      {\fboxsep0pt\colorbox{fortycolor}{\parbox{\tempdime}{\vspace*{6pt}%
       \tabfont\textcolor{white}{{\bfseries #1}{\enspace}#2}\vphantom{gh}\par
       \vspace*{4.8pt}}}\vspace*{-8pt}}%
  \else
    \ifAJW@indentsty
    \hbox to\hsize{\box\@tempboxa\hfill}\par
   \else
   \vspace*{-4pt}%
    {\fboxsep0pt\colorbox{fortycolor}{\parbox{\tempdime}{\vspace*{6pt}%
     \tabfont\textcolor{white}{{\bfseries #1}{\enspace}#2}\vphantom{gh}\par
     \vspace*{4.8pt}}}\vspace*{-8pt}}%
   \fi
 \fi}


% FLOATINGBOX
%
\newcounter{floatingbox}[chapter]
\def\thefloatingbox{\thechapter.\@arabic\c@floatingbox}
\def\fps@floatingbox{tbp}
\def\ftype@floatingbox{2}
\def\ext@floatingbox{lofb}
\def\fnum@floatingbox{\boxxname\ \thefloatingbox}

\def\AJW@boxone[#1]{\@xfloat{floatingbox}[#1]\AJW@float}
\edef\AJW@boxzero{%
  \noexpand\@xfloat{floatingbox}[\fps@floatingbox]\noexpand\AJW@float}

\def\AJW@wideboxone[#1]{\@xfloat{floatingbox}[#1]\AJW@widefloat}
\edef\AJW@wideboxzero{\noexpand\@xfloat{floatingbox}[\fps@floatingbox]\noexpand\AJW@widefloat}

\newenvironment{floatingbox}
  {\let\@makecaption\@floatingboxcaption\@float{floatingbox}}
  {\end@float}

\newenvironment{floatingbox*}
  {\@dblfloat{floatingbox}}
  {\end@dblfloat}

\def\sidewaysfloatingbox{\@rotfloat{floatingbox}\let\@makecaption\@floatingboxcaption}
\let\endsidewaysfloatingbox\end@rotfloat

% -------------new version-------------
\long\def\@floatingboxcaption#1#2{%
%
% Box number
%
\vbox to-1pt{\hspace*{-5pc}\includegraphics[width=421pt]{PT1box.eps}}%
\hspace*{-5pc}%
\hbox to5pc{\hfill\color{white}\floatingboxsize\seriffont\bfseries
  {\strut #1}\hfill\hspace{6pt}}%
%
% Box title
%
\hbox to\textwidth{\hfill{\color{white}\floatingboxsize\seriffont\bfseries
  {\strut #2}}\hfill}%
%
% Box contents
%
\\*[-20.5pt]}% remove space between box title and contents


%\newdimen\tempdime % for floating boxes this is \hsize

\newbox\temptbox
 \newenvironment{processfloatingbox}[2]{%
  \setbox\temptbox=\hbox{{\floatingboxsize\seriffont\bfseries #2}}%
  \tempdime\wd\temptbox%
  \@processfloatingbox{#1}{#2}{\textwidth}%
}{\relax}


\newcommand{\@processfloatingbox}[3]{%
  \caption{#1}{%
    \begin{shaded}%
      {{\color{black}{\floatingboxsize\seriffont\strut #2\par}}}%
    \end{shaded}%
    %\end{tikzpicture}%
  }\par
}

% From ifmtarg.sty
% Copyright Peter Wilson and Donald Arseneau, 2000
\begingroup
\catcode`\Q=3
\long\gdef\@ifmtarg#1{\@xifmtarg#1QQ\@secondoftwo\@firstoftwo\@nil}
\long\gdef\@xifmtarg#1#2Q#3#4#5\@nil{#4}
\long\gdef\@ifnotmtarg#1{\@xifmtarg#1QQ\@firstofone\@gobble\@nil}
\endgroup

\newdimen\tempdime
\newbox\temptbox
 \newenvironment{processtable}[2]{%
  \setbox\temptbox=\hbox{{\tablesize\selectfont#2}}%
  \tempdime\wd\temptbox
  \@processtable{#1}{#2}{\tempdime}%
  }%
  {\relax}

 \newcommand{\@processtable}[3]{%
  \hbox to \hsize{\hss\begin{minipage}[t]{#3}%
    \caption{#1\vspace{3pt}}{%
      \fboxsep0pt\begin{shaded}\vspace*{3pt}%
        \begin{minipage}%
          {\tempdime}\tablesize\selectfont#2%correct
        \end{minipage}\hskip-6pt\vspace*{3pt}%
      \end{shaded}%
    }\par
  \end{minipage}\hss
  }%
}

% BOXES - referred to as boxxes - go back For coding

% lower half of tables
\renewenvironment{shaded}{%
  \def\FrameCommand{\colorbox{fifteencolor}}%
  \parindent0pt%
  \MakeFramed {\advance\hsize-\width \FrameRestore}}%
 {\endMakeFramed}

% for eqns, etc.
\newenvironment{shadedbox}{%
\setlength\fboxsep{5\p@}%
  \def\FrameCommand{\colorbox{tencolor}}%
  \parindent0pt%
  \MakeFramed {\advance\hsize-\width \FrameRestore}}%
 {\endMakeFramed}


%    ****************************************
%    *                TITLE                 *
%    ****************************************
%
% TITLEPAGE
%
\newif\ifAJW@titlepage
\def\titlepage{\@restonecolfalse
 \if@twocolumn\@restonecoltrue\onecolumn
  \else \newpage
 \fi
 \AJW@titlepagetrue
 \thispagestyle{empty}%
}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi
  \AJW@titlepagefalse}
%
\def\maketitle{\openin1 \jobname.ttl
 \ifeof1
  \typeout{No file \jobname.ttl.}
  \begin{titlepage}
   \seriffont
   \ifAJW@indentsty \begin{indented}
              \else \begin{center}
   \fi
   %\ifAJW@largesty \vspace*{4\p@} \else
   \vspace*{2\p@}
   %\fi
   \huge \@title\par
   \if\@subtitle\empty\else \vskip 9\p@ \Large\@subtitle \par \fi
   \vspace{28\p@}
   {\large
     \ifAJW@multisty
       Edited by \@author\\[3\baselineskip]
       \@details
     \else
       \@author\\[3\baselineskip]
       \@details
     \fi
   % \ifAJW@indentsty \begin{author@tabular}[t]{@{}l@{}}
   %   \else \begin{author@tabular}[t]{c}
   % \fi \@author
   % \end{author@tabular}\par}
% four lines above re-written to avoid clash with array.sty
   %%%%  \@author
     \par}
   \vfill
   \ifAJW@indentsty \end{indented}
              \else \end{center}
   \fi
  \end{titlepage}
  \cleardoublepage
 \else
  \closein1 \relax\@@input \jobname.ttl
 \fi
 \setcounter{footnote}{0} % Reset definitions
 \let\thanks\relax
 \gdef\@thanks{}
 \gdef\@author{}
 \gdef\@shortauthor{}
 \gdef\@title{}
 \let\maketitle\relax
}
\gdef\@title{\mbox{}}
\def\title{\@ifnextchar[ {\@titletwo}{\@titleone}}
\def\@titletwo[#1]#2{\gdef\@title{#2}\gdef\@subtitle{#1}}
\def\@titleone#1{\gdef\@title{#1}\gdef\@subtitle{}}
\def\subtitle#1{\gdef\@subtitle{#1}}
\gdef\@subtitle{}
\def\halftitle#1{\gdef\@halftitle{#1}}
\gdef\@halftitle{\@title}
%
\gdef\@author{\mbox{}}
\gdef\@shortauthor{}

\newcommand\editor[1]{\gdef\@editor{#1}}
\gdef\@editor{{\itshape A\,N\,Editor}}


\def\author@tabular{\def\@halignto{}\@authortable}

\let\endauthor@tabular=\endtabular

\def\@authortable{\leavevmode \hbox \bgroup $\let\@acol=\@tabacol
 \let\@classz=\@tabclassz \let\@classiv=\@tabclassiv
 \ifAJW@titlepage \let\\=\AJWauthor@tabcrone
            \else \let\\=\author@tabcrone
 \fi
 \ignorespaces \@tabarray}

\def\author@tabcrone{{\ifnum0=`}\fi\@xtabularcr\enotesize\itshape
 \ifAJW@titlepage \let\\=\AJWauthor@tabcrtwo
            \else \let\\=\author@tabcrtwo
 \fi
 \ignorespaces}

\def\author@tabcrtwo{{\ifnum0=`}\fi\@xtabularcr[-3\p@]\enotesize\itshape
 \ifAJW@titlepage \let\\=\AJWauthor@tabcrtwo
            \else \let\\=\author@tabcrtwo
 \fi
 \ignorespaces}

\def\AJWauthor@tabcrone{{\ifnum0=`}\fi\@xtabularcr[-2\p@]\footnotesize\itshape
 \let\\=\AJWauthor@tabcrtwo\ignorespaces}

\def\AJWauthor@tabcrtwo{{\ifnum0=`}\fi\@xtabularcr[-5\p@]\footnotesize\itshape
 \let\\=\AJWauthor@tabcrtwo\ignorespaces}

\def\and{\end{author@tabular}\vskip 6\p@\par
 \ifAJW@indentsty \begin{author@tabular}[t]{@{}l@{}}
            \else \begin{author@tabular}[t]{@{}c@{}}
 \fi}



%    ****************************************
%    *            PAGE STYLES               *
%    ****************************************
%

\newcommand\copyrightline[1]{%
  \def\@copyrightline{#1}}

\edef\@copyrightline{\relax}

\newcommand\c@pyrightline[1]{%
  \gdef\@c@pyrightline{#1}}

\gdef\@c@pyrightline{%
  \vbox to 5.5\p@{\noindent
  \parbox[t]{\textwidth}{\normalfont\footnotesize\baselineskip 9\p@
  \@copyrightline
  }%
  \vss}%
}


% Initializes TeX's marks
%
\mark{{}{}}
%
% centred \ps@plain and \ps@empty defined in LATEX.TEX
%
\ifAJW@indentsty
  \def\ps@plain{\let\@mkboth=\@gobbletwo
    \def\@oddfoot {\listsize\rm\hfill\thepage}
    \def\@evenfoot{\listsize\rm\thepage\hfill}
    \def\@oddhead {}
    \def\@evenhead{}}
\else
  \def\ps@plain{\leftskip\z@\let\@mkboth\@gobbletwo\vfuzz=5\p@
    \def\@oddhead{}%
    \def\@evenhead{}%
    \def\@oddfoot{%
        \llap{\hbox to 5pc{\vbox{%
          \hsize=4.5pc\centering\headandfootsize\seriffont\thepage\\[-6pt]%
          \textcolor{fortycolor}{\rule{4.5pc}{1pt}}}}}%
          {\normalfont\selectfont\@c@pyrightline}\hfil}
    \def\@evenfoot{%
        \llap{\hbox to 5pc{\vbox{%
          \hsize=4.5pc\centering\headandfootsize\seriffont\thepage\\[-6pt]%
          \textcolor{fortycolor}{\rule{4.5pc}{1pt}}}}}%
          {\normalfont\selectfont\@c@pyrightline}\hfil}
    \def\sectionmark##1{}%
    \def\subsectionmark##1{}%
 }
\fi

%
% Definition of 'headings' page style
%
\def\rcboxhead{\raisebox{-5.2pt}{\includegraphics[width=4.5pc]{PT1header.eps}}}

\def\thin@rule{%
  {\parindent0pt\par\vbox to -6pt{%
    \hbox to 35pc{%
      \rcboxhead{}%
      \kern6pt%
      {\rule[0.8pt]{30pc}{1pt}}%
    }\par
  }}%
}

\def\ps@headings{%
  \let\@mkboth=\markboth
  \def\@evenfoot{}%
  \def\@oddfoot {}%
  \def\@evenhead{\hspace*{-5pc}\hbox to 5pc{%
    \hfil\headandfootsize\seriffont\thepage\hfil\hspace{0.5pc}}%
    \hbox to \textwidth{%
    \hfill{\normalsize\seriffont%
    {\leftmark}}\hfill}%
    \hspace*{-35pc}{\thin@rule}%
  }%
  %
  \def\@oddhead{\hspace*{-5pc}\hbox to 5pc{%
    \hfil\headandfootsize\seriffont\thepage\hfil\hspace{0.5pc}}%
    \hbox to \textwidth{%
    \hfill\normalsize\seriffont
    {\rightmark}\hfill}%
    \hspace*{-35pc}{\thin@rule}%
  }%
  %\def\chaptermark##1{\markboth{##1}{##1}}%
  %\def\sectionmark##1{\markright{\ifnum \c@secnumdepth >\z@
  %  \thesection\enskip\fi ##1}}%
  \ifAJW@multisty
    \def\chaptermark##1{}%
    \def\sectionmark##1{}%
  \else
    \def\chaptermark##1{\markboth{##1}{##1}}%
    \def\sectionmark##1{\markright{\ifnum \c@secnumdepth >\z@
      \fi ##1}}%
  \fi
}

%
% Definition of 'myheadings' page style
%
\def\ps@myheadings{%
  \let\@mkboth=\@gobbletwo
  \def\@evenfoot{}%
  \def\@oddfoot {}%
  \def\@evenhead{\hspace*{-5pc}\hbox to 5pc{%
    \hfil\headandfootsize\seriffont\thepage\hfil\hspace{0.5pc}}%
    \hbox to \textwidth{%
    \hfill{\normalsize\seriffont%
    {\leftmark}}\hfill}%
    \hspace*{-35pc}{\thin@rule}%
  }%
  %
  \def\@oddhead{\hspace*{-5pc}\hbox to 5pc{%
    \hfil\headandfootsize\seriffont\thepage\hfil\hspace{0.5pc}}%
    \hbox to \textwidth{%
    \hfill\normalsize\seriffont
    {\rightmark}\hfill}%
    \hspace*{-35pc}{\thin@rule}%
  }%
  \ifAJW@multisty
    \def\chaptermark##1{}%
    \def\sectionmark##1{}%
  \else
    \def\chaptermark##1{}%
    \def\sectionmark##1{}%
  \fi
}

\newcommand\contentsname{Contents}
\newcommand\listfigurename{Illustrations}
\newcommand\listtablename{Tables}
\newcommand\listfloatingboxname{Boxes}
\newcommand\listcontribname{Contributors}
\newcommand\bibname{Bibliography}
\newcommand\refname{References}
\newcommand\indexname{Index}
\newcommand\figurename{Fig.\/}
\newcommand\tablename{Table}
\newcommand\boxxname{Box}
\newcommand\partname{P\ls A\ls R\ls T}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}

%    ****************************************
%    *            MISCELLANEOUS             *
%    ****************************************
%
% DATE
%
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
%
% EQUATION and EQNARRAY -- put here because it must follow \chapter definition
%
\@addtoreset{equation}{chapter}
\def\theequation{\thechapter.\arabic{equation}}
% \jot = 3\p@ % Extra space added between lines of an eqnarray environment
%
% Flush left / indented equations
%
% FLEQN DOCUMENT-STYLE OPTION - released 9 March 1987
%    for LaTeX version 2.09
% Copyright (C) 1987 by Leslie Lamport
%
\ifAJW@indentsty
 \def\[{\relax\ifmmode\@badmath\else\begin{trivlist}\item[]\leavevmode
  \hbox to\linewidth\bgroup$ \displaystyle \hskip\AJW@indent\bgroup\fi}
 \def\]{\relax\ifmmode \egroup $\hfil \egroup \end{trivlist}%
  \else \@badmath \fi}
 \def\equation{\refstepcounter{equation}\trivlist \item[]\leavevmode
  \hbox to\linewidth\bgroup $ \displaystyle \hskip\AJW@indent}
 \def\endequation{$\hfil \displaywidth\linewidth\@eqnnum\egroup \endtrivlist}
 \def\eqnarray{\stepcounter{equation}\let\@currentlabel=\theequation
  \global\@eqnswtrue
  \global\@eqcnt\z@\tabskip\AJW@indent\let\\=\@eqncr
  \abovedisplayskip\topsep\ifvmode\advance\abovedisplayskip\partopsep\fi
  \belowdisplayskip\abovedisplayskip
  \belowdisplayshortskip\abovedisplayskip
  \abovedisplayshortskip\abovedisplayskip
  $$\halign
   to \linewidth\bgroup\@eqnsel\hskip\@centering$\displaystyle\tabskip\z@
   {##}$&\global\@eqcnt\@ne \hskip 2\arraycolsep \hfil${##}$\hfil
   &\global\@eqcnt\tw@ \hskip 2\arraycolsep $\displaystyle{##}$\hfil
    \tabskip\@centering&\llap{##}\tabskip\z@\cr}
 \def\endeqnarray{\@@eqncr\egroup
  \global\advance\c@equation\m@ne$$\global\@ignoretrue}%
\else
 \renewenvironment{eqnarray} % remove exta space around alignment marks
  {\stepcounter{equation}%
   \let\@currentlabel=\theequation
   \global\@eqnswtrue
   \m@th
   \global\@eqcnt\z@
   \tabskip\@centering
   \let\\\@eqncr
   $$\everycr{}\halign to \displaywidth\bgroup
     \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
   &\global\@eqcnt\@ne \hfil${\;##\;}$\hfil
   &\global\@eqcnt\tw@
     $\displaystyle\tabskip\z@skip{##}$\hfil \tabskip\@centering
   &\hb@xt@\z@\bgroup\hss##\egroup
     \tabskip\z@skip \cr}
  {\@@eqncr\egroup\global\advance\c@equation\m@ne
   $$\global\@ignoretrue}
\fi

% this allows proof boxes to be right-aligned in eqnarrays
\newcommand\arrayqed{\arrayqedtrue}
\newcommand\arrayqedhere{\arrayqedfalse}
\newif\ifarrayqed

\def\@@eqncr{\let\reserved@a\relax
    \ifcase\@eqcnt \def\reserved@a{& & &}\or \def\reserved@a{& &}%
     \or \def\reserved@a{&}\else
       \let\reserved@a\@empty
       \@latex@error{Too many columns in eqnarray environment}\@ehc\fi
     \reserved@a
     \if@eqnsw
       \ifarrayqed
         \qedsymbol%
       \else
         \@eqnnum\stepcounter{equation}%
       \fi
    \fi
     \global\@eqnswtrue\global\@eqcnt\z@\cr}


%    ****************************************
%    *            CUP SPECIALS              *
%    ****************************************
%
% redefinition of \cleardoublepage to get blank empty pages
%
\def\clearemptypage{\clearpage
 \if@twoside
  \ifodd\c@page
   \else
    \thispagestyle{empty}
    \mbox{}\newpage
    \if@twocolumn
      \hbox{}\newpage
    \fi
  \fi
 \fi
}
\let\cleardoublepage=\clearemptypage

% the section blocks all from cup-scz - to get rule under unnumbered section heads
\def\@startsection#1#2#3#4#5#6{\if@noskipsec \leavevmode \fi
  \par \@tempskipa #4\relax
  \ifnum#2=1
    \@afterindentfalse
  \else
    \@afterindentfalse
  \fi
  \ifdim \@tempskipa <\z@
    \@tempskipa -\@tempskipa \@afterindentfalse
  \fi
  \if@nobreak
    \ifnum#2=1          % (section)
      \vspace*{0pt}     % required space between 2 heads
    \else
      \ifnum#2=2        % (subsection)
        \vspace*{0pt}   % required space between 2 heads
      \fi
    \fi
    \everypar{}%
  \else
    \addpenalty{\@secpenalty}\addvspace{\@tempskipa}%
  \fi
  \@ifstar
  {\@ssect{#2}{#3}{#4}{#5}{#6}}{\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}


\def\sec@rule{{\color{fortycolor}\parindent0pt\nopagebreak\par\rule{30pc}{1pt}\nopagebreak\par}}%

% redefinition of sections to get en space after chapter number
%

\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth % LEVEL -- is heading to be numbered?
  \def\@svsec{}%
 \else
  \refstepcounter{#1}
  \edef\@svsec{\csname the#1\endcsname \hskip 0.6em\mbox{}} %increased from 0.5em
 \fi
 \@tempskipa #5\relax % AFTERSKIP (-ve indicates a run-in heading)
 \ifdim \@tempskipa>\z@
  \begingroup #6\relax
%ajw   \@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M {#8}\par}
  \ifnum#2=1
    {\hskip #3\relax\@svsec}{\interlinepenalty \@M {#8}\nopagebreak\par
    \hspace*{-5pc}%
    \rule[6.25pt]{0pt}{4pt}%increase space between heading and rule
    \includegraphics[width=35pc]{PT1secrule.eps}}%
  \else
  \ifnum#2=2
    {\hskip #3\relax\@svsec}{\interlinepenalty \@M {#8}\par
    \vspace*{-7pt}\sec@rule}%
  \else
  \ifnum#2=3
    {\hskip #3\relax\@svsec}{\interlinepenalty \@M {#8}\par
    %\vspace*{-6pt}\sec@rule
    }%
    %\vspace*{-7pt}%
  \else
    {\hskip #3\relax\@svsec}{\interlinepenalty \@M {#8}\par}%
  \fi\fi\fi
  \csname #1mark\endcsname{#7}\endgroup
%
  \ifnum#2=6% xheads, yheads
    \relax
  \else
    \addcontentsline{toc}{#1}%
    {\ifnum #2>\c@secnumdepth \else
    \protect\numberline{\csname the#1\endcsname}\fi #7}
  \fi
%
  \else
  \def\@svsechd{#6\hskip #3\@svsec {#8}%
   \csname #1mark\endcsname{#7}%
   \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth \else
    \protect\numberline{\csname the#1\endcsname}\fi#7}}%
 \fi
 \@xsect{#5}}

% add shaded line below unnumbered section heads
\def\@xsect#1{\@tempskipa #1\relax
  \ifdim \@tempskipa>\z@
    \par \nobreak
    \vskip \@tempskipa
    \@afterheading
    \else
      \global\@nobreakfalse \global\@noskipsectrue
      \everypar{\if@noskipsec \global\@noskipsecfalse
      \clubpenalty\@M \hskip -\parindent
      \begingroup \@svsechd \endgroup \unskip
      \hskip -#1\relax  % relax added 14 Jan 91
    \else
      \clubpenalty \@clubpenalty
      \everypar{}%
    \fi}%
  \fi\ignorespaces}


\def\@ssect#1#2#3#4#5#6{%
  \@tempskipa #4\relax
  \ifdim \@tempskipa>\z@
    \begingroup #5%
    \ifnum#1=1
      \@hangfrom{\hskip #2}{\interlinepenalty \@M {#6}\nopagebreak\par
         \hspace*{-5pc}\includegraphics[width=35pc]{PT1secrule.eps}}%
      \vspace*{6pt}%
      \addcontentsline{toc}{section}{#6}%
    \else
      \@hangfrom{\hskip #2}{\interlinepenalty \@M #6\par}%
    \fi
    \endgroup
  \else
    \def\@svsechd{#5\hskip #2\relax #6}%
  \fi
  \@xsect{#4}}


% remove the vertical rules in tables, by default
\def\AJW@fullhline{\noalign{\ifnum0=`}\fi \vskip 3\p@
  \hrule \@height\arrayrulewidth width\textwidth
  \vskip 3\p@
  \futurelet \@tempa\@xhline}
\def\AJW@hline{\noalign{\ifnum0=`}\fi
  \vskip 3\p@
  \hrule \@height \arrayrulewidth
  \vskip 3\p@
  \futurelet \@tempa\@xhline}
\let\AJW@currenthline=\AJW@hline
%
\def\tabular{\def\@halignto{}%
  \let\hline=\AJW@currenthline
  \def\@xhline{\ifx\@tempa\hline
    \vskip -6\p@
    \vskip \doublerulesep
  \fi
  \ifnum0=`{\fi}}%
  \def\@arrayrule{\@addtopreamble{\hskip -.5\arrayrulewidth
%   \vrule \@width \arrayrulewidth
    \hskip .5\arrayrulewidth}}%
  \@tabular}


% If authors really need to re-instate the rules,

\newcommand\reinstaterules{%
  \def\tabular{\let\@halignto\@empty\@tabular}%
}

% ABSTRACT and SOURCE
%
\newenvironment{chapterquote}{%
  \abstractsize
  \trivlist\leftskip 2pc\rightskip\leftskip\item}
  {\endlist}

\newcommand\source[1]{\\*\mbox{}\hfill #1}


% EXTRACT
%
\newenvironment{extract}{%
  \extractsize
  \trivlist\leftskip 2pc\rightskip\z@\item}
  {\endlist}



%    ****************************************
%    *           INITIALIZATION             *
%    ****************************************
%
% Default initializations
%
\ifAJW@landscape \pagestyle{empty}\else \pagestyle{headings}\fi
\pagenumbering{arabic}
\onecolumn
\flushbottom
\frenchspacing

\endinput

