%% This is file 'cambridgeauthordate.bst', 2009/05/13, v1.00
%% Developed from authordate1.bst; for use by Cambridge University Press authors

%  This BibTeX 0.99 style file is intended for documents that use the
%  author-date citation system.  It should be used in conjunction with the
%  authordate1-4.sty (or equivalent) LaTeX style-option.
%
%  In deciding on the bibliography layout to be implemented, the following
%  works have been consulted:
%  1. British Standard 5605:  "Citing publications by bibliographic
%         references", 1978.
%  2. British Standard 1629:  "Bibliographic references", 1976.
%     (Note:  There is now a 1989 edition.)
%  3. "Copy-editing" by Judith Butcher, Cambridge University Press, 1981.
%  4. "The Oxford Dictionary for Writers and Editors", Oxford University
%         Press, 1981 (for abbreviations).
%  5. "The Chicago Manual of Style", Chicago University Press, 1982.
%
%  Discretion has been used in certain matters.  In particular:
%  - Titles are italic except when there is a "contribution title" and
%    a "main publication title", in which case the "contribution title"
%    is roman while the "main publication title" is italic.
%  - When there is a PUBLISHER, it is assumed that ADDRESS contains "place of
%    publication";  output is of the form "place-of-publication: publisher".
%    When there is an ORGANIZATION as well as a PUBLISHER, output is of the
%    form "place-of-publication: publisher, for organization".  Otherwise,
%    it is assumed that ADDRESS is a postal address, and output is of the
%    form "institution, address", "organization, address", "school, address"
%    or "howpublished, address".
%  - If there is a MONTH but no VOLUME or NUMBER for a journal-article, the
%    MONTH is used on the assumption that it contains information that is
%    equivalent to VOLUME or NUMBER.  MONTH is ignored for other documents
%    that have a PUBLISHER, but output after YEAR for anything else.
%  - When CHAPTER and/or PAGES are specified with INBOOK, it is assumed that
%    these fields are being used to pick out particular pages of interest
%    (as an alternative to using the optional [...] argument of \cite);
%    the information is put at the very end of the entry, in line with BS 1629.
%    When CHAPTER and/or PAGES are specified for INCOLLECTION or INPROCEEDINGS,
%    it is assumed that the fields are being used to specify the location of
%    the part/article of interest; the information is put before the
%    BOOKTITLE, in line with Chicago's treatment of "parts" and "chapters".
%  - Since "volume" can be used in various ways (see page 219 of Butcher),
%    this file assumes that, when there is a SERIES, any VOLUME and NUMBER
%    refer to position in the SERIES (i.e. Butcher's meaning 4).
%    If there is a VOLUME but no SERIES, it is assumed that VOLUME refers
%    to a volume within a particular work (i.e. Butcher's meaning 1, 2 or 3).
%    For INPROCEEDINGS and INCOLLECTION, this information goes after the
%    BOOKTITLE.  (If you prefer to have volume information before EDITOR,
%    as on page 449 of Chicago, you can set TYPE = "Vol." with INCOLLECTION.)
%
%  This file differs from authordate2.bst, authordate3.bst and authordate4.bst
%  in that:
%  - The titles of articles, journals and books are left in whatever
%    combination of upper-case and lower-case appears in the .bib file.
%  - Author's and editor's names are in roman.
%
%  The combination of authordate1.bst and authordate1-4.sty allows citations
%  of the form \shortcite{bloggs-60} as well as the usual \cite{bloggs-60}.
%  When
%       ... \cite{bloggs-60} ... \shortcite{bloggs-60} ...
%  appears in the input file,
%       ... (Bloggs, 1960) ... (1960) ...
%  appears in the final document.
%
%  This file was developed from apalike.bst.  It also uses code taken from
%  acm.bst, aaai-named.bst, btxbst.doc, ieeetr.bst and siam.bst.
%
%                                                     David Rhead
%                                                     Cripps Computing Centre
%                                                     Nottingham University
%                                                     March 1990
%

ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
  }
  {}
  { label extra.label sort.label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
	{ add.period$ write$
	  newline$
	  "\newblock " write$
	}
	{ output.state before.all =
	    'write$
	    { add.period$ " " * write$ }
	  if$
	}
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {output.year.month.check}
{ year empty$
    { "empty year in " cite$ * warning$ }
    { add.period$ write$
      month empty$
        { " " year * extra.label * "." *
          after.sentence 'output.state :=
        }
        { " " year * extra.label * " (" * month * ")." *
          after.sentence 'output.state :=
        }
      if$
    }
  if$
}

FUNCTION {output.year.check}
{ year empty$
    { "empty year in " cite$ * warning$ }
    { add.period$ write$
      " " year * extra.label * "." *
      after.sentence 'output.state :=
    }
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem[" write$
  label write$
  "]{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {boldface}
{ duplicate$ empty$
    { pop$ "" }
    { "{\bf " swap$ * "}" * }
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\em " swap$ * "}" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}{, jj}{, ff}" format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
            { ", " * t * }
            { t "others" =
                { ", {et~al.}" * }
                { ", and " * t * }                           %  Butcher, pages
              if$                                           %  186-189.
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.key}
{ empty$
    { key field.or.null }
    { "" }
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      editor num.names$ #1 >                                %  Use ODWE abbrevs.
        { " (eds)" * }                                      %  to avoid
        { " (ed)" * }                                       %  ambiguity between
      if$                                                   %  "editor" and
    }                                                       %  "edition".
  if$
}

FUNCTION {format.title}                                     %  Nothing needs
{ title empty$                                              %  doing here in
    { "" }                                                  %  authordate1.bst
    { title }                                               %  or
  if$                                                       %  authordate3.bst.
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
               { "-" *
                 t #2 global.max$ substring$ 't :=
               }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
        if$
    }
  while$
}

FUNCTION {format.btitle}
{ title empty$
   { "" }                                                   %  Don't change case
   { title emphasize }                                      %  in
   if$                                                      %  authordate1.bst
}                                                           %  or
                                                            %  authordate3.bst.
FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.numberinseries}
{ number empty$
    { "" }
    { number multi.page.check
        { ", nos. " number n.dashify tie.or.space.connect }
        { ", no. "  number tie.or.space.connect }
      if$
    }
  if$
}

FUNCTION {booklike.series.volume.number}                    %  Chicago, pages
{ series empty$                                             %  450-451.
    { volume empty$
      { " " }
      { " Vol. " volume * }
      if$
    }
    {
      volume empty$
        { number empty$
            { series }
            { series format.numberinseries * }
          if$
        }
        { number empty$
            { series ", vol. " volume * * }
            { series ", vol. " * volume * format.numberinseries * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {incollectionlike.series.volume.number}
{ series empty$
    { volume empty$
      { " " }
      { " vol. " volume * }
      if$
    }
    { new.block
      volume empty$
        { number empty$
            { series }
            { series format.numberinseries * }
          if$
        }
        { number empty$
            { series ", vol. " volume * * }
            { series ", vol. " * volume * format.numberinseries * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { output.state mid.sentence =
        { edition "l" change.case$ " edn." * }
        { edition "t" change.case$ " edn." * }
      if$
    }
  if$
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
        { "" pages n.dashify tie.or.space.connect }
        { "" pages tie.or.space.connect }
      if$
    }
  if$
}

FUNCTION {format.pagesinbook}                               %  By the time the
{ pages empty$                                              %  reader has read
    { "" }                                                  %  address, pub'r,
    { pages multi.page.check                                %  note (where the
        { "Pages " pages n.dashify tie.or.space.connect }   %  note may end with
        { "Page " pages tie.or.space.connect }              %  numbers), s/he
      if$                                                   %  may not recognise
    }                                                       %  a number-range as
  if$                                                       %  meaning pages.
}                                                           %  Avoid ambiguity
                                                            %  (Butcher, p.181).

FUNCTION {format.vol.num.date.pages}
{ volume empty$
   { month empty$
       { "" }
       { month }
     if$
   }
    { volume boldface field.or.null
        number empty$
        { month empty$
            'skip$
            { "(" month * ")" *  * }
          if$
        }
        { "(" number * ")" *  *
            volume empty$
            { "there's a number but no volume in " cite$ * warning$ }
            'skip$
          if$
        }
      if$
    }
  if$
  pages empty$
    'skip$
    { duplicate$ empty$
        { pop$ format.pages }
        { ", " * pages n.dashify * }
      if$
    }
  if$
}

FUNCTION {format.chapter.pages.inbook}
{ chapter empty$
    'format.pagesinbook
    { type empty$
        { "Chap." }
        { type }
      if$
      chapter tie.or.space.connect
      pages empty$
        'skip$
        { ", " * format.pagesinbook "l" change.case$ * }
      if$
    }
  if$
}

FUNCTION {format.chapter.pages.incoll}
{ chapter empty$
    { pages empty$
        { "{In:} " }
        { "{" format.pagesinbook " of:} " * * }
      if$
    }
    { type empty$
        { "{Chap. " chapter * }
        { "{" type * " " * chapter * }
      if$
      pages empty$
        { " of:} " * }
        { ", " * format.pagesinbook "l" change.case$ " of:} " * * }
      if$
    }
  if$
}

FUNCTION {format.in.ed.booktitle}                           %  Achieves effect
{ booktitle empty$                                          %  shown in 16.51
    { "" }                                                  %  of Chicago, at
    { editor empty$                                         %  expense of not
        { format.chapter.pages.incoll                       %  achieving effects
          booktitle                  emphasize * }          %  shown in 16.50
        { format.chapter.pages.incoll                       %  of Chicago, on
          format.editors * ", " *                           %  page 189 of
          booktitle                  emphasize * }          %  Butcher and in
      if$                                                   %  4.4 of BS 1629.
    }
  if$                                                       %  Don't change
}                                                           %  case.

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type                                                  %  Don't change
    }                                                       %  case.
  if$
}

FUNCTION {format.tr.number}
{ type empty$
    { "Tech. rept."  }                                      %  ODWE abbrevs.
    'type
  if$
  number empty$
    {                  }                                    %  Whatever was
    { number tie.or.space.connect }                         %  having its case
  if$                                                       %  changed, leave
}                                                           %  it alone.

FUNCTION {format.addr.pub}
{ publisher empty$
    { "" }
    { address empty$
        { "" }
        { address ": " * }
      if$
      publisher *
    }
  if$
}

FUNCTION {format.addr.pub.org}                              %  If there's an
{ address empty$                                            %  an organization
  { publisher * ", for " * organization * }                 %  and a publisher
  { address ": " * publisher * ", for " * organization * }  %  too.
  if$
}

FUNCTION {format.addr.inst}
{ address empty$
  { institution empty$
    { "" }
    { institution }
    if$
  }
  { institution empty$
    { "" }
    { institution ", " * }
    if$
    address *
  }
  if$
}

FUNCTION {format.addr.org}
{ address empty$
  { organization empty$
    { "" }
    { organization }
    if$
  }
  { organization empty$
    { "" }
    { organization ", " * }
    if$
    address *
  }
  if$
}

FUNCTION {format.article.crossref}
{ "{\em In:}"
  " \cite{" * crossref * "}" *
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      "{\em In:}"
    }
    { " Vol." volume tie.or.space.connect
      " of " *
    }
  if$
  "\cite{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{ "{\em In:}"
  " \cite{" * crossref * "}" *
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  output.year.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { journal                  emphasize                    %  Don't change
      "journal" output.check                                %  case.
      format.vol.num.date.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  output.year.check
  new.block
  format.btitle "title" output.check
  crossref missing$
    { new.sentence
      format.edition output
      new.block
      booklike.series.volume.number output
      new.block
      format.addr.pub "publisher" output.check
    }
    { new.block
      format.book.crossref output.nonnull
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {booklet}
{ output.bibitem
  format.authors output
  author format.key output
  output.year.month.check
  new.block
  format.btitle "title" output.check
  new.block
  howpublished output
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  output.year.check
  new.block
  format.btitle "title" output.check
  crossref missing$
    { new.sentence
      format.edition output
      new.block
      booklike.series.volume.number output
      new.block
      format.addr.pub "publisher" output.check
    }
    { new.block
      format.book.crossref output.nonnull
    }
  if$
  new.block
  note output
  new.block                                                 %  BS 1629 (rather
  format.chapter.pages.inbook "chapter and pages" output.check
  fin.entry                                                 %  than Chicago,
}                                                           %  p. 451)

FUNCTION {incollection}
{ output.bibitem
  author empty$
    { format.editors "editor" output.check
      editor format.key output             }
    { format.authors "author" output.check
      author format.key output             }
  if$
  output.year.check
  new.block
  format.title "title" output.check
  new.block
  crossref missing$                                         %  Chapter and/or
    { format.in.ed.booktitle output                         %  page numbers can
      format.edition output                                 %  come out via this
      incollectionlike.series.volume.number output          %  route, too.
      new.block
      format.addr.pub "publisher" output.check
    }
    { format.incoll.inproc.crossref output.nonnull
      new.block
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  publisher empty$
    { output.year.month.check }
    { output.year.check }
  if$
  new.block
  format.title "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      incollectionlike.series.volume.number output
      new.block
      publisher empty$
      {
        organization empty$
        'skip$
        { format.addr.org output }
        if$
      }
      {
        organization empty$
        { format.addr.pub output }
        { format.addr.pub.org output }
        if$
      }
      if$
    }
    { format.incoll.inproc.crossref output.nonnull
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  format.authors output
  author format.key output
  output.year.month.check
  new.block
  format.btitle "title" output.check
  new.sentence
  format.edition output
  new.block
  organization address new.block.checkb
  format.addr.org output
  new.block
  note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  output.year.month.check
  new.block
  format.btitle "title" output.check
  new.block
  "M.Phil. thesis" format.thesis.type output.nonnull        %  ODWE abbrev.
  school "school" output.check
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  author format.key output
  output.year.month.check
  new.block
  format.btitle output
  new.block
  howpublished output
  new.block
  note output
  fin.entry
}

FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  output.year.month.check
  new.block
  format.btitle "title" output.check
  new.block
  "Ph.D. thesis" format.thesis.type output.nonnull          %  Butcher,
  school "school" output.check                              %  page 174.
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  format.editors "author and editor" output.check
  editor format.key output
  publisher empty$
    { output.year.month.check }
    { output.year.check }
  if$
  new.block
  format.btitle "title" output.check
  new.block
  booklike.series.volume.number output
      new.block
      publisher empty$
      {
        organization empty$
        'skip$
        { format.addr.org output }
        if$
      }
      {
        organization empty$
        { format.addr.pub output }
        { format.addr.pub.org output }
        if$
      }
      if$
  new.block
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  output.year.month.check
  new.block
  format.btitle "title" output.check
  new.block
  format.tr.number output.nonnull
  new.sentence
  format.addr.inst "institution" output.check
  new.block
  note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  output.year.month.check
  new.block
  format.btitle "title" output.check
  new.block
  note "note" output.check
  fin.entry
}

FUNCTION {default.type} { misc }

MACRO {jan} {"Jan."}                                        %  ODWE, "months", &
                                                            %  Chicago, p. 383.
MACRO {feb} {"Feb."}

MACRO {mar} {"Mar."}

MACRO {apr} {"Apr."}

MACRO {may} {"May"}

MACRO {jun} {"June"}

MACRO {jul} {"July"}

MACRO {aug} {"Aug."}

MACRO {sep} {"Sept."}

MACRO {oct} {"Oct."}

MACRO {nov} {"Nov."}

MACRO {dec} {"Dec."}

MACRO {acmcs} {"ACM Computing Surveys"}

MACRO {acta} {"Acta Informatica"}

MACRO {cacm} {"Communications of the ACM"}

MACRO {ibmjrd} {"IBM Journal of Research and Development"}

MACRO {ibmsj} {"IBM Systems Journal"}

MACRO {ieeese} {"IEEE Transactions on Software Engineering"}

MACRO {ieeetc} {"IEEE Transactions on Computers"}

MACRO {ieeetcad}
 {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}

MACRO {ipl} {"Information Processing Letters"}

MACRO {jacm} {"Journal of the ACM"}

MACRO {jcss} {"Journal of Computer and System Sciences"}

MACRO {scp} {"Science of Computer Programming"}

MACRO {sicomp} {"SIAM Journal on Computing"}

MACRO {tocs} {"ACM Transactions on Computer Systems"}

MACRO {tods} {"ACM Transactions on Database Systems"}

MACRO {tog} {"ACM Transactions on Graphics"}

MACRO {toms} {"ACM Transactions on Mathematical Software"}

MACRO {toois} {"ACM Transactions on Office Information Systems"}

MACRO {toplas} {"ACM Transactions on Programming Languages and Systems"}

MACRO {tcs} {"Theoretical Computer Science"}

READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

INTEGERS { len }

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}

FUNCTION {format.lab.names}
{ 's :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ duplicate$
  #2 >
    { pop$ " {et~al.}" * }
    { #2 <
        'skip$
        { s #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            { " {\em et~al.}" * }
            { " and " * s #2 "{vv~}{ll}" format.name$ * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
        { cite$ #1 #3 substring$ }
        'key
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
        { key empty$
            { cite$ #1 #3 substring$ }
            'key
          if$
        }
        { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.label}
{ editor empty$
    { key empty$
        { cite$ #1 #3 substring$ }
        'key
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.label}
  { type$ "book" =
    type$ "inbook" =
    type$ "incollection" =                                  %  For sensible
  or or                                                     %  treatment of
    'author.editor.key.label                                %  Singer in
    { type$ "proceedings" =                                 %  BS 1629.
        'editor.key.label
        'author.key.label
      if$
    }
  if$
  duplicate$
  year empty$
  {
    "\protect\citename{" swap$ * ", }" *
    "n.d." * 'label :=                                      %  Chicago,
  }                                                         %  page 457.
  {
    "\protect\citename{" swap$ * ", }" *
    year
    *
    'label :=
  }
  if$
  year field.or.null purify$ *
  sortify 'sort.label :=
}


FUNCTION {sort.format.names}                                %  To produce the
{ 's :=                                                     %  order of entries
  #1 'nameptr :=                                            %  specified in item
  ""                                                        %  (3) on page 187
  s num.names$ 'numnames :=                                 %  of Butcher.
  numnames 'namesleft :=
    { namesleft #0 > nameptr #3 < and }                     %  Only 1st 2 names
    { nameptr #1 >                                          %  matter for
        { "   " * }                                         %  sorting.
        'skip$
      if$                                                   %  Anything that's
      nameptr #2 = numnames #2 > and                        %  "et al" goes
      { "zzzzz" * }                                         %  after works by 2
      {                                                     %  authors.
        s nameptr "{vv{ } }{ll{ }}{  ff{ }}{  jj{ }}" format.name$ 't :=
        nameptr numnames = t "others" = and
          { "zzzzz" * }
          { t sortify * }
        if$
      }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
        { "to sort, need author or key in " cite$ * warning$
          ""
        }
        { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { key empty$
            { "to sort, need author, editor, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.sort}
{ editor empty$
    { key empty$
        { "to sort, need editor or key in " cite$ * warning$
          ""
        }
        { key sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

FUNCTION {presort}                                          %  Two sorting
{ calc.label                                                %  passes, from
  label sortify                                             %  apalike.bst.
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  type$ "incollection" =                                    %  For Singer
  or or                                                     %  in BS 1629.
    'author.editor.sort
    { type$ "proceedings" =
        'editor.sort
        'author.sort
      if$
    }
  if$
  #1 entry.max$ substring$
  'sort.label :=
  sort.label
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}

SORT

STRINGS { last.label next.extra }

INTEGERS { last.extra.num number.label }

FUNCTION {initialize.extra.label.stuff}
{ #0 int.to.chr$ 'last.label :=
  "" 'next.extra :=
  #0 'last.extra.num :=
  #0 'number.label :=
}

FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
  number.label #1 + 'number.label :=
}

FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$                                                       %  Code needed here
                                                            %  if \citeauthor
                                                            %  and \citeyear
  label extra.label * 'label :=                             %  were supported.
  extra.label 'next.extra :=
}

EXECUTE {initialize.extra.label.stuff}

ITERATE {forward.pass}

REVERSE {reverse.pass}

FUNCTION {bib.sort.order}
{ sort.label
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {bib.sort.order}

SORT

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{" number.label int.to.str$ * "}" *
  write$ newline$
  "\expandafter\ifx\csname natexlab\endcsname\relax\def\natexlab#1{#1}\fi"
  write$ newline$
  "\expandafter\ifx\csname selectlanguage\endcsname\relax"
  write$ newline$
  "  \def\selectlanguage#1{\relax}\fi"
  write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
%% End of cambridgeauthordate.bst

