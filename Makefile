all: html

build/%.ipynb: %.md build/env.yml $(wildcard d2l/*.py d2l/*/*.py)
	mkdir -p $(@D)
	cd $(@D); python ../utils/md2ipynb.py ../../$< ../../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard */index.md chapter_appendix/d2l.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

FRONTPAGE_DIR = img/frontpage
FRONTPAGE = $(wildcard $(FRONTPAGE_DIR)/*)
FRONTPAGE_DEP = $(patsubst %, build/%, $(FRONTPAGE))

IMG_NOTEBOOK = $(filter-out $(FRONTPAGE_DIR), $(wildcard img/*))
ORIGIN_DEPS = $(IMG_NOTEBOOK) $(wildcard data/* d2l/* d2l/*/*) environment.yml README.md
DEPS = $(patsubst %, build/%, $(ORIGIN_DEPS))

PKG = build/_build/html/d2l-en.zip

pkg: $(PKG)

build/_build/html/d2l-en.zip: $(OBJ) $(DEPS)
	cd build; zip -r $(patsubst build/%, %, $@ $(DEPS)) chapter*/*md chapter*/*ipynb

# Copy XX to build/XX if build/XX is depended (e.g., $(DEPS))
build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

html: $(DEPS) $(FRONTPAGE_DEP) $(OBJ)
	make -C build html
	python build/utils/post_html.py
	# Enable horitontal scrollbar for wide code blocks
	sed -i s/white-space\:pre-wrap\;//g build/_build/html/_static/sphinx_materialdesign_theme.css
	cp -r img/frontpage/ build/_build/html/_images/

TEX=build/_build/latex/d2l-en.tex

#build/_build/latex/%.pdf: img/%.svg
#	@mkdir -p $(@D)
#	rsvg-convert -f pdf -z 0.80 -o $@ $<

#SVG=$(wildcard img/*.svg)

#PDFIMG = $(patsubst img/%.svg, build/_build/latex/%.pdf, $(SVG))

pdf: $(DEPS) $(OBJ)
	make -C build latex
	sed -i s/\\.svg/.pdf/g ${TEX}
	sed -i s/\}\\.gif/\_00\}.pdf/g $(TEX)
	sed -i s/{tocdepth}{0}/{tocdepth}{1}/g $(TEX)
	sed -i s/{\\\\releasename}{Release}/{\\\\releasename}{}/g $(TEX)
	sed -i s/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\}\\\]/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\},formatcom=\\\\footnotesize\\\]/g $(TEX)
	sed -i s/\\\\usepackage{geometry}/\\\\usepackage[paperwidth=187mm,paperheight=235mm,left=20mm,right=20mm,top=20mm,bottom=15mm,includefoot]{geometry}/g $(TEX)
	sed -i s/\\\\maketitle/\\\\maketitle\ \\\\pagebreak\\\\hspace{0pt}\\\\vfill\\\\begin{center}This\ draft\ is\ a\ testing\ version\ \(draft\ date:\ \\\\today\).\\\\\\\\\ Visit\ \\\\url{https:\\/\\/d2l.ai}\ to\ obtain\ a\ later\ or\ release\ version.\\\\end{center}\\\\vfill\\\\hspace{0pt}\\\\pagebreak/g $(TEX)
	sed -i s/’/\\\'/g ${TEX}
	# Replace nbsp to space in latex
	sed -i s/ /\ /g ${TEX}
	# Allow figure captions to include space and autowrap
	sed -i s/Ⓐ/\ /g ${TEX}
	# Remove un-translated long table descriptions
	sed -i /\\\\multicolumn{2}{c}\%/d $(TEX)
	sed -i /\\\\sphinxtablecontinued{Continued\ on\ next\ page}/d $(TEX)
	sed -i /{\\\\tablename\\\\\ \\\\thetable{}\ --\ continued\ from\ previous\ page}/d $(TEX)

	python build/utils/post_latex.py en

	cd build/_build/latex && \
	bash ../../utils/convert_output_svg.sh && \
	buf_size=10000000 xelatex d2l-en.tex && \
	buf_size=10000000 xelatex d2l-en.tex

clean:
	rm -rf build/chapter* build/_build build/img build/data build/environment.yml build/README.md $(PKG)
