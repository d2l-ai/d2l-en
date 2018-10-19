all: html

build/%.ipynb: %.md build/env.yml build/md2ipynb.py
	@mkdir -p $(@D)
	cd $(@D); python ../md2ipynb.py ../../$< ../../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

ORIGN_DEPS = $(wildcard img/* data/* ) environment.yml README.md
DEPS = $(patsubst %, build/%, $(ORIGN_DEPS))

PKG = build/_build/html/d2l-en.tar.gz build/_build/html/d2l-en.zip

pkg: $(PKG)

build/_build/html/d2l-en.zip: $(OBJ) $(DEPS)
	cd build; zip -r $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/_build/html/d2l-en.tar.gz: $(OBJ) $(DEPS)
	cd build; tar -zcvf $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/%: %
	@mkdir -p $(@D)
	@cp -r $< $@

html: $(DEPS) $(OBJ)
	make -C build html
	cp build/index.html build/_build/html/
	cp -r img/frontpage/ build/_build/html/_images/

TEX=build/_build/latex/d2l-en.tex

build/_build/latex/%.pdf: img/%.svg
	@mkdir -p $(@D)
	rsvg-convert -f pdf -z 0.80 -o $@ $<

SVG=$(wildcard img/*.svg)

PDFIMG = $(patsubst img/%.svg, build/_build/latex/%.pdf, $(SVG))

pdf: $(DEPS) $(OBJ) $(PDFIMG)
	@echo $(PDFIMG)
	make -C build latex
	sed -i s/\\.svg/.pdf/g ${TEX}
	sed -i s/\}\\.gif/\_00\}.pdf/g $(TEX)
	sed -i s/{tocdepth}{0}/{tocdepth}{1}/g $(TEX)
	sed -i s/{\\\\releasename}{发布}/{\\\\releasename}{}/g $(TEX)
	sed -i s/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\}\\\]/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\},formatcom=\\\\footnotesize\\\]/g $(TEX)
	sed -i s/\\\\usepackage{geometry}/\\\\usepackage[paperwidth=187mm,paperheight=235mm,left=20mm,right=20mm,top=20mm,bottom=15mm,includefoot]{geometry}/g $(TEX)
	# Remove un-translated long table descriptions
	sed -i /\\\\multicolumn{2}{c}\%/d $(TEX)
	sed -i /\\\\sphinxtablecontinued{Continued\ on\ next\ page}/d $(TEX)
	sed -i /{\\\\tablename\\\\\ \\\\thetable{}\ --\ continued\ from\ previous\ page}/d $(TEX)
	cd build/_build/latex && \
	buf_size=10000000 xelatex d2l-en.tex && \
	buf_size=10000000 xelatex d2l-en.tex

clean:
	rm -rf build/chapter* build/_build $(DEPS) $(PKG)
