all: html

build/%.ipynb: %.md build/env.yml $(wildcard gluonbook/*)
	mkdir -p $(@D)
	cd $(@D); python ../utils/md2ipynb.py ../../$< ../../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

FRONTPAGE_DIR = img/frontpage
FRONTPAGE = $(wildcard $(FRONTPAGE_DIR)/*)
FRONTPAGE_DEP = $(patsubst %, build/%, $(FRONTPAGE))

IMG_NOTEBOOK = $(filter-out $(FRONTPAGE_DIR), $(wildcard img/*))
ORIGIN_DEPS = $(IMG_NOTEBOOK) $(wildcard data/* gluonbook/*) environment.yml README.md
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
	sed -i s/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\}\\\]/{OriginalVerbatim}\\\[commandchars=\\\\\\\\\\\\{\\\\},formatcom=\\\\footnotesize\\\]/g $(TEX)
	sed -i s/\\\\usepackage{geometry}/\\\\usepackage[paperwidth=187mm,paperheight=235mm,left=20mm,right=20mm,top=20mm,bottom=15mm,includefoot]{geometry}/g $(TEX)
	# Remove un-translated long table descriptions
	sed -i /\\\\multicolumn{2}{c}\%/d $(TEX)
	sed -i /\\\\sphinxtablecontinued{Continued\ on\ next\ page}/d $(TEX)
	sed -i /{\\\\tablename\\\\\ \\\\thetable{}\ --\ continued\ from\ previous\ page}/d $(TEX)
	cd build/_build/latex && \
	bash ../../utils/convert_output_svg.sh && \
	buf_size=10000000 xelatex d2l-en.tex && \
	buf_size=10000000 xelatex d2l-en.tex

clean:
	rm -rf build/chapter* build/_build build/img build/data build/environment.yml build/README.md $(PKG)
