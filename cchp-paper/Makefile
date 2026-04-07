LATEXMK = latexmk
LATEXMKRC = .latexmkrc
MAIN = main.tex
BUILD_DIR = build

.PHONY: pdf clean distclean

pdf:
	$(LATEXMK) -r $(LATEXMKRC) $(MAIN)

clean:
	$(LATEXMK) -r $(LATEXMKRC) -c $(MAIN)

distclean:
	$(LATEXMK) -r $(LATEXMKRC) -C $(MAIN)

