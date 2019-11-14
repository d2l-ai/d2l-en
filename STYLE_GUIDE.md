# STYLE GUIDE

## In General

* Be clear, engaging, pragmatic, and consistent

## Text

* Chapters and Sections
    * Provide an overview at the beginning of each chapter
    * Be consistent in the structure of each section
        * Summary
        * Exercises
        * Scan the QR Code to Access Discussions
        * References (if any)
* Environments
    * Use double quotes
* Quotes
    * Use double quotes
* Symbol Descriptions
    * timestep t（not t timestep）
* Tools, Class, and Functions
    * Gluon, MXNet, NumPy, spaCy, NDArray, Symbol, Block, HybridBlock, ResNet-18, Fashion-MNIST, matplotlib
        * Consider these as words without accents (``)
    * Sequential class/instance, HybridSequential class/instance
        * Without accents (``)
    * `backward`function
        * not `backward()` function
    * for loop
* Terminologies
    * Consistently use
        * function (not method)
        * instance (not object)
        * weight, bias, label
        * model training, model prediction (model inference)
        * training/testing/validation dataset
    * Distinguish：
        * hyperparameter vs parameter
        * minibatch stochastic gradient descent vs stochastic gradient descent
    * List
        * https://github.com/mli/gluon-tutorials-zh/blob/master/TERMINOLOGY.md

## Math

* Be consistent in math format
    * https://github.com/goodfeli/dlbook_notation/blob/master/notation_example.pdf
* Reference
    * the equation above/below (Equation numbering is to be consolidated by the Press)
    * the N equations above/below
* Place punctuations within equations if necessary
    * e.g., comma and period
* Assignment sumbol
    * \leftarrow

## Figure

* Software
    * Use OmniGraffle to make figures.
      * Export pdf (infinite canvas) in 100%, then use pdf2svg to convert to svg
        * `ls *.pdf | while read f; do pdf2svg $f ${f%.pdf}.svg; done`
      * Do not export svg directly from Omnigraffle (font size may slightly change)
* Style
    * Size：
        * Horizontal：<= 400 pixels  (limited by page width)
        * Vertical：<= 200 pixels (exceptions may be made)
    * Thickness：
        * StickArrow
        * 1pt
        * arrow head size: 50%
    * Font：
        * STIXGeneral, 9pt（subscripts/superscripts：6pt）
        * Do not italize numbers or parentheses in subscripts or superscripts
    * Color：
        * Blue as background (text is black)
            * (Try to avoid) Extra Dark：3FA3FD
            * Dark：66BFFF
            * Light：B2D9FF
            * (Try to avoid) Extra Light: CFF4FF
* Be careful about copyright
* Reference
    * e.g., Figure 7.1 (manually)
* matplotlib

## Code

* Each line must have <=80 characters (limited by page width)
* Use utils.py to encapsulate classes/functions that are repetitively used
    * Give full implementation when it is used for the first time
* Python
    * PEP8
        * e.g., (https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator)
* To save space, put several assignments on the same line
  * e.g, `num_epochs, lr = 5, 0.1`
* Be consistent in variable names
    * `num_epochs`
        * number of epochs
    * `num_hiddens`
        * number of hidden units
    * `num_inputs`
        * number of inputs
    * `num_outputs`
        * number of outputs
    * `net`
        * model
    * `lr`
        * learning rate
    * `acc`
        * accuracy
    * During iterations
        * features：`X`
        * labels：`y`, `y_hat` or `Y`, `Y_hat`
        * `for X, y in data_iter`
    * Data sets：
        * features：`features` or `images`
        * labels：`labels`
        * DataLoader instance：`train_iter`, `test_iter`, `data_iter`
* Comments
    * Add period at the end of comments.
* imports
    * import alphabetically
    * `from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils`
* Print outputs
    * `epoch, lr, loss, train acc, time`
    * Around 5 lines
* Print variables
    * if possible use `x, y` instead of `print('x:', x, 'y:', y)` at the end of the code block
* String
    * Use single quotes
* Other items
    * `nd.f(x)` → `x.nd`
    * `random_normal` → `random.normal`
    * multiple imports
    * `.1` → `1.0`
    * 1. → `1.0`
    * remove namescope


## References

* Refer to [d2lbook](http://book.d2l.ai/examples/markdown.html#cross-references)how to add references for figure, table and equations.


## QR Code

* https://www.the-qrcode-generator.com/
    * 75pixel, SVG

## Citations


1. Run `pip install git+https://github.com/d2l-ai/d2l-book`
1. Use bibtool to generate consistent keys for bibtex entries. Install it by `brew install bib-tool`
1. Add an bibtex entry to `d2l.bib` on the root directory. Say the original entry is
```
@article{wood2011sequence,
  title={The sequence memoizer},
  author={Wood, Frank and Gasthaus, Jan and Archambeau, C{\'e}dric and James, Lancelot and Teh, Yee Whye},
  journal={Communications of the ACM},
  volume={54},
  number={2},
  pages={91--98},
  year={2011},
  publisher={ACM}
}
```
4. Run `bibtool -f "%3n(author).%d(year)" d2l.bib -o d2l.bib`. Now the added entry will have consistent keys:
```
@Article{	  Wood.Gasthaus.Archambeau.ea.2011,
  title		= {The sequence memoizer},
  author	= {Wood, Frank and Gasthaus, Jan and Archambeau, C{\'e}dric
		  and James, Lancelot and Teh, Yee Whye},
  journal	= {Communications of the ACM},
  volume	= {54},
  number	= {2},
  pages		= {91--98},
  year		= {2011},
  publisher	= {ACM}
}
```
5. In the text, use the following to cite the added paper:
```
:cite:`Wood.Gasthaus.Archambeau.ea.2011` 
```

