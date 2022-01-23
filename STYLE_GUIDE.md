# STYLE GUIDE

## In General

* Be precise, clear, engaging, pragmatic, and consistent

## Text

* Chapters and Sections
    * Provide an overview at the beginning of each chapter
    * Be consistent in the structure of each section
        * Summary
        * Exercises
* Quotes
    * Use double quotes
* Symbol Descriptions
    * timestep t（not t timestep）
* Tools, Class, and Functions
    * Gluon, MXNet, NumPy, spaCy, NDArray, Symbol, Block, HybridBlock, ResNet-18, Fashion-MNIST, matplotlib
        * Consider these as words without accents (``)
    * Sequential class/instance, HybridSequential class/instance
        * Without accents (``)
    * `backward` function
        * not `backward()` function
    * "for-loop" not "for loop"
* Terminologies
    * Consistently use
        * function (not method)
        * instance (not object)
        * weight, bias, label
        * model training, model prediction (model inference)
        * training/testing/validation dataset
        * prefer using "data/training/testing example" over "data instance" or "data point"
    * Distinguish：
        * hyperparameter vs parameter
        * minibatch stochastic gradient descent vs stochastic gradient descent
* Use numerals when they are explaining or part of code or math.
* Acceptable abbreviations
    * AI, MLP, CNN, RNN, GRU, LSTM, model names (e.g., ELMo, GPT, BERT)
    * We spell out full names in most cases to be clear (e.g., NLP -> natural language processing)

## Math

* Be consistent in [math notation](chapter_notation/index.md)
* Place punctuations within equations if necessary
    * e.g., comma and period
* Assignment symbol
    * \leftarrow
* Use mathematical numerals only when they are part of math: "$x$ is either $1$ or $-1$", "the greatest common divisor of $12$ and $18$ is $6$".
* We do not use "thousands separator" (since different publishing houses have different styles). E.g., 10,000 should be written as 10000 in the source markdown files.

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
        * Arial (for text), STIXGeneral (for math), 9pt（subscripts/superscripts：6pt）
        * Do not italicize numbers or parentheses in subscripts or superscripts
    * Color：
        * Blue as background (text is black)
            * (Try to avoid) Extra Dark：3FA3FD
            * Dark：66BFFF
            * Light：B2D9FF
            * (Try to avoid) Extra Light: CFF4FF
* Be careful about copyright


## Code

* Each line must have <=78 characters (limited by page width)
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
    * No period at the end of comments.
    * For clarity, surround variable names with accents, e.g.,  # shape of `X`
* imports
    * import alphabetically
* Print variables
    * if possible use `x, y` instead of `print('x:', x, 'y:', y)` at the end of the code block
* String
    * Use single quotes
    * Use f-strings. To break a long f-string into multi-lines, just use one f-string per line.
* Other items
    * `nd.f(x)` → `x.nd`
    * `.1` → `1.0`
    * 1. → `1.0`


## References

* Refer to [d2lbook](https://book.d2l.ai/user/markdown.html#cross-references) on how to add references for figure, table and equations.



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
4. Run `bibtool -s -f "%3n(author).%d(year)" d2l.bib -o d2l.bib`. Now the added entry will have consistent keys. And as a side-effect, it'll appear in alphabetically sorted order relative to all other papers in the file:
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


## Edit and Test Code in One Framework

1. Say we want to edit and test MXNet code in xx.md, run `d2lbook activate default xx.md`. Then code of other frameworks is deactivated in xx.md.
2. Open xx.md using Jupyter notebook, edit code and use "Kernel -> Restart & Run All" to test code.
3. Run `d2lbook activate all xx.md` to re-activate code of all the frameworks. Then git push.

Likewise, `d2lbook activate pytorch/tensorflow xx.md` will only activate PyTorch/TensorFlow code in xx.md.
