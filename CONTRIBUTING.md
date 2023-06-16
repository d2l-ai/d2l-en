# Guidelines for contributing

Thank you for your interest in contributing to this open source book! We greatly value feedback and contributions from our community.

Please read through this document before you submit any pull requests or issues. It will help us work together more effectively.

## What to expect when you contribute

When you submit a pull request, our team is notified and will respond as quickly as we can. We'll do our best to work with you to ensure that your pull request adheres to our style and standards. If we merge your pull request, we might make additional edits later for style or clarity.

The source files on GitHub aren't published directly to the official website. If we merge your pull request, we'll publish your changes to the documentation website as soon as we can, but they won't appear immediately or automatically.

We look forward to receiving your pull requests for:

* New content you'd like to contribute (such as new code samples or tutorials)
* Inaccuracies in the content
* Information gaps in the content that need more detail to be complete
* Typos or grammatical errors
* Suggested rewrites that improve clarity and reduce confusion

**Note:** We all write differently, and you might not like how we've written or organized something currently. We want that feedback. But please be sure that your request for a rewrite is supported by the previous criteria. If it isn't, we might decline to merge it.

## How to contribute

To contribute, start by reading [contributing section](https://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html) and eventually
send us a pull request. For small changes, such as fixing a typo or adding a link, you can use the [GitHub Edit Button](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files). For larger changes:

1. [Fork the repository](https://help.github.com/articles/fork-a-repo/).
2. In your fork, make your change in a new branch (e.g., by [`git branch`](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)) that's based on this repo's **master** branch.
3. Commit the change to your fork, using a clear and descriptive commit message.
4. [Create a pull request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/), answering any questions in the pull request form.

Before you send us a pull request, please be sure that:

1. You're working from the latest source on the **master** branch.
2. You check [existing open](https://github.com/d2l-ai/d2l-en/pulls), and [recently closed](https://github.com/d2l-ai/d2l-en/pulls?q=is%3Apr+is%3Aclosed), pull requests to be sure that someone else hasn't already addressed the problem.
3. You [create an issue](https://github.com/d2l-ai/d2l-en/issues/new) before working on a contribution that will take a significant amount of your time.

For contributions that will take a significant amount of time, [open a new issue](https://github.com/d2l-ai/d2l-en/issues/new) to pitch your idea before you get started. Explain the problem and describe the content you want to see added to the documentation. Let us know if you'll write it yourself or if you'd like us to help. We'll discuss your proposal with you and let you know whether we're likely to accept it. We don't want you to spend a lot of time on a contribution that might be outside the scope of the documentation or that's already in the works.

## Finding contributions to work on

If you'd like to contribute, but don't have a project in mind, look at the [open issues](https://github.com/d2l-ai/d2l-en/issues) in this repository for some ideas. Issues with the [help wanted](https://github.com/d2l-ai/d2l-en/labels/help%20wanted), [good first issue](https://github.com/d2l-ai/d2l-en/labels/good%20first%20issue) or [enhancement](https://github.com/d2l-ai/d2l-en/labels/enhancement) labels are a great place to start.

In addition to written content, we really appreciate new examples and code samples for our documentation, such as examples for different platforms or environments, and code samples in additional languages.


## How to change code in one of the frameworks?

This section describes the development environment setup and workflow
which should be followed when modifying/porting python code and making
changes to one of the machine learning frameworks in the book.
We follow a set of pre-defined [style guidelines](https://github.com/d2l-ai/d2l-en/blob/master/STYLE_GUIDE.md)
for consistent code quality throughout the book and expect the same
from our community contributors. You may need to check other chapters
from other contributors as well for this step.

All the chapter sections are generated from markdown (.md file, not .ipynb file)
source files. When making changes in code, for the ease of development
and making sure it is error free, we never edit the markdown files directly.
Instead we can read/load these markdown files as jupyter notebooks
and then make the required changes in the notebook to edit the markdown
file automatically (more on that below). This way, before raising the PR,
one can easily test the changes locally in the jupyter notebook.

Start by cloning the repo.

* Clone your d2l-en repo fork to a local machine.
```
git clone https://github.com/<UserName>/d2l-en.git
```

* Setup your local environment: Create an empty conda environment
(you may refer to our [Miniconda Installation](https://d2l.ai/chapter_installation/index.html#installing-miniconda) section in the book).

* Install the required packages after activating the environment.
What are the required packages? This depends on the framework you wish to edit. Note that master and release branches may have different
versions of a framework. For more details, you may refer to our [installation section](https://d2l.ai/chapter_installation/index.html).
See example installation below:

```bash
conda activate d2l

# PyTorch
pip install torch==<version> torchvision==<version>
# pip install torch==2.0.0 torchvision==0.15.0

# MXNet
pip install mxnet==<version>
# pip install mxnet==1.9.1
# or for gpu
# pip install mxnet-cu112==1.9.1

# Tensorflow
pip install tensorflow==<version> tensorflow-probability==<version>
# pip install tensorflow==2.12.0 tensorflow-probability==0.19.0
```

Compilation of the book is powered by the
[`d2lbook`](https://github.com/d2l-ai/d2l-book) package.
Simply run `pip install git+https://github.com/d2l-ai/d2l-book` in the
d2l conda environment to install the package.
We'll explain some of the basic `d2lbook` features below. 

NOTE: `d2l` and `d2lbook` are different packages. (avoid any confusion)

* Install the `d2l` library in development mode (only need to run once)

```bash
# Inside root of local repo fork
cd d2l-en

# Install the d2l package
python setup.py develop
```

Now you can use `from d2l import <framework_name> as d2l` within the
environment to access the saved functions and also edit them on the fly.

When adding a code cell from a specific framework, one needs to specify
the framework by commenting the following on top of a cell: `#@tab tensorflow`
for example. If the code tab is exactly the same for all frameworks then
use `#@tab all`. This information is required by the `d2lbook` package to
build the website, pdf, etc. We recommend looking at some of the notebooks
for reference.


### How to open/edit markdown files using Jupyter Notebook?

Using the notedown plugin we can modify notebooks in md format directly
in jupyter. First, install the notedown plugin, run jupyter, and
load the plugin as shown below:

```bash
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

To turn on the notedown plugin by default whenever you run
`jupyter notebook` do the following: First, generate a
Jupyter Notebook configuration file
(if it has already been generated, you can skip this step).

```bash
jupyter notebook --generate-config
```

Then, add the following line to the end of the Jupyter Notebook
configuration file (for Linux/macOS, usually in the path `~/.jupyter/jupyter_notebook_config.py`):

```bash
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

After that, you only need to run the jupyter notebook
command to turn on the notedown plugin by default.

Please refer to the section on [markdown files in jupyter](https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#markdown-files-in-jupyter)
for more details.


#### d2lbook activate

Now to start working on a particular framework for a section,
only activate the framework tab you wish to use,
like this -> `d2lbook activate <framework_name> chapter_preliminaries/ndarray.md`,
so the `<framework_name>` code blocks become python blocks and
other frameworks are ignored when running the notebook.

When you are done editing a notebook, please save it and
remember to strictly clear all outputs and activate all
tabs by using `d2lbook activate`.

```bash
# Example
d2lbook activate all chapter_preliminaries/ndarray.md`
```

#### d2lbook build lib

Note: Remember to mark a function which will be reused later by
`#save` and in the end when all the above steps are completed
just run the following in the root directory to copy all the
saved functions/classes into `d2l/<framework_name>.py`

```bash
d2lbook build lib
```

If the saved functions require some packages to be imported, you can add
them to `chapter_preface/index.md` under the respective framework tab and
run `d2lbook build lib`. Now the import will also be reflected in the d2l
library after running and the saved functions can access the imported lib.

NOTE: Ensure that the output/results are consistent after the change, across the frameworks, by multiple runs of the notebook locally.


Finally send in a PR, if all checks succeed, with a review of the PR from the authors, your contributions shall be merged. :)

Hope this is comprehensive enough to get you started. Feel free to ask the authors and other contributors in case of any doubt. We always welcome feedback.

## Code of conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information, see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact [opensource-codeofconduct@amazon.com](mailto:opensource-codeofconduct@amazon.com) with any additional questions or comments.

## Security issue notifications

If you discover a potential security issue, please notify AWS Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public issue on GitHub.

## Licensing

See the [LICENSE](https://github.com/d2l-ai/d2l-en/blob/master/LICENSE) file for this project's licensing. We will ask you to confirm the licensing of your contribution. We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
