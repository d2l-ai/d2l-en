# How to Contribute to This Book

Contributions by readers [1] help us improve this book. If you find a typo, an outdated link, something where you think we missed a citation, where the code doesn't look elegant or where an explanation is unclear, please contribute back and help us help our readers. While in regular books the delay between print runs (and thus between typo corrections) can be measured in years, it typically takes hours to days to incorporate an improvement in this book. This is all possible due to version control and continuous integration testing. To do so you need to install Git and submit a pull request [2] to the GitHub repository. When your pull request is merged into the code repository by the author, you will become a contributor. In a nutshell the process works as described in the diagram below. 

![Contributing to the book.](../img/contribute.svg)

## From Reader to Contributor in 6 Steps

We will walk you through the steps in detail. If you are already familiar with Git you can skip this section. For concreteness we assume that the contributor's user name is `smolix`. 

### Install Git

The Git open source book [3] describes how to install Git. This typically works via `sudo apt install git` on Ubuntu Linux, by installing the Xcode developer tools on macOS, or by using GitHub's [desktop client](https://desktop.github.com). If you don't have a GitHub account, you need to sign up for one [4].

### Log in to GitHub

Enter the address of the book's code repository in your browser [2]. Click on the `Fork` button in the red box at the top-right of the figure below, to make a copy of the repository of this book. This is now *your copy* and you can change it any way you want. 

![The code repository page.](../img/git-fork.png)

Now, the code repository of this book will be copied to your username, such as `smolix/d2l-en` shown at the top-left of the screenshot below.

![Copy the code repository.](../img/git-forked.png)

### Clone the Repository

To clone the repository (i.e. to make a local copy) we need to get its repository address. The green button on the picture below displays this. Make sure that your local copy is up to date with the main repository if you decide to keep this fork around for longer. For now simply follow the instructions in the [Installation](../chapter_prerequisite/install.md) section to get started. The main difference is that you're now downloading *your own fork* of the repository. 

![Git clone.](../img/git-clone.png)

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```

On Unix, the above command copies all the code from GitHub to the directory `d2l-en`. 

### Edit the Book and Push

Now it's time to edit the book. It's best to edit the notebooks in Jupyter following the [instructions](../chapter_appendix/jupyter.md) in the appendix. Make the changes and check that they're OK. Assume we have modified a typo in the file `~/d2l-en/chapter_appendix/how-to-contribute.md`. 
You can then check which files you have changed:

```
git status
```

At this point Git will prompt that the `chapter_appendix/how-to-contribute.md` file has been modified.

```
mylaptop:d2l-en smola$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix/how-to-contribute.md
```

After confirming that this is what you want, execute the following command:

```
git add chapter_appendix/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

The changed code will then be in your personal fork of the repository. To request the addition of your change, you have to create a pull request for the official repository of the book.

### Pull Request

Go to your fork of the repository on GitHub and select "New pull request". This will open up a screen that shows you the changes between your edits and what is current in the main repository of the book. 

![Pull Request.](../img/git-newpr.png)


### Submit Pull Request

Finally, submit a pull request. Make sure to describe the changes you have made in the pull request. This will make it easier for the authors to review it and to merge it with the book. Depending on the changes, this might get accepted right away, rejected, or more likely, you'll get some feedback on the changes. Once you've incorporated them, you're good to go. 

![Create Pull Request.](../img/git-createpr.png)

Your pull request will appear among the list of requests in the main repository. We will make every effort to process it quickly. 

## Summary

* You can use GitHub to contribute to this book.
* Forking a repositoy is the first step to contributing, since it allows you to edit things locally and only contribute back once you're ready.
* Pull requests are how contributions are being bundled up. Try not to submit huge pull requests since this makes them hard to understand and incorporate. Better send several smaller ones. 

## Exercises

1. Star and fork the `d2l-en` repository. 
1. Find some code that needs improvement and submit a pull request.
1. Find a reference that we missed and submit a pull request. 


## References

[1] List of contributors to this book. https://github.com/d2l-ai/d2l-en/graphs/contributors

[2] Address of the code repository of this book. https://github.com/d2l-ai/d2l-en

[3] Install Git. https://git-scm.com/book/zh/v2

[4] URL of GitHub. https://github.com/

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2401)

![](../img/qr_how-to-contribute.svg)
