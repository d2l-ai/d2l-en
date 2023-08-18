# Building

## Installation for Developers

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh  # For py3.8, wget  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b  # For py3.8: sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
. ~/.bashrc
conda create --name d2l python=3.9 -y  # For py3.8: conda create --name d2l python=3.8 -y
conda activate d2l
pip install torch torchvision
pip install d2lbook
git clone https://github.com/d2l-ai/d2l-en.git
jupyter notebook --generate-config
echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >> ~/.jupyter/jupyter_notebook_config.py
cd d2l-en
pip install -e .  # Install the d2l library from source
jupyter notebook
```

Optional: using `jupyter_contrib_nbextensions`

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
# jupyter nbextension enable execute_time/ExecuteTime
```



## Building without Evaluation

Change `eval_notebook = True` to `eval_notebook = False` in `config.ini`.


## Building PDF

```
# Install d2lbook
pip install git+https://github.com/d2l-ai/d2l-book

sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
sudo apt-get install pandoc  # If not working, conda install pandoc

# To import d2l
cd d2l-en
pip install -e .

# Build PDF
d2lbook build pdf
```

### Fonts for PDF

```
wget https://raw.githubusercontent.com/d2l-ai/utils/master/install_fonts.sh
sudo bash install_fonts.sh
```


## Building HTML

```
bash static/build_html.sh
```

## Install Fonts

```
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

wget -O Inconsolata.zip https://www.fontsquirrel.com/fonts/download/Inconsolata
unzip Inconsolata -d Inconsolata
sudo mv Inconsolata /usr/share/fonts/opentype/

sudo fc-cache -f -v

```

## Release checklist

### d2l-en

- release d2lbook
- [optional, only for hardcopy books or partner products]
    - fix versions of libs in [setup.py](http://setup.py) → requirements and static/build.yml (including d2lbook)
    - re-evaluate
    - fix d2l version (to appear on pypi below) in installation
- add docstring for d2l.xxx
- update frontpage announcement
- (only major) wa 0.8.0 to see if anything needs to be fixed in the main text
- d2lbook build lib
- test a random colab
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-en-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- Evaluate release PR
- ensure fixed attention randomness in badahnau and transformer
- ensure libs (e.g., under sagemaker) version consistent between config.ini and build.yml
- modify version number in config.ini & d2l/__init__.py, and d2l version in installation.md
- merge master to release by keeping individual commits (create a merge commit)
- git checkout master
- rr -rf d2l.egg-info dist
- upload d2l to pypi (team account)
- re-test colab and d2l
- git tag on the release branch
- git checkout master
- update README latest version in a branch, then squash and merge to restore
- [optional] Invalidate CloudFront cache
- [optional, only for hardcopy books]
    - config.ini: other_file_s3urls
- [optional, only for hardcopy books or partner products]
    - restore versions of libs in [setup.py](http://setup.py) → requirements
 
### d2l-zh

- update frontpage announcement
- (need or not?) d2lbook build lib
- test a random colab
- upgrade static/build.yml to that in d2l-en
- [http://ci.d2l.ai/computer/(master)/script](http://ci.d2l.ai/computer/(master)/script)
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-zh-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- Evaluate release PR (fix attention randomness in badahnau and transformer)
- ensure libs (e.g., under sagemaker)version consistent between config.ini and build.yml
- modify version number in config.ini & d2l/__init__.py
- merge master to release by keeping individual commits (create a merge commit)
- re-test colab
- git tag on the release branch
- git checkout master
- update README latest version in a branch, then squash and merge to restore
- 2.0.0 release additional
    - on s3 console
        - copy [zh-v2.d2l.ai](http://zh-v2.d2l.ai) bucket/d2l-zh.zip to d2l-webdata bucket/d2l-zh.zip
        - rename d2l-webdata bucket/d2l-zh.zip to d2l-webdata bucket/d2l-zh-2.0.0.zip
        - run CI for d2l-zh/release to trigger other_file_s3urls in config
        - Invalidate cloudfront cache to test installation
    - test install
