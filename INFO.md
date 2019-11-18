# Building

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
python setup.py develop

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
d2lbook build html
```
