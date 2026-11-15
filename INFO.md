# Kūrimas (build)

## Diegimas kūrėjams

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh  # Py3.8 atveju: wget  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b  # Py3.8 atveju: sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
. ~/.bashrc
conda create --name d2l python=3.9 -y  # Py3.8 atveju: conda create --name d2l python=3.8 -y
conda activate d2l
pip install torch torchvision
pip install d2lbook
git clone https://github.com/d2l-ai/d2l-en.git
jupyter notebook --generate-config
echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >> ~/.jupyter/jupyter_notebook_config.py
cd d2l-en
pip install -e .  # Įdiegti d2l biblioteką iš šaltinio
jupyter notebook
```

Pasirinktinai: naudojant `jupyter_contrib_nbextensions`

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
# jupyter nbextension enable execute_time/ExecuteTime
```



## Kūrimas be vykdymo (evaluation)

Faile `config.ini` pakeiskite `eval_notebook = True` į `eval_notebook = False`.


## PDF kūrimas

```
# Įdiegti d2lbook
pip install git+https://github.com/d2l-ai/d2l-book

sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
sudo apt-get install pandoc  # Jei neveikia, conda install pandoc

# Kad būtų galima importuoti d2l
cd d2l-en
pip install -e .

# Sukurti PDF
d2lbook build pdf
```

### PDF šriftai

```
wget https://raw.githubusercontent.com/d2l-ai/utils/master/install_fonts.sh
sudo bash install_fonts.sh
```


## HTML kūrimas

```
bash static/build_html.sh
```

## Šriftų diegimas

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

## Išleidimo kontrolinis sąrašas

### d2l-en

- išleisti d2lbook
- [pasirinktinai, tik spausdintoms knygoms ar partnerių produktams]
    - užfiksuoti bibliotekų versijas [setup.py](http://setup.py) → requirements ir static/build.yml (įskaitant d2lbook)
    - iš naujo įvertinti (re-evaluate)
    - diegime užfiksuoti d2l versiją (kuri toliau atsiras pypi)
- pridėti docstring d2l.xxx
- atnaujinti pradinio puslapio skelbimą
- (tik didesniam leidimui) wa 0.8.0, kad patikrintume, ar pagrindiniame tekste ką nors reikia taisyti
- d2lbook build lib
- išbandyti atsitiktinį colab'ą
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-en-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- Įvertinti išleidimo PR
- užtikrinti fiksuotą dėmesio atsitiktinumą Badahnau ir transformerio skyriuose
- užtikrinti, kad bibliotekų (pvz., sagemaker) versijos sutaptų `config.ini` ir `build.yml`
- pakeisti versijos numerį `config.ini` ir `d2l/__init__.py`, o d2l versiją – `installation.md`
- sulieti master į release išsaugant atskirus commit'us (sukurti merge commit)
- git checkout master
- rr -rf d2l.egg-info dist
- įkelti d2l į pypi (komandos paskyra)
- iš naujo išbandyti colab ir d2l
- git tag release šakoje
- git checkout master
- atnaujinti naujausią versiją README atskiroje šakoje, tada squash ir merge
- [pasirinktinai] panaikinti CloudFront talpyklą
- [pasirinktinai, tik spausdintoms knygoms]
    - config.ini: other_file_s3urls
- [pasirinktinai, tik spausdintoms knygoms ar partnerių produktams]
    - atkurti bibliotekų versijas [setup.py](http://setup.py) → requirements
 
### d2l-zh

- atnaujinti pradinio puslapio skelbimą
- (ar reikia?) d2lbook build lib
- išbandyti atsitiktinį colab'ą
- atnaujinti static/build.yml iki d2l-en versijos
- [http://ci.d2l.ai/computer/(master)/script](http://ci.d2l.ai/computer/(master)/script)
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-zh-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- Įvertinti išleidimo PR (sutvarkyti dėmesio atsitiktinumą Badahnau ir transformerio skyriuose)
- užtikrinti, kad bibliotekų (pvz., sagemaker) versijos sutaptų `config.ini` ir `build.yml`
- pakeisti versijos numerį `config.ini` ir `d2l/__init__.py`
- sulieti master į release išsaugant atskirus commit'us (sukurti merge commit)
- iš naujo išbandyti colab
- git tag release šakoje
- git checkout master
- atnaujinti naujausią versiją README atskiroje šakoje, tada squash ir merge
- papildomai 2.0.0 išleidimui
    - s3 konsolėje
        - nukopijuoti [zh-v2.d2l.ai](http://zh-v2.d2l.ai) bucket/d2l-zh.zip į d2l-webdata bucket/d2l-zh.zip
        - pervadinti d2l-webdata bucket/d2l-zh.zip į d2l-webdata bucket/d2l-zh-2.0.0.zip
        - paleisti CI d2l-zh/release, kad suveiktų other_file_s3urls konfigūracijoje
        - panaikinti cloudfront talpyklą, kad būtų galima išbandyti diegimą
    - išbandyti diegimą
