# IBIO-Banyuls2023-Python


## Working with Google Colab

### Opening Colab

Got to [Colab's website](https://colab.research.google.com/), and in the popup window click on **Github**, then in the seach bar paste the following :
```
https://github.com/EmeEmu/IBIO-Banyuls2023-Python
```
You can then select the notebook you want to open, for example `day1_discovering_the_data.ipynb`.

### Downloding the Helper files

In this repo, you will find a buch of helper functions in the sub-directory `Helper_Functions`. To load them into Colab run the following code in a cell :
```
!mkdir /content/Helper_Functions/
!wget -P /content/Helper_Functions/ https://raw.githubusercontent.com/EmeEmu/IBIO-Banyuls2023-Python/main/Helper_Functions/accessing_data.py
!wget -P /content/Helper_Functions/ https://raw.githubusercontent.com/EmeEmu/IBIO-Banyuls2023-Python/main/Helper_Functions/hmm_plotters.py
!wget -P /content/Helper_Functions/ https://raw.githubusercontent.com/EmeEmu/IBIO-Banyuls2023-Python/main/Helper_Functions/OrthoViewer.py
```

### Downloading Data

All the data for this Python workshop is located in the following [Google Drive](https://drive.google.com/drive/folders/1k21VhLoonOnoxxXyswrmE45VIB4FF00n?usp=sharing "Link to the Google Drive"). To work on this data from Colab, it needs to be downloaded to Colab's cloud storage.

Within Colab, you can download the entire folder by running the folling code :
```
!gdown --folder 1k21VhLoonOnoxxXyswrmE45VIB4FF00n
```
this will download all the files from the Google Drive to `/content/banyuls_data/`.

You can also download indivual files by providing the unique identifier in the url of the file in the following code. For example, for the file [https://drive.google.com/file/d/1jIQw8EEIoS516plFFXtRIwS-0oey7kFt/view?usp=sharing](https://drive.google.com/file/d/1jIQw8EEIoS516plFFXtRIwS-0oey7kFt/view?usp=sharing), the identifier is `1jIQw8EEIoS516plFFXtRIwS-0oey7kFt` :

```
!gdown 1jIQw8EEIoS516plFFXtRIwS-0oey7kFt
```
and the file will be downloaded as `/content/fish1_different_directions.hdf5`

### Setting up Matplotlib for interactive plots

Working with interactive plots in Colab requires a little bit of preparation. You should run the following code :
```python
!pip install ipympl
from google.colab import output
output.enable_custom_widget_manager()
%matplotlib widget
```


## Working locally

### Setting up the environnement

```bash
conda create --file ibio_env.txt --name ibio
```
