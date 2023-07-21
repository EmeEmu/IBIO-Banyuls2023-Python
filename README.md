# IBIO-Banyuls2023-Python


## Working with Google Colab

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

## Working locally

### Setting up the environnement

```bash
conda env create --file ibio_env.txt
```
