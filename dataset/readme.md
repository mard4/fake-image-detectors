# **Datasets and DataLoaders**

<h1>Datasets Available</h1>

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="#trueface-dataset"><strong>TrueFace Dataset</strong></a></td>
      <td>A dataset containing real and fake images of faces pre and post social network from different models.</td>
    </tr>
    <tr>
      <td><a href="#truefake-dataset"><strong>TRUEFAKE Dataset</strong></a></td>
      <td>A dataset containing real and fake images pre and post social network from different models and categories (faces, landscapes, etc).</td>
    </tr>
    <tr>
      <td><a href="#forlab-dataset"><strong>FORLAB Dataset</strong></a></td>
      <td>Exclusively includes real images for specialized tasks (camera recognition, PNRU, etc) — not useful for our task.</td>
    </tr>
  </tbody>
</table>


<h1>DataLoaders</h1>
<ul>
  <li>
    <h3><a href="#dataloaders">DataLoaders</a></h3>
    <p>Flexible loaders to the data preparation pipeline, supporting custom models and folders, transformations, batching, etc.</p>
  </li>
</ul>


------------
<h2 href="#trueface-dataset">TrueFace</h2>

Dataset containing only Faces (Pre Social Network and Post Social Network image processing). 
It contains both Real and Fake images from different architectures and different social Networks.
- **Models**: StyleGAN, StyleGAN2
- **Social Networks**: Facebook, Telegram, Twitter, Whatsapp

Directory Structure

- **TrueFace**: `/media/NAS/TrueFace/TrueFace`
  - **TrueFace_PreSocial**
    - **Real**
      - `FFHQ`
    - **Fake**
      - **StyleGAN**
        - `images-psi-0.5`
        - `images-psi-0.7`
      - **StyleGAN2**
        - `conf-f-psi-0.5`
        - `conf-f-psi-1`
  - **TrueFace_PostSocial**
    - **Facebook**
      - **Real**
        - `FFHQ`
      - **Fake**
        - **StyleGAN**
          - `images-psi-0.5`
          - `images-psi-0.7`
        - **StyleGAN2**
          - `conf-f-psi-0.5`
          - `conf-f-psi-1`
    - **Telegram**
      - **Real**
        - `FFHQ`
      - **Fake**
        - **StyleGAN**
          - `images-psi-0.5`
          - `images-psi-0.7`
        - **StyleGAN2**
          - `conf-f-psi-0.5`
          - `conf-f-psi-1`
    - **Twitter**
      - **Real**
        - `FFHQ`
      - **Fake**
        - **StyleGAN**
          - `images-psi-0.5`
          - `images-psi-0.7`
        - **StyleGAN2**
          - `conf-f-psi-0.5`
          - `conf-f-psi-1`
    - **Whatsapp**
      - **Real**
        - `FFHQ`
      - **Fake**
        - **StyleGAN**
          - `images-psi-0.5`
          - `images-psi-0.7`
        - **StyleGAN2**
          - `conf-f-psi-0.5`
          - `conf-f-psi-1`

---------------------------------------------------------------------

<h2 href="#truefake-dataset">TRUEFAKE</h2>

Dataset containing Pre Social Network images of different categories. 

- **Models**: FLUX.1, StableDiffusion1.5, StableDiffusion2, StableDiffusion3, StableDiffusionXL, StyleGAN3

- **TrueFake**: `/media/NAS/TrueFake`
  - **PreSocial**
    - **Real**
      - FFHQ  
      - FORLAB
   - **Fake**
      - **FLUX.1**
        - animals
        - faces
        - general  
        - landscapes
      - **StableDiffusion1.5**
        - animals
        - animals_backup
        - faces
        - general
        - landscapes
      - **StableDiffusion2**
        - animals
        - animals_backup
        - faces
        - general
        - landscapes
      - **StableDiffusion3**
        - animals
        - animals_backup
        - faces
        - general
        - landscapes
      - **StableDiffusionXL**
        - animals
        - animals_backup
        - faces
        - general
        - landscapes
      - **StyleGAN3**
        - conf-t-psi-0.5  
        - conf-t-psi-1


---------------------------------------------------------------------
<!-- ### FORLAB Dataset
The **FORLAB** dataset contains only real images.

**NOTA** subfolders are not complete bc idk if it's useful


- **FORLAB**: `/media/NAS/FORLAB`
    - broken_images  
    - camera_fingerprint  
    - cameras
        - canon      
        - gopro  
        - 'leica camera ag'    
        - 'olympus corporation'   
        - sigma
        - fujifilm 
        - leica  
        - 'nikon corporation'   
        - panasonic              
        - sony
    - download.sh  
    - nohup.log  
    - out_download.log  
    - smartphones -->

----------------------------------------------------
<h2 href="#dataloaders"> Dataloaders</h2> 

in ModuleDataLoader.py

The `Datasettone` class navigates through the defined directories to extract images and their associated labels. It returns both the image and its corresponding label.

## TrueFace Dataloader
This dataloader loads data for both the 'Real' and 'Fake' classes, concatenating them for use in training or evaluation.

### Args:
- **main_category** (`str`): Main category of the dataset.
- **platform** (`str`): Platform name (e.g., 'Facebook').
- **styleGAN_type** (`str`): Type of StyleGAN used.
- **psi_value** (`str`): Psi value for StyleGAN.
- **batch_size** (`int`, optional): Batch size for the DataLoader.
- **transform** (`callable`, optional): Transformations to apply to the images.
- **max_images_per_class** (`int`, optional): Maximum number of images to load per class.

### Returns:
- **DataLoader**: A PyTorch DataLoader containing the combined dataset of both classes.

---

## TRUEFAKE Dataloader

This dataloader is designed to load both 'Fake' and 'Real' class images, with flexible configuration options for selecting models and folders.

## Features

- Load data from 'Fake' and 'Real' classes simultaneously.
- Select specific models and folders for 'Fake' images.
- Select specific folders for 'Real' images.
- Support for applying transformations and limiting the number of images per class.

## Args

- **batch_size** (`int`, optional): Batch size for the DataLoader. Default is `32`.
- **transform** (`callable`, optional): Transformations to apply to the images. If not provided, a default transformation pipeline is applied:
  - Resize images to `(224, 224)`.
  - Normalize with ImageNet mean and standard deviation.
- **max_images_per_class** (`int`, optional): Maximum number of images to load per class. If not specified, all available images are loaded.
- **ALL_MODELS** (`bool`, optional): 
  - `True` (default): Load all models from the 'Fake' images directory.
  - `False`: Load only the models specified in the `models` parameter.
- **models** (`list of str`, optional): List of model names to load when `ALL_MODELS` is `False`. Required if `ALL_MODELS` is `False`.
- **ALL_FOLDERS_FAKE** (`bool`, optional): 
  - `True` (default): Load all folders under each selected model in the 'Fake' images directory.
  - `False`: Load only the folders specified in the `folders_fake` parameter.
- **folders_fake** (`list of str`, optional): List of folder names to load under each selected model in the 'Fake' images directory. Required if `ALL_FOLDERS_FAKE` is `False`.
- **ALL_FOLDERS_REAL** (`bool`, optional): 
  - `True` (default): Load all folders in the 'Real' images directory.
  - `False`: Load only the folders specified in the `folders_real` parameter.
- **folders_real** (`list of str`, optional): List of folder names to load in the 'Real' images directory. Required if `ALL_FOLDERS_REAL` is `False`.

## Returns

- **DataLoader**: A PyTorch DataLoader containing the combined dataset for the selected configuration.

## Examples

### Load All Models and Folders

To load all available models and folders for both 'Fake' and 'Real' images:
```python
data_loader = DataloaderTRUEFAKE()
dataloader = data_loader.load_data()
```

### Load specific models and specific folders
```python
data_loader = DataloaderTRUEFAKE()
dataloader = data_loader.load_data(
    max_images_per_class=10,
    ALL_MODELS=False, 
    models=['StableDiffusion1.5'], 
    ALL_FOLDERS_FAKE=False, 
    folders_fake=['faces'],  # Folders under each model in 'Fake' images
    ALL_FOLDERS_REAL=False, 
    folders_real=['FFHQ']    # Folders in 'Real' images
)
```