<h1>Datasets</h1>

- [**TrueFace**](#trueface-dataset)
- [**TRUEFAKE**](#truefake-dataset)
- [**FORLAB** only Real Images](#forlab-dataset)
- [**DataLoaders**][#dataloaders]

---

## TrueFace 

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

### TrueFake Dataset
The **TrueFake** is an experimental (in development) dataset that includes a variety of fake images.

- **Models**: FLUX.1, StableDiffusion1.5, StableDiffusion2, StableDiffusion3, StableDiffusionXL, StyleGAN3

**NOTE** labels? where are the real images?????

- **TrueFake**: `/media/NAS/TrueFake`
  - **Extension**
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
### FORLAB Dataset
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
    - smartphones
