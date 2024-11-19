# Datasets
- [**TrueFace** Dataset Faces (Pre Social Network and Post Social Network)](#trueface-dataset)
- [**TrueFake**](#truefake-dataset)
- [**FORLAB** only Real Images](#forlab-dataset)


---

## Dataset Faces (Pre Social Network and Post Social Network)

[Directory Structure](#trueface-directory-structure)

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


<!-- /media/NAS/
└── TrueFace
    ├── TrueFace_PreSocial
    │   ├── Real
    │   │   └── FFHQ
    │   └── Fake
    │       ├── StyleGAN
    │       │   ├── images-psi-0.5
    │       │   └── images-psi-0.7
    │       └── StyleGAN2
    │           ├── conf-f-psi-0.5
    │           └── conf-f-psi-1
    └── TrueFace_PostSocial
        ├── Facebook
        │   ├── Real
        │   │   └── FFHQ
        │   └── Fake
        │       ├── StyleGAN
        │       │   ├── images-psi-0.5
        │       │   └── images-psi-0.7
        │       └── StyleGAN2
        │           ├── conf-f-psi-0.5
        │           └── conf-f-psi-1
        ├── Telegram
        │   ├── Real
        │   │   └── FFHQ
        │   └── Fake
        │       ├── StyleGAN
        │       │   ├── images-psi-0.5
        │       │   └── images-psi-0.7
        │       └── StyleGAN2
        │           ├── conf-f-psi-0.5
        │           └── conf-f-psi-1
        ├── Twitter
        │   ├── Real
        │   │   └── FFHQ
        │   └── Fake
        │       ├── StyleGAN
        │       │   ├── images-psi-0.5
        │       │   └── images-psi-0.7
        │       └── StyleGAN2
        │           ├── conf-f-psi-0.5
        │           └── conf-f-psi-1
        └── Whatsapp
            ├── Real
            │   └── FFHQ
            └── Fake
                ├── StyleGAN
                │   ├── images-psi-0.5
                │   └── images-psi-0.7
                └── StyleGAN2
                    ├── conf-f-psi-0.5
                    └── conf-f-psi-1 -->

---------------------------------------------------------------------

### TrueFake Dataset
The **TrueFake** dataset includes a variety of real and fake faces generated for evaluation and testing purposes.

/media/NAS/
- TrueFake
  - Extension
    - FLUX.1  
      - animals
      - faces
      - general  
      - landscapes
    - StableDiffusion1.5
      - animals
      - animals_backup
      - faces
      - general
      - landscapes
    - StableDiffusion2
      - animals
      - animals_backup
      - faces
      - general
      - landscapes
    - StableDiffusion3
      - animals
      - animals_backup
      - faces
      - general
      - landscapes
    - StableDiffusionXL  
      - animals
      - animals_backup
      - faces
      - general
      - landscapes
    - StyleGAN3
      - conf-t-psi-0.5  
      - conf-t-psi-1


---------------------------------------------------------------------
### FORLAB Dataset
The **FORLAB** dataset contains only real images.

**NOTA** subfolders are not complete bc idk if it's useful

/media/NAS/
- FORLAB
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
