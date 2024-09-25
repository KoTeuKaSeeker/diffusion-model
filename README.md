# Cat diffusion

![Static Badge](https://img.shields.io/badge/Python-%237F52FF?style=for-the-badge&logo=Python&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorch-%23FE7B7B?style=for-the-badge&logo=PyTorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorchXLA-%234DA651?style=for-the-badge&logo=PyG&logoColor=white)
![Static Badge](https://img.shields.io/badge/CometML-%234A2296?style=for-the-badge&logo=bitcomet&logoColor=white)
![Static Badge](https://img.shields.io/badge/TPU-%2325A162?style=for-the-badge&logo=turbo&logoColor=white)
![Static Badge](https://img.shields.io/badge/Git-%23EA330E?style=for-the-badge&logo=git&logoColor=white)

The project is a diffusion model written in [PyTorch](https://pytorch.org/) that can generate random images of cats. The model was written from scratch. Training took place on a [dataset](https://www.kaggle.com/datasets/danildolgov/cat-dataset/data) containing 9,971 images of cats of various sizes, each of which was then scaled to the required resolution. 
All training was done on the [Kaggle](https://www.kaggle.com/) platform using the TPU accelerator. The training code is contained [here](https://www.kaggle.com/code/danildolgov/diffusion-model).

The trained model is also available on [Kaggle](https://www.kaggle.com/):
1. [Cat-diffusion | size 64x64 | epochs 5000](https://www.kaggle.com/models/danildolgov/cat-diffusion)
2. Cat-diffusion | size 128x128 | epochs ? (coming soon...)

# Running
To generate cat images using the cat-diffusion model, you need to download the [git project](https://github.com/KoTeuKaSeeker/diffusion-model/archive/refs/heads/master.zip) and run the <b>inference.ipynb</b> notebook. If the model that should be used for generation is not specified, the [cat-diffusion64x64ep5000](https://www.kaggle.com/models/danildolgov/cat-diffusion) model will be downloaded automatically.

# Preview
Results of the [cat-diffusion64x64ep5000](https://www.kaggle.com/models/danildolgov/cat-diffusion):<br>

<table>
  <tr>
    <td>
      <div align="center">
        <h3>Image 1</h3>
        <img src="https://github.com/KoTeuKaSeeker/diffusion-model/blob/master/assets/images/readme/1.png" alt="cat-diffusion64x64ep5000" width="500"/>
      </div>
    </td>
    <td>
      <div align="center">
        <h3>Image 2</h3>
        <img src="https://github.com/KoTeuKaSeeker/diffusion-model/blob/master/assets/images/readme/2.png" alt="cat-diffusion64x64ep5000" width="500"/>
      </div>
    </td>
  </tr>
  <tr>
    <td>
      <div align="center">
        <h3>Image 3</h3>
        <img src="https://github.com/KoTeuKaSeeker/diffusion-model/blob/master/assets/images/readme/3.png" alt="cat-diffusion64x64ep5000" width="500"/>
      </div>
    </td>
    <td>
      <div align="center">
        <h3>Image 4</h3>
        <img src="https://github.com/KoTeuKaSeeker/diffusion-model/blob/master/assets/images/readme/4.png" alt="cat-diffusion64x64ep5000" width="500"/>
      </div>
    </td>
  </tr>
</table>

You can see that some of the images are repeated - this behavior is demonstrated by the model itself, which is related to how diffuse models are trained.
