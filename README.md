# Cat diffusion

![Static Badge](https://img.shields.io/badge/Python-%237F52FF?style=for-the-badge&logo=Python&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorch-%23FE7B7B?style=for-the-badge&logo=PyTorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorchXLA-%234DA651?style=for-the-badge&logo=PyG&logoColor=white)
![Static Badge](https://img.shields.io/badge/CometML-%234A2296?style=for-the-badge&logo=bitcomet&logoColor=white)
![Static Badge](https://img.shields.io/badge/TPU-%2325A162?style=for-the-badge&logo=turbo&logoColor=white)
![Static Badge](https://img.shields.io/badge/Git-%23EA330E?style=for-the-badge&logo=git&logoColor=white)

The project is a diffusion model written in [PyTorch](https://pytorch.org/) that can generate random images of cats. The model was written from scratch. Training took place on a dataset containing 9,971 images of cats of various sizes, each of which was then scaled to the required resolution. 
All training was done on the [Kaggle](https://www.kaggle.com/) platform using the TPU accelerator. The training code is contained [here](https://www.kaggle.com/code/danildolgov/diffusion-model).

The trained model is also available on [Kaggle](https://www.kaggle.com/):
1. [Cat-diffusion | size 64x64 | epochs 5000](https://www.kaggle.com/models/danildolgov/cat-diffusion)
2. Cat-diffusion | size 128x128 | epochs ? (coming soon...)

# Running
To generate cat images using the cat-diffusion model, you need to download the [git project](https://github.com/KoTeuKaSeeker/diffusion-model/archive/refs/heads/master.zip) and run the <b>inference.ipynb</b> notebook.

# Preview


Here is a demonstration of what a dialogue with a model will look like and what responses it gives.
![model preview](https://github.com/KoTeuKaSeeker/DialogLLM/blob/main/assets/dialogllm_preview.gif)

Below are examples of dialogues with the model. All of them are the result of the model's work, you can verify this by downloading the project and asking the same questions. <br>

Dialogue 1 demonstrates that even such a simple model has some understanding of the world: it was able to give its own definition of the concept of "computer", which even makes logical sense.
| Dialog 1 |
| ----------------------------------------------------------------------------------------- | 
|[SPEAKER]: What is a computer? Tell me<br> |
|[BOT]:  A computer is a record-keeping system . It keeps all the data in a single place . |
|[SPEAKER]: How can I use it?<br> |
|[BOT]: You simply put all your documents in one place and then it copies them into memory .|
|[SPEAKER]: Can I borrow one from you for a while?<br> |
|[BOT]: Yes , you can .<br> |
|[SPEAKER]: Thank you, I really appreciate it! Do you have a college degree?<br> |
|[BOT]: Yes , a bachelor's degree and a master's .<br> |

<br>

Dialogue 2 shows that the model is able to navigate simple everyday situations.

| Dialog 2 |
| ----------------------------------------------------------------------------------------- | 
|[SPEAKER]: Do you like apples? |
|[BOT]:  Yes , they're very popular in my neighborhood .  |
|[SPEAKER]: How much does one apple cost now? |
|[BOT]:  Six cents . That's a lot of money .  |
|[SPEAKER]: Yes, that's a lot. Shall we go to the store for them? |
|[BOT]:  Okay . Look , apples have a higher sales price in my neighborhood .  |
|[SPEAKER]: Then we can go get them somewhere else. |
|[BOT]:  Where else ?  |
|[SPEAKER]: I think they are cheaper in the city center. |
|[BOT]:  I see . |

<br>

Dialogue 3 is the most interesting - it demonstrates the model's understanding of the world. It is clear that the model has some understanding of the height of objects, but it does not understand at all what speed is.

| Dialog 3 |
| ----------------------------------------------------------------------------------------- | 
|[SPEAKER]: What is taller - a tree or a man? |
|[BOT]:  A tree.He is taller than me .  |
|[SPEAKER]: Who is lower - a man or a tree? |
|[BOT]:  A man and a half .  |
|[SPEAKER]: This is also true! And who is faster - a person or a car? |
|[BOT]:  Someone who can move fast , right ?  |
|[SPEAKER]: Right! |
|[BOT]:  Are you crazy ?  |
|[SPEAKER]: No, why are you asking? |
|[BOT]:  As a matter of fact , this man is taller than me .  |
|[SPEAKER]: Maybe. Could you walk faster than a car? |
|[BOT]:  Maybe . But I prefer a car myself .  |
|[SPEAKER]: Usually people drive cars because they move faster than people. |
|[BOT]:  Yeah , that ’ s right . But I drive a car because I want to be able to move fast . |
