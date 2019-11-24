## Scrubber Watch - Smoke detection


### Installation

1. Download the dataset from kaggle and unpack it https://www.kaggle.com/arpitjain007/game-of-deep-learning-ship-datasets/. 

You should now have the following folder structure:
```bash
---smoke_detection/train/images/*.png
```

2. copy `train_smoke.csv` to `train` folder

```bash
cp train_smoke.csv train
```

3. train the model recognition model

```bash
python smoke_classification.py
```

4. run inference on input image

```bash
python predict_smoke.py image.jpg
```


### Dataset


* We detect broken or missing scrubbers by looking at images of ships.
* This allows to track scrubber usage simply from a surveillance camera.
* To analyze the images, we rely on state-of-the-art computer vision. 
* However, smoke detection and quantification is a very niche topic, which is why there is no dataset readily available
* To solve this problem, we created an annotation tool and manually annotated a publicly available dataset of ships based on emitted smoke.
* The dataset we use is from the [game-of-deep-learning-ship-datasets](https://www.kaggle.com/arpitjain007/game-of-deep-learning-ship-datasets) kaggle challenge.
* This dataset comprises 6000 images of ships with their type annotations in 5 categories : Cargo, Military, Carrier, Cruise, Tankers. 
* We built a simple annotation tool that suits our problem. 
* Using our annotation tool, we annotated 2000 images from the kaggle challenge dataset and annotated if the smoke output is a noticeable or not. 
* We additionally scraped images of ships with significant smoke output and added it to the dataset.
* We will make our annotations publicly available soon.

* TODO: insert figure

### Recognition model

* We train a simple object recognition model based on our manually annotated data.
* To do so, we use the [winning kernel](https://www.kaggle.com/sandeeppat/ship-classification-top-3-5-kernel) from the same kaggle challenge and adapt it to our data.





