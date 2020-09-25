# deep-learning

#### Face-Mask classification

Predict whether a person in an image is wearing a mask.
The Challenge can be found on [Zindi](https://zindi.africa/hackathons/spot-the-mask-challenge/) for more info.

Solution summary:

* I used two out-of-the-box(less/No tuning) approaches:
  * Built a 3-layer ConvNet with a Dense Classification layer and output layer model; making it a 5-layer model.

  * The 2nd approach was transfer learning using the MobileNet-V2 model for feature extraction with an added top layer for classification.

  * Approach 2 gave a better score but both have alot of improvement required but i had no time to do it.


* Library: TensorFlow.v.2.0
* evaluation metrics logloss.

**Private_LB Score:** 0.2858 **Rank:** 83/146
