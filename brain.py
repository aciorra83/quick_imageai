# https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Classification/README.md


from imageai.Classification import ImageClassification
import os
execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(
    execution_path, "/home/alex/Desktop/ZTM/smart_brain/resnet50_imagenet_tf.2.0.h5"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(
    os.path.join(execution_path, "godzilla.jpg"), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
