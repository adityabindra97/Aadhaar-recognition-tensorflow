# Aadhaar-recognition-tensorflow
Using tensorflow to detect if a given image is that of an aadhaar card and if yes, cropping out the portion from the image that contains the aadhaar and saving it to a new file. 

Used mobilenet_v1_coco for training data for about 12,000 steps with 364 images.

After correctly installing tensorflow and dependencies, steps to follow:
1. Put the image you want to test in the test_images directory and rename it to "image3.jpg". (Use only jpg images)
2. Run detectionfinal.py, it'll do the rest. It'll give your output "cropped.jpg" (300dpi)
   in the output directory and also print "true" if it's an Aadhaar card and "False" if not.


NOTE: limited detection boxes to 1 and minimum detection threshold to 0.8 to give accurate results.
      Modify according to your needs.
