# Problem and Approach
The dataset we worked on is derived from the Google Landmark Recognition Challenge that took place on Kaggle. The challenge at hand was to build models that classify the images provided in such a way that it matches the correct landmark with each unique image.

We have to classify these landmarks from (15 Thousand!) different classes of landmarks.The landmark recognition training data originally contained over 1.2 million images with around 15K classes.To put things simply, this means that we would require a lot of computing power, coupled with a lot of time and patience. We worked on Nvidia DGX GPU (supercomputer) because of the same.

Another problem we faced is that we were given image URL's, so first we wrote the python script for downloading the images from those URL's and place them into their respective classes.

Now we seperated these folders into 3 parts train(80%), test(10%) and validation(10%) using python script 'folder_splitting.py'

