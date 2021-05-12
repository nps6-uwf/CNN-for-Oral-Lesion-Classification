# CNN-for-Oral-Lesion-Classification
In this paper I introduce a new oral lesion database that I created by scraping Images from research papers and dentists online.  Next, I showcase a novel method for generating NxN pixel samples from human lips by isolating the labial fissure via image segmentation induced via the Kmean algorithm.  I experiment with a variety of convolutional neural network architectures including: a simple network, AlexNet, VGG, and ResNet in an effort to correctly classify oral tissue samples of: tongue, lip, HSV-1, and squamous cell carcinoma.  First I consider the dataset as a whole and attempt multi class classification, then I carry out multiple binary classification experiments including: lesion versus non-lesion, herpes versus squamous cell carcinoma, and tongue versus lip.  Finally I implement a novel method that attempts to perform the same multi class classification that I started with.  This method involves an ensemble of binary ResNet networks.  I conclude with an overview of lots of future research that can be done on this problem.

<img src="https://github.com/nps6-uwf/CNN-for-Oral-Lesion-Classification/blob/main/project%20figures/ALL_AUG_samples.PNG?raw=true"></img>

# Can you accurately classify oral lesions?
<table>
  <tbody>
    <tr>
    <td><strong>Herpes</strong></td>
    <td><strong>Squamous Cell Carcinoma</strong></td>
    <td><strong>???</strong></td>
    </tr>
    <tr>
    <td><img src="https://github.com/nps6-uwf/CNN-for-Oral-Lesion-Classification/blob/main/project%20figures/herpessimplex_101_2.png?raw=true"></img></td>
    <td style="align:center;"><img style="text-align:center;" src="https://github.com/nps6-uwf/CNN-for-Oral-Lesion-Classification/blob/main/project%20figures/squamouscellcarcinoma_normalized_33_1.png?raw=true"></img></td>
    <td><img src="https://github.com/nps6-uwf/CNN-for-Oral-Lesion-Classification/blob/main/project%20figures/herpessimplex_77_3.png?raw=true"></img></td>
    </tr>
  </tbody>
  </table>

# A novel classifcation model:
<table>
  <tbody>
  <tr>
    <td><img src="https://github.com/nps6-uwf/CNN-for-Oral-Lesion-Classification/blob/main/project%20figures/binary_ensemble_resnet.PNG?raw=true"></img></td>
    <td><img src="https://github.com/nps6-uwf/CNN-for-Oral-Lesion-Classification/blob/main/project%20figures/resNetArchitecture.PNG?raw=true"></img></td>
  </tr>
  </tbody>
</table>

# Interesting Ideas for Future research
<ol>
  <li>Apply tranfer learning via a neaural network trained on a large high quality skin lesion database such as HAM10000.</li>
  <li>Use state of the art object detection algorithms sucha as yolo and faster-R-cnn to automate lesion data collection process from online journals.</li>
  <li>Compare classifcation accuracy between whole images of the mouth (containing an oral lesion) and the image broken up into arbitrarily small fragments.</li>
  <li>Use generative algorithms to generate lesions images, use cutom algorithm to affix the newly created lesion image onto an image of a mouth.</li>
  <li>Expand the oral lesion database, add new types of lesions, design complex ensemble structures to achieve high classifcation accuracy.</li>
</ol>

# Link to Paper:
<a href="https://docs.google.com/document/d/1D-lKCRJTVxpOz_hiG894c6aR0FUbq7wTq9mUn5uE02o/edit?usp=sharing">Paper</a>
