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
    <td><img src="https://github.com/nps6-uwf/CNN-for-Oral-Lesion-Classification/blob/main/project%20figures/squamouscellcarcinoma_normalized_33_1.png?raw=true"></img></td>
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
</ol>
