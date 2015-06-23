# awesome-random-forest
Random Forest - a curated list of resources regarding random forest

## Table of Contents

 - [Theory](#theory)
   - [Lectures](#lectures)
   - [Books](#books)
   - [Papers] (#papers)
 - [Applications] (#applications)
   - [Image Classification] (#image-classification)
   - [Object Detection] (#object-detection)
   - [Object Tracking] (#object-tracking)
   - [Edge Detection] (#edge-detection)
   - [Semantic Segmentation] (#semantic-segmentation)
   - [Human / Hand Pose Estimation] (#human--hand-pose-estimation)
   - [Camera Relocalization] (#camera-relocalization)
   - [Low-Level Vision] (#low-level-vision)
 - [Codes] (#codes)
    
   
## Theory
### Lectures
* [ICCV 2013 Tutorial : Decision Forests and Fields for Computer Vision] (http://research.microsoft.com/en-us/um/cambridge/projects/iccv2013tutorial/) by Jamie Shotton and Sebastian Nowozin
  * [Lecture 1] (http://techtalks.tv/talks/randomized-decision-forests-and-their-applications-in-computer-vision-jamie/59432/) : Randomized Decision Forests and their Applications in Computer Vision I (Decision Forest, Classification Forest, 
  * [Lecture 2] (http://techtalks.tv/talks/decision-jungles-jamie-second-half-of-above/59434/) : Randomized Decision Forests and their Applications in Computer Vision II (Regression Forest, Decision Jungle)
  * [Lecture 3] (http://techtalks.tv/talks/entropy-estimation-and-streaming-data-sebastian/59433/) : Entropy estimation and streaming data
  * [Lecture 4] (http://techtalks.tv/talks/decision-and-regression-tree-fields-sebastian/59435/) : Decision and Regression Tree Fields
* [UBC Machine Learning] (http://www.cs.ubc.ca/~nando/540-2013/lectures.html) by Nando de Freitas
  * [Lecture 8 slide] (http://www.cs.ubc.ca/~nando/540-2013/lectures/l8.pdf) , [Lecture 8 video] (https://www.youtube.com/watch?v=-dCtJjlEEgM&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6&index=11) : Decision trees
  * [Lecture 9 slide] (http://www.cs.ubc.ca/~nando/540-2013/lectures/l9.pdf) , [Lecture 9 video] (https://www.youtube.com/watch?v=3kYujfDgmNk&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6&index=12) : Random forests
  * [Lecture 10 video] (https://www.youtube.com/watch?v=zFGPjRPwyFw&index=13&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6) : Random forest applications
  
### Books
* Antonio Criminisi, Jamie Shotton (2013)
  * [Decision Forests for Computer Vision and Medical Image Analysis] (http://link.springer.com/book/10.1007%2F978-1-4471-4929-3)
* Trevor Hastie, Robert Tibshirani, Jerome Friedman (2008)
  * [The Elements of Statistical Learning, (Chapter 10, 15, and 16)] (http://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf)
  
### Papers
* Global Refinement of Random Forest [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ren_Global_Refinement_of_2015_CVPR_paper.pdf)]
  * Shaoqing Ren, Xudong Cao, Yichen Wei, Jian Sun, Global Refinement of Random Forest, CVPR 2015
* Feature-Budgeted Random Forest [[Paper] (http://jmlr.org/proceedings/papers/v37/nan15.pdf)] [[Supp](http://jmlr.org/proceedings/papers/v37/nan15-supp.pdf)]
  * Feng Nan, Joseph Wang, Venkatesh Saligrama, Feature-Budgeted Random Forest, ICML 2015 
* Bayesian and Empirical Bayesian Forests [[Paper] (http://jmlr.org/proceedings/papers/v37/matthew15.pdf)]
  * Taddy Matthew, Chun-Sheng Chen, Jun Yu, Mitch Wyle, Bayesian and Empirical Bayesian Forests, ICML 2015
* Narrowing the Gap: Random Forests In Theory and In Practice [[Paper] (http://jmlr.org/proceedings/papers/v32/denil14.pdf)]
  * Misha Denil, David Matheson, Nando de Freitas, Narrowing the Gap: Random Forests In Theory and In Practice, ICML 2014
* Semi-supervised Node Splitting for Random Forest Construction [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Liu_Semi-supervised_Node_Splitting_2013_CVPR_paper.pdf)]
  * Xiao Liu, Mingli Song, Dacheng Tao, Zicheng Liu, Luming Zhang, Chun Chen and Jiajun Bu, Semi-supervised Node Splitting for Random Forest Construction, CVPR 2013
* Improved Information Gain Estimates for Decision Tree Induction [[Paper] (http://www.nowozin.net/sebastian/papers/nowozin2012infogain.pdf)]
  * Sebastian Nowozin, Improved Information Gain Estimates for Decision Tree Induction, ICML 2012
* MIForests: Multiple-Instance Learning with Randomized Trees [[Paper] (http://lrs.icg.tugraz.at/pubs/leistner_eccv_10.pdf)] [[Code] (http://www.ymer.org/amir/software/milforests/)]
  * Christian Leistner, Amir Saffari, and Horst Bischof, MIForests: Multiple-Instance Learning with Randomized Trees, ECCV 2010


 
## Applications

### Image classification
* ETH Zurich [[Paper-CVPR15] (http://www.iai.uni-bonn.de/~gall/download/jgall_coarse2fine_cvpr15.pdf)]
			 [[Paper-CVPR14] (http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Ristin_Incremental_Learning_of_2014_CVPR_paper.pdf)]
			 [[Paper-ECCV] (http://www.vision.ee.ethz.ch/~lbossard/bossard_eccv14_food-101.pdf)]
  * Marko Ristin, Juergen Gall, Matthieu Guillaumin, and Luc Van Gool, From Categories to Subcategories: Large-scale Image Classification with Partial Class Label Refinement, CVPR 2015
  * Marko Ristin, Matthieu Guillaumin, Juergen Gall, and Luc Van Gool, Incremental Learning of NCM Forests for Large-Scale Image Classification, CVPR 2014
  * Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool, Food-101 – Mining Discriminative Components with Random Forests, ECCV 2014
* University of Girona & University of Oxford [[Paper] (http://www.cs.huji.ac.il/~daphna/course/CoursePapers/bosch07a.pdf)]
  * Anna Bosch, Andrew Zisserman, and Xavier Munoz, Image Classification using Random Forests and Ferns, ICCV 2007

### Object Detection
* Graz University of Technology [[Paper-CVPR] (http://lrs.icg.tugraz.at/pubs/schulter_cvpr_14.pdf)] [[Paper-ICCV] (http://lrs.icg.tugraz.at/pubs/schulter_iccv_13.pdf)]
  * Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, and Horst Bischof, Accurate Object Detection with Joint Classification-Regression Random Forests, CVPR 2014
  * Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, and Horst Bischof, Alternating Regression Forests for Object Detection and Pose Estimation, ICCV 2013
* ETH Zurich + MSRC [[Paper] (http://www.iai.uni-bonn.de/~gall/download/jgall_houghforest_cvpr09.pdf) ]
  * Juergen Gall, and Victor Lempitsky, Class-Specific Hough Forests for Object Detection, CVPR 2009

### Object Tracking
* Technische Universitat Munchen [[Paper] (http://campar.in.tum.de/pub/tanda2014cvpr/tanda2014cvpr.pdf)]
  * David Joseph Tan, and Slobodan Ilic, Multi-Forest Tracker: A Chameleon in Tracking, CVPR 2014
* ETH Zurich + Leibniz University Hannover + Stanford University [[Paper] (http://www.igp.ethz.ch/photogrammetry/publications/pdf_folder/LeaFenKuzRosSavCVPR14.pdf)]
  * Laura Leal-Taixe, Michele Fenzi, Alina Kuznetsova, Bodo Rosenhahn, and Silvio Savarese, Learning an image-based motion context for multiple people tracking, CVPR 2014
* Graz University of Technology [[Paper] (https://lrs.icg.tugraz.at/pubs/godec_iccv_11.pdf)]
  * Martin Godec, Peter M. Roth, and Horst Bischof, Hough-based Tracking of Non-Rigid Objects, ICCV 2011

### Edge Detection
### Semantic Segmentation
### Human / Hand Pose Estimation
### Camera Relocalization
### Low-Level vision

## Codes
* Matlab
  * [Piotr Dollar's toolbox] (http://vision.ucsd.edu/~pdollar/toolbox/doc/)
  * [Andrej Karpathy's toolbox] (https://github.com/karpathy/Random-Forest-Matlab)
* R
  * [Breiman and Cutler's random forests] (http://cran.r-project.org/web/packages/randomForest/)
* C/C++
  * [Sherwood library] (http://research.microsoft.com/en-us/downloads/52d5b9c3-a638-42a1-94a5-d549e2251728/)
  * [Regression tree package by Pierre Geurts] (http://www.montefiore.ulg.ac.be/~geurts/Software.html)
* Python
  * [Scikit-learn] (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
* JavaScript
  * [Forestjs] (https://github.com/karpathy/forestjs)

 
Maintainers - [Jiwon Kim](http://github.com/kjw0612), [Janghoon Choi](http://github.com/JanghoonChoi), [Jung Kwon Lee](http://github.com/deruci)
