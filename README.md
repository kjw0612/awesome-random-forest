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
   - [3D Localization] (#3d-localization)
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
* Bayesian Forests [[Paper] (http://jmlr.org/proceedings/papers/v37/matthew15.pdf)]
  * Taddy Matthew, Chun-Sheng Chen, Jun Yu, Mitch Wyle, Bayesian and Empirical Bayesian Forests, ICML 2015
* Mondrian Forests: Efficient Online Random Forests [[Paper]](http://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf) [[Code]](http://www.gatsby.ucl.ac.uk/~balaji/mondrianforest/) [[Slides]](http://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_slides.pdf)
  * Balaji Lakshminarayanan, Daniel M. Roy and Yee Whye Teh, Mondrian Forests: Efficient Online Random Forests, NIPS 2014
* Random Forests In Theory and In Practice [[Paper] (http://jmlr.org/proceedings/papers/v32/denil14.pdf)]
  * Misha Denil, David Matheson, Nando de Freitas, Narrowing the Gap: Random Forests In Theory and In Practice, ICML 2014
* Decision Jungles [[Paper] (http://research.microsoft.com/pubs/205439/DecisionJunglesNIPS2013.pdf)]
  * Jamie Shotton, Toby Sharp, Pushmeet Kohli, Sebastian Nowozin, John Winn, and Antonio Criminisi, Decision Jungles: Compact and Rich Models for Classification, NIPS 2013
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
* ETH Zurich + Microsoft Research Cambridge [[Paper] (http://www.iai.uni-bonn.de/~gall/download/jgall_houghforest_cvpr09.pdf)]
  * Juergen Gall, and Victor Lempitsky, Class-Specific Hough Forests for Object Detection, CVPR 2009

### Object Tracking
* Technische Universitat Munchen [[Paper] (http://campar.in.tum.de/pub/tanda2014cvpr/tanda2014cvpr.pdf)]
  * David Joseph Tan, and Slobodan Ilic, Multi-Forest Tracker: A Chameleon in Tracking, CVPR 2014
* ETH Zurich + Leibniz University Hannover + Stanford University [[Paper] (http://www.igp.ethz.ch/photogrammetry/publications/pdf_folder/LeaFenKuzRosSavCVPR14.pdf)]
  * Laura Leal-Taixe, Michele Fenzi, Alina Kuznetsova, Bodo Rosenhahn, and Silvio Savarese, Learning an image-based motion context for multiple people tracking, CVPR 2014
* Graz University of Technology [[Paper] (https://lrs.icg.tugraz.at/pubs/godec_iccv_11.pdf)]
  * Martin Godec, Peter M. Roth, and Horst Bischof, Hough-based Tracking of Non-Rigid Objects, ICCV 2011

### Edge Detection
* University of California, Irvine [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hallman_Oriented_Edge_Forests_2015_CVPR_paper.pdf)]
  * Sam Hallman, and Charless C. Fowlkes, Oriented Edge Forests for Boundary Detection, CVPR 2015
* Microsoft Research [[Paper] (http://research-srv.microsoft.com/pubs/202540/DollarICCV13edges.pdf)]
  * Piotr Dollar, and C. Lawrence Zitnick, Structured Forests for Fast Edge Detection, ICCV 2013
* Massachusetts Inst. of Technology + Microsoft Research [[Paper] (http://research.microsoft.com/en-us/um/people/larryz/cvpr13sketchtokens.pdf)]
  * Joseph J. Lim, C. Lawrence Zitnick, and Piotr Dollar, Sketch Tokens: A Learned Mid-level Representation for Contour and Object Detection, CVPR 2013

### Semantic Segmentation
* Fondazione Bruno Kessler, Microsoft Research Cambridge [[Paper] (http://www.dsi.unive.it/~srotabul/files/publications/CVPR2014a.pdf)]
  * Samuel Rota Bulo, and Peter Kontschieder, Neural Decision Forests for Semantic Image Labelling, CVPR 2014
* INRIA +  Microsoft Research Cambridge [[Paper] (http://step.polymtl.ca/~rv101/MICCAI-Laplacian-Forest.pdf)]
  * Herve Lombaert, Darko Zikic, Antonio Criminisi, and Nicholas Ayache, Laplacian Forests:Semantic Image Segmentation by Guided Bagging, MICCAI 2014
* Microsoft Research Cambridge +  GE Global Research Center + University of California +  Rutgers Univeristy [[Paper] (http://research.microsoft.com/pubs/146430/criminisi_ipmi_2011c.pdf)]
  * Albert Montillo1, Jamie Shotton, John Winn, Juan Eugenio Iglesias, Dimitri Metaxas, and Antonio Criminisi, Entangled Decision Forests and their Application for Semantic Segmentation of CT Images, IPMI 2011
* University of Cambridge + Toshiba Corporate R&D Center [[Paper] (http://mi.eng.cam.ac.uk/~cipolla/publications/inproceedings/2008-CVPR-semantic-texton-forests.pdf)]
  * Jamie Shotton, Matthew Johnson, and Roberto Cipolla, Semantic Texton Forests for Image Categorization and Segmentation, CVPR 2008
  
### Human / Hand Pose Estimation
* Microsoft Research Cambridge [[Paper-CHI] (http://research.microsoft.com/pubs/238453/pn362-sharp.pdf)][[Video-CHI] (http://research.microsoft.com/pubs/238453/pn362-sharp-video.mp4)]
                               [[Paper-CVPR] (http://research.microsoft.com/pubs/162510/vm.pdf)]
  * Toby Sharp, Cem Keskin, Duncan Robertson, Jonathan Taylor, Jamie Shotton, David Kim, Christoph Rhemann, Ido Leichter, Alon Vinnikov, Yichen Wei, Daniel Freedman, Pushmeet Kohli, Eyal Krupka, Andrew Fitzgibbon, and Shahram Izadi, Accurate, Robust, and Flexible Real-time Hand Tracking, CHI 2015
  * Jonathan Taylor, Jamie Shotton, Toby Sharp, and Andrew Fitzgibbon, The Vitruvian Manifold:Inferring Dense Correspondences for One-Shot Human Pose Estimation, CVPR 2012
* Microsoft Research Haifa [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Krupka_Discriminative_Ferns_Ensemble_2014_CVPR_paper.pdf)]
  * Eyal Krupka, Alon Vinnikov, Ben Klein, Aharon Bar Hillel, and Daniel Freedman, Discriminative Ferns Ensemble for Hand Pose Recognition, CVPR 2014
* Microsoft Research Asia [[Paper] (http://research.microsoft.com/en-us/people/yichenw/cvpr14_facealignment.pdf)]
  * Shaoqing Ren, Xudong Cao, Yichen Wei, and Jian Sun, Face Alignment at 3000 FPS via Regressing Local Binary Features, CVPR 2014
* Imperial College London [[Paper-CVPR-Face] (http://www.iis.ee.ic.ac.uk/icvl/doc/cvpr14_xiaowei.pdf)]
                          [[Paper-CVPR-Hand] (http://www.iis.ee.ic.ac.uk/icvl/doc/cvpr14_danny.pdf)]
						  [[Paper-ICCV] (http://www.iis.ee.ic.ac.uk/icvl/doc/ICCV13_danny.pdf)]
  * Xiaowei Zhao, Tae-Kyun Kim, and Wenhan Luo, Unified Face Analysis by Iterative Multi-Output Random Forests, CVPR 2014
  * Danhang Tang, Hyung Jin Chang, Alykhan Tejani, and Tae-Kyun Kim, Latent Regression Forest: Structured Estimation of 3D Articulated Hand Posture, CVPR 2014
  * Danhang Tang, Tsz-Ho Yu, and Tae-Kyun Kim, Real-time Articulated Hand Pose Estimation using Semi-supervised Transductive Regression Forests, ICCV 2013
* ETH Zurich + Microsoft [[Paper] (https://lirias.kuleuven.be/bitstream/123456789/398648/2/3601_open+access.pdf)]
  * Matthias Dantone, Juergen Gall, Christian Leistner, and Luc Van Gool, Human Pose Estimation using Body Parts Dependent Joint Regressors, CVPR 2013
  
### 3D localization 
* Imperial College London [[Paper] (http://www.iis.ee.ic.ac.uk/icvl/doc/ECCV2014_aly.pdf)]
  * Alykhan Tejani, Danhang Tang, Rigas Kouskouridas, and Tae-Kyun Kim, Latent-Class Hough Forests for 3D Object Detection and Pose Estimation, ECCV 2014
* Microsoft Research Cambridge + University of Illinois + Imperial College London [[Paper] (http://abnerguzman.com/publications/gkgssfi_cvpr14.pdf)]
  * Abner Guzman-Rivera, Pushmeet Kohli, Ben Glocker, Jamie Shotton, Toby Sharp, Andrew Fitzgibbon, and Shahram Izadi, Multi-Output Learning for Camera Relocalization, CVPR 2014
* Microsoft Research Cambridge [[Paper] (http://research.microsoft.com/pubs/184826/relocforests.pdf)]
  * Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon, Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images, CVPR 2013

### Low-Level vision
* Super-Resolution
  * Graz University of Technology [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schulter_Fast_and_Accurate_2015_CVPR_paper.pdf)]
    * Samuel Schulter, Christian Leistner, and Horst Bischof, Fast and Accurate Image Upscaling with Super-Resolution Forests, CVPR 2015
* Denoising 
  * Microsoft Research + iCub Facility - Istituto Italiano di Tecnologia [[Paper] (http://research.microsoft.com/pubs/217099/CVPR2014ForestFiltering.pdf)]
    * Sean Ryan Fanello, Cem Keskin, Pushmeet Kohli, Shahram Izadi, Jamie Shotton, Antonio Criminisi, Ugo Pattacini, and Tim Paek, Filter Forests for Learning Data-Dependent Convolutional Kernels, CVPR 2014
	
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
