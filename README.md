# Awesome Random Forest

Random Forest - a curated list of resources regarding tree-based methods and more, including but not limited to random forest, bagging and boosting.

## Contributing
Please feel free to [pull requests](https://github.com/kjw0612/awesome-random-forest/pulls), email Jung Kwon Lee (deruci@snu.ac.kr) or join our chats to add links.

[![Join the chat at https://gitter.im/kjw0612/awesome-random-forest](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/kjw0612/awesome-random-forest?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

![randomforest](https://31.media.tumblr.com/79670eabe93cdd448c15f5bcb198d0fb/tumblr_inline_n8e398YbKv1s04rc3.png)

## Table of Contents

 - [Codes] (#codes)
 - [Theory](#theory)
   - [Lectures](#lectures)
   - [Books](#books)
   - [Papers] (#papers)
     - [Analysis / Understanding] (#analysis--understanding)
     - [Model variants] (#model-variants)
   - [Thesis] (#thesis)
 - [Applications] (#applications)
   - [Image Classification] (#image-classification)
   - [Object Detection] (#object-detection)
   - [Object Tracking] (#object-tracking)
   - [Edge Detection] (#edge-detection)
   - [Semantic Segmentation] (#semantic-segmentation)
   - [Human / Hand Pose Estimation] (#human--hand-pose-estimation)
   - [3D Localization] (#3d-localization)
   - [Low-Level Vision] (#low-level-vision)
   - [Facial Expression Recognition] (#facial-expression-recognition)
   - [Interpretability, regularization, compression pruning and feature selection](#Interpretability, regularization, compression pruning and feature selection)
   

## Codes
* Matlab
  * [Piotr Dollar's toolbox] (http://vision.ucsd.edu/~pdollar/toolbox/doc/)
  * [Andrej Karpathy's toolbox] (https://github.com/karpathy/Random-Forest-Matlab)
  * [M5PrimeLab by Gints Jekabsons] (http://www.cs.rtu.lv/jekabsons/regression.html)
* R
  * [Breiman and Cutler's random forests] (http://cran.r-project.org/web/packages/randomForest/)
  * [Hothorn et al.'s party package with `cforest` function](http://cran.r-project.org/web/packages/party/)
* C/C++
  * [Sherwood library] (http://research.microsoft.com/en-us/downloads/52d5b9c3-a638-42a1-94a5-d549e2251728/)
  * [Regression tree package by Pierre Geurts] (http://www.montefiore.ulg.ac.be/~geurts/Software.html)
* Python
  * [Scikit-learn] (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
* JavaScript
  * [Forestjs] (https://github.com/karpathy/forestjs)
* Go (golang)
  * [CloudForest] (https://github.com/ryanbressler/CloudForest)
   
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
* Luc Devroye, Laszlo Gyorfi, Gabor Lugosi (1996) 
  * [A Probabilistic Theory of Pattern Recognition (Chapter 20, 21)](http://www.szit.bme.hu/~gyorfi/pbook.pdf)
  
### Papers
#### Analysis / Understanding
* Consistency of random forests [[Paper]](http://www.normalesup.org/~scornet/paper/article.pdf) 
 * Scornet, E., Biau, G. and Vert, J.-P. (2015). Consistency of random forests, The Annals of Statistics, in press. 
* On the asymptotics of random forests [[Paper]](http://arxiv.org/abs/1409.2090)
 * Scornet, E. (2015). On the asymptotics of random forests, Journal of Multivariate Analysis, in press.
* Random Forests In Theory and In Practice [[Paper] (http://jmlr.org/proceedings/papers/v32/denil14.pdf)]
  * Misha Denil, David Matheson, Nando de Freitas, Narrowing the Gap: Random Forests In Theory and In Practice, ICML 2014
* Explaining the Success of AdaBoost and Random Forests as Interpolating Classifiers Abraham J. Wyner, Matthew Olson, Justin Bleich, David Mease [[Paper](https://arxiv.org/abs/1504.07676)]

  
#### Model variants
* Deep Neural Decision Forests [[Paper](http://research.microsoft.com/pubs/255952/ICCV15_DeepNDF_main.pdf)]
  * Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, and Samuel Rota Bulo, Deep Neural Decision Forests, ICCV 2015
* Canonical Correlation Forests [[Paper](http://arxiv.org/pdf/1507.05444.pdf)]
  * Tom Rainforth, and Frank Wood, Canonical Correlation Forests, arxiv 2015
* Relating Cascaded Random Forests to Deep Convolutional Neural Networks [[Paper] (http://arxiv.org/pdf/1507.07583.pdf)]
  * David L Richmond, Dagmar Kainmueller, Michael Y Yang, Eugene W Myers, and Carsten Rother, Relating Cascaded Random Forests to Deep Convolutional Neural Networks for Semantic Segmentation, arxiv 2015
* Bayesian Forests [[Paper] (http://jmlr.org/proceedings/papers/v37/matthew15.pdf)]
  * Taddy Matthew, Chun-Sheng Chen, Jun Yu, Mitch Wyle, Bayesian and Empirical Bayesian Forests, ICML 2015
* Mondrian Forests: Efficient Online Random Forests [[Paper]](http://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf) [[Code]](http://www.gatsby.ucl.ac.uk/~balaji/mondrianforest/) [[Slides]](http://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_slides.pdf)
  * Balaji Lakshminarayanan, Daniel M. Roy and Yee Whye Teh, Mondrian Forests: Efficient Online Random Forests, NIPS 2014
* Extremely randomized trees P Geurts, D Ernst, L Wehenkel - Machine learning, 2006 [[Paper](http://orbi.ulg.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)] [[Code] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)]
* Decision Jungles [[Paper] (http://research.microsoft.com/pubs/205439/DecisionJunglesNIPS2013.pdf)]
  * Jamie Shotton, Toby Sharp, Pushmeet Kohli, Sebastian Nowozin, John Winn, and Antonio Criminisi, Decision Jungles: Compact and Rich Models for Classification, NIPS 2013
  * Laptev, Dmitry, and Joachim M. Buhmann. Transformation-invariant convolutional jungles. CVPR 2015. [[Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Laptev_Transformation-Invariant_Convolutional_Jungles_2015_CVPR_paper.pdf)]
* Semi-supervised Node Splitting for Random Forest Construction [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Liu_Semi-supervised_Node_Splitting_2013_CVPR_paper.pdf)]
  * Xiao Liu, Mingli Song, Dacheng Tao, Zicheng Liu, Luming Zhang, Chun Chen and Jiajun Bu, Semi-supervised Node Splitting for Random Forest Construction, CVPR 2013
* Improved Information Gain Estimates for Decision Tree Induction [[Paper] (http://www.nowozin.net/sebastian/papers/nowozin2012infogain.pdf)]
  * Sebastian Nowozin, Improved Information Gain Estimates for Decision Tree Induction, ICML 2012
* MIForests: Multiple-Instance Learning with Randomized Trees [[Paper] (http://lrs.icg.tugraz.at/pubs/leistner_eccv_10.pdf)] [[Code] (http://www.ymer.org/amir/software/milforests/)]
  * Christian Leistner, Amir Saffari, and Horst Bischof, MIForests: Multiple-Instance Learning with Randomized Trees, ECCV 2010
* Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof: Alternating Decision Forests. CVPR 2013 [Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Schulter_Alternating_Decision_Forests_2013_CVPR_paper.pdf)
* Decision Forests, Convolutional Networks and the Models in-Between [[Paper](https://arxiv.org/abs/1603.01250)]
* Random Uniform Forests Saïp Ciss [[Paper](https://hal.archives-ouvertes.fr/hal-01104340/)] [[Code R](https://cran.r-project.org/web/packages/randomUniformForest/index.html)]
* Autoencoder Trees, Ozan İrsoy, Ethem Alpaydın 2015 [[Paper](http://www.jmlr.org/proceedings/papers/v45/Irsoy15.pdf)] 
 

## Thesis
* Understanding Random Forests
 * PhD dissertation, Gilles Louppe, July 2014. Defended on October 9, 2014. 
 * [[Repository]](https://github.com/glouppe/phd-thesis) with thesis and related codes

 
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
* University of California, Irvine [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hallman_Oriented_Edge_Forests_2015_CVPR_paper.pdf)] [[Code] (https://github.com/samhallman/oef)]
  * Sam Hallman, and Charless C. Fowlkes, Oriented Edge Forests for Boundary Detection, CVPR 2015
* Microsoft Research [[Paper] (http://research-srv.microsoft.com/pubs/202540/DollarICCV13edges.pdf)] [[Code] (https://github.com/pdollar/edges)]
  * Piotr Dollar, and C. Lawrence Zitnick, Structured Forests for Fast Edge Detection, ICCV 2013
* Massachusetts Inst. of Technology + Microsoft Research [[Paper] (http://research.microsoft.com/en-us/um/people/larryz/cvpr13sketchtokens.pdf)] [[Code] (https://github.com/joelimlimit/SketchTokens)]
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
  * Technicolor R&I Hannover [[Paper](https://technicolor-my.sharepoint.com/personal/jordi_salvador_technicolor_com/_layouts/15/guestaccess.aspx?guestaccesstoken=2z88Le9arMQ7tcGGYApHmdM9Pet2AqqoxMBDcu6eRbc%3d&docid=0e7f0b9ed1d0f4497829ae6b2b0deeec3)]
    * Jordi Salvador, and Eduardo Pérez-Pellitero, Naive Bayes Super-Resolution Forest, ICCV 2015
  * Graz University of Technology [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schulter_Fast_and_Accurate_2015_CVPR_paper.pdf)]
    * Samuel Schulter, Christian Leistner, and Horst Bischof, Fast and Accurate Image Upscaling with Super-Resolution Forests, CVPR 2015
* Denoising 
  * Microsoft Research + iCub Facility - Istituto Italiano di Tecnologia [[Paper] (http://research.microsoft.com/pubs/217099/CVPR2014ForestFiltering.pdf)]
    * Sean Ryan Fanello, Cem Keskin, Pushmeet Kohli, Shahram Izadi, Jamie Shotton, Antonio Criminisi, Ugo Pattacini, and Tim Paek, Filter Forests for Learning Data-Dependent Convolutional Kernels, CVPR 2014

### Facial expression recognition
* Sorbonne Universites [[Paper](http://www.isir.upmc.fr/files/2015ACTI3549.pdf)]
  * Arnaud Dapogny, Kevin Bailly, and Severine Dubuisson, Pairwise Conditional Random Forests for Facial Expression Recognition, ICCV 2015
  
### Interpretability, regularization, compression pruning and feature selection
* Global Refinement of Random Forest [[Paper] (http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ren_Global_Refinement_of_2015_CVPR_paper.pdf)]
  * Shaoqing Ren, Xudong Cao, Yichen Wei, Jian Sun, Global Refinement of Random Forest, CVPR 2015
* L1-based compression of random forest models Arnaud Joly, Fran¸cois Schnitzler, Pierre Geurts and Louis Wehenkel ESANN 2012 [[Paper](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2012-43.pdf)]
* Feature-Budgeted Random Forest [[Paper] (http://jmlr.org/proceedings/papers/v37/nan15.pdf)] [[Supp](http://jmlr.org/proceedings/papers/v37/nan15-supp.pdf)]
  * Feng Nan, Joseph Wang, Venkatesh Saligrama, Feature-Budgeted Random Forest, ICML 2015 
  * Pruning Random Forests for Prediction on a Budget Feng Nan, Joseph Wang, Venkatesh Saligrama NIPS 2016 [[Paper](https://papers.nips.cc/paper/6250-pruning-random-forests-for-prediction-on-a-budget.pdf)]
* Meinshausen, Nicolai. "Node harvest." The Annals of Applied Statistics 4.4 (2010): 2049-2072. [[Paper](http://projecteuclid.org/download/pdfview_1/euclid.aoas/1294167809)] [[Code R](https://cran.r-project.org/web/packages/nodeHarvest/index.html)] [[Code Python](https://github.com/mbillingr/NodeHarvest)]
* Making Tree Ensembles Interpretable: A Bayesian Model Selection Approach S. Hara, K. Hayashi, [[Paper](https://arxiv.org/abs/1606.09066)] [[Code](https://github.com/sato9hara/defragTrees)]
* Cui, Zhicheng, et al. "Optimal action extraction for random forests and boosted trees." ACM SIGKDD 2015. [[Paper](http://www.cse.wustl.edu/~ychen/public/OAE.pdf)]
* DART: Dropouts meet Multiple Additive Regression Trees K. V. Rashmi, Ran Gilad-Bachrach [[Paper](http://www.jmlr.org/proceedings/papers/v38/korlakaivinayak15.pdf)]
* Begon, Jean-Michel, Arnaud Joly, and Pierre Geurts. Joint learning and pruning of decision forests. (2016). [[Paper](http://orbi.ulg.ac.be/bitstream/2268/202344/1/Begon_jlpdf_abstract.pdf)]	



Maintainers - [Jiwon Kim](http://github.com/kjw0612), [Jung Kwon Lee](http://github.com/deruci)
