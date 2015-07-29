#Todo List

## May 29, 2015
do:
+change this to org
+change skin detection to a class, provide interface to face detection

read:
+viola jones (better trained xml?)
+svm in pyopencv

## June 2, 2015
course:
+[Cornell](http://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lectures.html)

## June 22, 2015
Current face_detect use skin_detect to judge skin area sufficiency, while main function will do face_detect and skin_detect in a row, consider remove one of them.

## July 12, 2015
Try:
1. erosion/dilation to remove hole in detected faces
2. finding convex hull of it

Modify softmax or sigmoid (or log?) to produce a 0-1 barrier for 0<Alpha<1.

## Jul 28, 2015
1. Make (hist)gaussian smoother self adaptive, migrate from face/sky to util
2. Sky-prob-map's not robust to the wide range of sky pixels, especially dark ones.

## Jul 29, 2015
1. Try Gaussian kernel density estimation for bimodal detection.