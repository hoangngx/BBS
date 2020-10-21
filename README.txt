

----------------------------------This is one of Sun Xiangyu's academic code - "BBS_SXY"-----------------------------------------------------------




======   1. Description    ======------------------------------------------------------------------------------------------------------------------

	*  This program is inspired by an article named "Best-Buddies Similarity for Robust Template Matching£¬CVPR2015".
	After reading this passage, I started to realize this method described in it by using OPENCV and c++. And here is the source code I wrote.
	
	*  "BBS" is short for "Best-Buddies Similarity", which is a useful, robust, and parameter-free similarity measure between two sets of points.
	
	*  BBS is based on counting the number of Best-Buddies Pairs (BBPs)-pairs of points in source and target sets, 
	where each point is the nearest neighbor of the other. 
	
	*  BBS has several key features that make it robust against complex geometric deformations and high levels of outliers, 
	such as those arising from background clutter and occlusions. 
	
	*  And this program I wrote can deal with certain input image with a given rectangle, and mark a rectangle of the highest BBS in another image, 
	which is similar to the previous one. For example, a particular frame of a video and its next frame.
	
	*  And the output of this source code on the challenging real-world dataset is amazingly precise, far beyond my previous expectation.
	
	
======   2. Directory Structure    ======----------------------------------------------------------------------------------------------------------

	*  BBS.cpp 			-- The main source code
		
	*  OUTPUT_IMG		-- Include some previous output images of this program
	
	*  OUTPUT_CSV  		-- Include some previous output .csv files of this program
	
	
======   3. Guide    ======------------------------------------------------------------------------------------------------------------------------

	*  configure the directory of input images & .txt files(to mark the rectangle of the source image) in source code.
	
	*  set the number of images to be calculated.
	
	*  configure the directory of output iamges and .csv files in source code.
	
	*  run the source code, and wait for the output.
	

======   4. Others    ======----------------------------------------------------------------------------------------------------------------------- 

	*  This source code was finished in May, 2017
	   
	*  Want more information? see: https://csardasxy.github.io/homepage


---------------------------------------------------------------------------------------------------------------------------------------------------