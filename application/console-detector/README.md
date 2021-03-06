# Spotted Cats Detector, v0.1, console version

Console version of the Spotted Cat Detector tries to predict presence of a spotted cat on an image.

Typical usage looks like this:


    console-detector.py input_source output.csv

where


**input_source**	- either path to directory with an image set or text file including paths to images

**output.csv**		- where to save the results of prediction


### Example 1

    console-detector.py ./path/to/img/dir output.csv

Take all images from the directory _./path/to/img/dir_. For every image in the directory make a suggestion if there is a spotted cat.

Result, in form of real number from [0,1] put to text comma-separated file _output.csv_
		
Content of the 'output.csv' looks like path/to/image.jpg,probability
		
	d:\image-01.jpg,0.613826860006
	d:\image-02,0.502818665636
	d:\image-03,0.202661674923

### Example 2

    console-detector.py ./path/to/img/list.txt output.csv

Take all images indicated in _list.txt_ and make a sugestion if there is a spotted cat.

Result, in form of real number from [0,1] put to text comma-separated file _output.csv_
		
Content of list.txt looks like this
			
	d:\image-01.jpg
	d:\image-02.jpg
	d:\image-03.jpg
