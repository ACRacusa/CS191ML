



There is one(1) ICA presented here: 
	- Performing Blind source separation in two audio files (in codes folder)
	
      1. ICA for the two audio files
	- There are 3 python scripts here:
		*model.py
			- This python script performs the Independent Component Analysis by using the FastICA. First it combines the two separate audio files by using the zip() function which returns an iterative tuple of the mixed file, and then performs FastICA to separate the independent components which are then saved as a WAV file.
		*plot-model.py
			-This python script shows the visualization of the two audio source files

		*plot-results.py
			-This python script shows the visualization of the two Independent Components that are produced from the model.py
	

Output files are described in Blind_Source_Separation_using_Independent_Component_Analysis__ICA_.pdf
