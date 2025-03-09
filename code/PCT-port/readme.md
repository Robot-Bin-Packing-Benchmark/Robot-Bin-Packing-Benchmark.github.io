Robot Bin Packing Benchmark: Sample Script

This repository provides a sample Python script for the Robot Bin Packing Benchmark. The script is written in Python 3.7 and follows the specified requirements for submission. The necessary training data and test examples can be obtained from the GitHub.io page.

Requirements
	•	Python 3.7
	•	UTF-8 Encoding

Data
	•	Training and test datasets can be obtained from the Robot Bin Packing Benchmark GitHub.io site.
	•	The data and models should be configured based on the specific dataset type and container size, which can be adjusted in the base.yaml configuration file.

Script Structure

The sample consists of two main files:
	•	main.py
	•	pack.py

main.py Overview
	1.	get_args():
	•	This function defines the basic configuration for the algorithm:
	•	args: General configuration parameters.
	•	setting: Selected conditions for the experiment (e.g., training settings).
	•	data: Chosen test data.
	•	method: Name of the current algorithm.
	•	config_learning_method: Configuration of the current algorithm, including paths to the model files.
	•	The args_base imports the base.yaml configuration file, where the paths for the three types of data (training, validation, and test) are set, along with the container size.
	2.	load_data():
	•	This function is responsible for loading the data. The function’s input should be properly configured to ensure correct data loading.
	3.	pack():
	•	This function processes the loaded data (sizes) and computes the resulting actions (actions) and planning times (planning_times).
	•	The results are saved in the files:
	•	action.json: Contains the generated placement actions.
	•	planning_time.json: Contains the planning times for each action.

pack.py Overview

This file demonstrates the implementation of the packing algorithm, using the PCT algorithm as an example. You should modify this code according to your algorithm’s logic.
	1.	get_policy():
	•	This function loads the trained policy model. You need to load the model corresponding to the dataset you’re using (e.g., Ideal or Physics settings).
	2.	env = PackingDiscrete():
	•	This line imports the packing environment that will be used for bin packing operations.
	3.	obs = env.reset():
	•	This resets the environment and provides the initial observation (obs). This is the starting point for the packing algorithm.
	4.	pack_box():
	•	This function is where the actual packing process begins. You need to modify this section to implement your algorithm.
	•	The final result of the function should be a list of actions that follow the format:
	•	(rotation_flag, lx, ly), where rotation_flag represents the rotation of the box, and lx and ly are the placement coordinates.
	•	planning_time should be the time taken by your algorithm to complete the packing process. You should calculate the time from the moment the observation (obs) is input into the network until the action is generated.

Submitting Your Script
	1.	Input and Output Format:
	•	Your script must read the packed box sequence as input and output the appropriate placement actions. Specifically:
	•	Input: A sequence of packed boxes.
	•	Output: A tuple (rot, lx, ly) indicating the placement action for each box, and planning_time which is the time taken to generate the action.
	2.	Sample File Structure:
	•	Your project directory should look like this:

├── main.py
├── pack.py
├── base.yaml
├── action.json
├── planning_time.json


	3.	Model and Algorithm:
	•	Modify the pack_box() function to integrate your algorithm’s logic. Make sure the output format matches the required structure. The model loading and environment setup (env = PackingDiscrete()) should remain the same unless you are introducing custom modifications.
	4.	Training Your Model:
	•	For the submitted models, ensure you train them using the provided training datasets. You should generate models that can be loaded and used for evaluation by your script.

Additional Resources

For further information on the dataset and benchmark rules, please visit the official Robot Bin Packing Benchmark GitHub.

⸻

This revised version uses formal and precise language while explaining the script’s functionality in a professional manner. Let me know if you’d like to make any additional adjustments!
