# synthetic_data

01 and 02 are used to generate synthetic HTML data and fine-tune a classification model based on a Medium example (detailed in 01). The main difference is that I used a local instance of Ollama with LLaMA 3.2 to mimic the process. It was very time-consuming, but there was no cost involved.

03 is used to generate synthetic data using both traditional statistical methods and LLM-based methods. I created an evaluation to compare the two approaches. My personal laptop is a MacBook M1 with 8GB of RAM, so the process was very slow. Generating just 50 samples took me two days.

To go further, I set up a virtual environment with Python 3.11 to match the required LLM packages. I also tried Python 3.13, but it was not very compatible.

