# synthetic_data

01 and 02 are used to generate synthetic HTML data and fine-tune a classification model based on a Medium example (detailed in 01). The main difference is that I used a local instance of Ollama with LLaMA 3.2 to mimic the process. It was very time-consuming, but there was no cost involved.

03 is used to generate synthetic data using both traditional statistical methods and LLM-based methods. I created an evaluation to compare the two approaches. My personal laptop is a MacBook M1 with 8GB of RAM, so the process was very slow. Generating just 50 samples took me two days.
--- Comparison Summary ---

Generation Speed & Reliability:
- Statistical: Relatively Fast. Reliability based on code logic (Actual/Target ~95.3%).
- LLM (llama3.2-vision:latest): Significantly Slower. Reliability depends on LLM adherence to prompt & parsing robustness (Actual Receipts/Target Receipts = 100.0%).

Fidelity (Statistical Similarity):
- Review the distribution plots and KS test results generated above.
- Review the item/category frequency lists and plots.
- LLM (llama3.2-vision:latest) output quality depends heavily on the model's capabilities and the prompt effectiveness.

Utility (Usefulness for Analysis):
- Statistical data utility analysis ran successfully (at least partially): True
- LLM data utility analysis ran successfully (at least partially): True
- Check if analyses yielded plausible results for both. LLM data might have more variance or unexpected patterns.
- Statistical method easily preserves customer weekly patterns. LLM method (as implemented) generates independent receipts, limiting longitudinal analysis.

Privacy (Basic Checks):
- Both methods avoided direct PII by design.
- Review outlier values and potential uniqueness risks noted above.
- Statistical method offers more control. LLM's output predictability is lower.

Overall Recommendation:
- **Statistical Method:** Still generally preferred for large-scale, consistent, structured data with known constraints and relationships (like weekly habits). Faster, reliable, controllable.
- **LLM Method (llama3.2-vision:latest):** Can generate diverse examples but faces challenges with speed, numerical/structural consistency, and maintaining relationships across records without advanced techniques. Parsing/validation is essential. Best suited for augmenting data or generating less structured/textual elements, or when creative variance is desired over strict consistency.

To go further, I set up a virtual environment with Python 3.11 to match the required LLM packages. I also tried Python 3.13, but it was not very compatible.

