import os
import argparse
import logging
import jsonlines

from tqdm import tqdm
from datasets import load_dataset
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.run_context import RunContext
from pipeline import run_pipeline


class Args:
    """
    Configuration class to handle command-line arguments and system settings.
    """
    def parseargs(self):
        parser = argparse.ArgumentParser(description="M1-Env Agentic Hospital Simulation")
        
        # --- Data Configuration ---
        parser.add_argument('--data_repo_path', type=str, default="SII-SPIRAL-MED/DiagnosisArena",
                            help="Path or HuggingFace ID for the dataset.")
        parser.add_argument("--parallel_workers", type=int, default=16,
                            help="Number of concurrent threads for processing.")
        
        # --- Patient Model Configuration ---
        parser.add_argument('--patient_model_name', type=str, default="gpt-oss-120b",
                            help="Name of the model simulating the patient.")
        parser.add_argument("--patient_base_url", type=str, default="YOUR_URL",
                            help="Base URL for the patient model API.")
        parser.add_argument("--patient_api_key", type=str, default="YOUR_KEY",
                            help="API key for the patient model.")

        # --- Doctor Model Configuration ---
        parser.add_argument('--doctor_responses_api', action='store_true', 
                            help="Flag to use the responses API interface. Required for certain models (e.g., GPT-5) to utilize tools correctly.")
        parser.add_argument('--doctor_model_name', type=str, default="gpt-5",
                            help="Name of the model simulating the doctor.")
        parser.add_argument("--doctor_base_url", type=str, default="YOUR_URL",
                            help="Base URL for the doctor model API.")
        parser.add_argument("--doctor_api_key", type=str, default="YOUR_KEY",
                            help="API key for the doctor model.")

        # --- Patient Care Constraints ---
        parser.add_argument('--max_encounter_nums', type=int, default=8, 
                            help="Maximum number of department transfers (encounters) allowed for a patient.")
        parser.add_argument('--max_talk_turns', type=int, default=5, 
                            help="Maximum conversation turns per consultation. If the limit is exceeded by a specific margin (e.g., 3), the session is forcibly paused.")
        parser.add_argument('--max_query_nums', type=int, default=3, 
                            help="Maximum number of queries allowed per dialogue interaction.")
        
        self.pargs = parser.parse_args()
        
        # Map parsed arguments to class attributes
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

    def __init__(self) -> None:
        self.parseargs()
        # Define the output file path based on model names
        os.makedirs("./outputs", exist_ok=True)
        self.output_path = f"./outputs/P_{self.patient_model_name}-D_{self.doctor_model_name}-results.jsonl"

# Initialize global arguments
args = Args()  
    

# --- Logging Setup ---
# Configure logging to write to a file with timestamps
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
    filename=f"./logs/mimic_process_{args.doctor_model_name}.log",
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
    
def run(data_item: dict, args: Args) -> None:
    """
    Process a single data item through the simulation pipeline.
    
    Args:
        data_item (dict): The dataset entry containing patient/case info.
        args (Args): The configuration object.
    """
    # Initialize the context for this specific run
    context = RunContext(data_item=data_item, output_path=args.output_path)
    
    # Execute the main interaction pipeline
    final_context = run_pipeline(args, context)
    
    # If the process failed, exit early (note: 'exit' here has no effect as a function call, likely intended as a return or pass)
    if not context.process_success:
        exit
    
    # Save the results of the agent interaction
    final_context.output_agent_results()
    
    return
    
    
if __name__ == "__main__":
    
    # Load the test split of the dataset
    data=load_dataset(args.data_repo_path, split="test")
    data = data.select(range(1))
    print(f"Total data size: {len(data)}")

    # --- Resume Logic ---
    # Check existing output file to skip already processed IDs
    processed_ids = []
    if os.path.exists(args.output_path):
        with jsonlines.open(args.output_path, "r") as reader:
            processed_ids = [obj["original_data"]["id"] for obj in reader]
    
    # Filter dataset to include only unprocessed items
    rest_data = [obj for obj in data if obj["id"] not in processed_ids]
    print(f"Remaining items to process: {len(rest_data)}")

    # Prepare the partial function with fixed arguments
    run_data = partial(run, args=args)
    
    # --- Concurrent Execution ---
    # Use ThreadPoolExecutor to process data in parallel
    with ThreadPoolExecutor(max_workers = args.parallel_workers) as executor:
        futures = [executor.submit(run_data, d) for d in rest_data]

        # Iterate through completed futures with a progress bar
        for future in tqdm(as_completed(futures), total=len(rest_data)):
            try:
                future.result() 
            except Exception as e:
                # Log or print errors without stopping the entire batch
                print(f"Error processing data item: {e}")
            
        # list(tqdm(executor.map(run_data, rest_data), total=len(rest_data)))
    