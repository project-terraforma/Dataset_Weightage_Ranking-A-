import os
import pandas as pd
import dspy
from pydantic import BaseModel # Standard Pydantic BaseModel
from typing import Literal

# Imports for local Mistral model
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# --- Configuration ---
# Local Mistral (Student Model) Configuration
MISTRAL_LOCAL_MODEL_PATH = r"mistral_models/7B-Instruct-v0.3" 

# OpenAI (Teacher Model) Configuration
# IMPORTANT: REPLACE WITH YOUR ACTUAL OPENAI API KEY IF DIFFERENT
OPENAI_API_KEY = "" # User-provided key
OPENAI_TEACHER_MODEL_NAME = "gpt-4o-mini" # Model for prompt_model in optimizer

# General Configuration
CSV_FILE_PATH = "cleaned.csv"
DEFAULT_LOCATION_CONTEXT = "San Francisco" 
OPTIMIZED_MATCHER_PATH = "drew_optimized_placematcher_mistral_student.json"

# --- 1. Load Local Mistral Model and Tokenizer (Student Model) ---
mistral_llm_model_obj = None 
mistral_llm_tokenizer_obj = None

try:
    print("🔧 Loading local Mistral model and tokenizer (Student Model)...")
    local_model_path_abs = os.path.abspath(MISTRAL_LOCAL_MODEL_PATH)
    tokenizer_file_path = os.path.join(local_model_path_abs, "tokenizer.model.v3")
    if not os.path.exists(tokenizer_file_path):
        tokenizer_file_path_alt = os.path.join(local_model_path_abs, "tokenizer.model")
        if os.path.exists(tokenizer_file_path_alt):
            tokenizer_file_path = tokenizer_file_path_alt
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file_path} or {tokenizer_file_path_alt}")

    mistral_llm_tokenizer_obj = MistralTokenizer.from_file(tokenizer_file_path)
    mistral_llm_model_obj = Transformer.from_folder(local_model_path_abs, device="cuda") # Assuming CUDA
    print("✅ Mistral Student Model and tokenizer loaded successfully (targeting CUDA).")
except Exception as e:
    print(f"❌ Failed to load local Mistral Student Model/tokenizer: {e}")
    exit(1)

# --- 2. Custom DSPy LM Client for Local Mistral (Student Model) ---
# --- 2. Custom DSPy LM Client for Local Mistral (Student Model) ---
class MistralLocalClient(dspy.LM):
    def __init__(self, llm_model, tokenizer, model_name_for_dspy=MISTRAL_LOCAL_MODEL_PATH, max_tokens=50, temperature=0.0, **kwargs):
        super().__init__(model=model_name_for_dspy) 
        self.actual_llm_model = llm_model 
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        # self.history = [] # Can be added if needed

    def basic_request(self, prompt, **kwargs):
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        
        final_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.kwargs, 
            **kwargs,     
        }

        out_tokens, _ = generate(
            [tokens], self.actual_llm_model, 
            max_tokens=final_kwargs["max_tokens"],
            temperature=final_kwargs["temperature"],
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        response_text = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0]).strip()
        return [response_text]

    def __call__(self, prompt=None, messages=None, only_completed=True, return_sorted=False, **kwargs):
        """
        Handle DSPy's calling convention. DSPy may pass either:
        1. A prompt string directly
        2. Messages in various formats
        3. Other formats depending on the DSPy version
        """
        
        # Handle different input formats that DSPy might use
        if prompt is not None:
            # Direct prompt string
            actual_prompt = prompt
        elif messages is not None:
            # Handle messages format
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict) and 'content' in messages[0]:
                    # List of message dicts
                    actual_prompt = messages[-1]['content']  # Use the last message
                elif isinstance(messages[0], str):
                    # List of strings
                    actual_prompt = messages[-1]
                else:
                    actual_prompt = str(messages[-1])
            else:
                actual_prompt = str(messages)
        else:
            # Fallback: look for other common parameter names
            if 'query' in kwargs:
                actual_prompt = kwargs['query']
            elif 'text' in kwargs:
                actual_prompt = kwargs['text']
            else:
                raise ValueError(f"No prompt provided. Got args: prompt={prompt}, messages={messages}, kwargs={kwargs}")
        
        # Ensure we have a string prompt
        if not isinstance(actual_prompt, str):
            actual_prompt = str(actual_prompt)
            
        try:
            response = self.basic_request(actual_prompt, **kwargs)
            return response
        except Exception as e:
            print(f"Error in MistralLocalClient.__call__: {e}")
            print(f"Prompt was: {actual_prompt[:100]}...")
            raise

    def generate(self, prompt, **kwargs):
        """Alternative method that some DSPy versions might call"""
        return self.__call__(prompt=prompt, **kwargs)

# --- Pydantic Model for Place Input ---
class Place(BaseModel):
    name: str
    address: str

# --- DSPy Signature for Place Matching ---
class PlaceMatcherSignature(dspy.Signature):
    """Determine if two points of interest refer to the same place.
Given two records representing places or businesses (place_one, place_two), each with a name and address,
and a general location_context, analyze the information and determine if they refer to the same real-world entity.
Consider minor differences such as case, diacritics, transliteration, abbreviations, or formatting.
If there are significant differences in either the name or address, even if one field matches exactly, they are likely not a match.
"""
    place_one: Place = dspy.InputField(desc="The first place with its name and address.")
    place_two: Place = dspy.InputField(desc="The second place with its name and address.")
    location_context: str = dspy.InputField(desc="The general location (e.g., city or region) for context.")
    match: bool = dspy.OutputField(desc="Output 'True' if they refer to the same place, 'False' otherwise.")
    match_confidence: Literal["low", "medium", "high"] = dspy.OutputField(desc="Your confidence in the match decision: 'low', 'medium', or 'high'.")

# --- Main DSPy Program Logic ---
def main():
    print(f"🚀 Starting DSPy Place Matching Program (OpenAI Teacher, Mistral Student) 🚀")

    # --- 1. Initialize LLMs ---
    # Student Model (Local Mistral)
    student_lm = MistralLocalClient(llm_model=mistral_llm_model_obj, tokenizer=mistral_llm_tokenizer_obj)
    print("✅ Mistral Student LM for DSPy configured.")

    # Teacher Model (OpenAI)
    if not OPENAI_API_KEY or "YOUR_OPENAI_API_KEY" in OPENAI_API_KEY or OPENAI_API_KEY == "":
        if OPENAI_API_KEY == "":
            print(f"ℹ️ Using OpenAI API key provided: {OPENAI_API_KEY[:10]}...")
        else:
            print("🔴 CRITICAL ERROR: OPENAI_API_KEY is a placeholder. Please set your actual key.")
            return
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 

    teacher_lm = None
    try:
        print(f"🔧 Configuring Teacher LM: openai/{OPENAI_TEACHER_MODEL_NAME}")
        teacher_lm = dspy.LM(model=f"openai/{OPENAI_TEACHER_MODEL_NAME}", api_key=OPENAI_API_KEY)
        print(f"✅ OpenAI Teacher LM (openai/{OPENAI_TEACHER_MODEL_NAME}) configured.")
    except Exception as e:
        print(f"❌ Failed to configure OpenAI Teacher LM: {e}")
        return

    # Configure DSPy settings: Default LM for execution will be the student (Mistral)
    dspy.settings.configure(lm=student_lm)
    print(f"✅ DSPy global LM configured to use Mistral Student LM for execution.")


    # --- 2. Define DSPy Module (Predictor) ---
    # This is the 'matcher' referred to in Drew's tp.compile(matcher, trainset=trainset)
    student_program_template = dspy.Predict(PlaceMatcherSignature) 
    print("✅ PlaceMatcher module template created.")

    # --- 3. Load and Prepare Data ---
    print(f"💾 Loading data from {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {CSV_FILE_PATH} not found.")
        return
        
    data_examples = []
    for _, row in df.iterrows():
        place1 = Place(name=str(row['osm_name']), address=str(row['osm_name']))
        place2 = Place(name=str(row['gers_name']), address=str(row['gers_name']))
        loc_context = str(row.get('location_context', DEFAULT_LOCATION_CONTEXT))
        gold_match_bool = None
        verified_label_val = row['verified_label']
        if isinstance(verified_label_val, bool): gold_match_bool = verified_label_val
        elif isinstance(verified_label_val, (int, float)): gold_match_bool = bool(int(verified_label_val))
        elif isinstance(verified_label_val, str): gold_match_bool = verified_label_val.strip().lower() in ['yes', 'true', '1']
        else:
            print(f"Warning: Unrecognized verified_label format: {verified_label_val} for osm_name: {row['osm_name']}. Defaulting to False.")
            gold_match_bool = False
        example = dspy.Example(
            place_one=place1,
            place_two=place2,
            location_context=loc_context,
            match=gold_match_bool 
        ).with_inputs("place_one", "place_two", "location_context")
        data_examples.append(example)
    
    if not data_examples: print("❌ No data loaded. Exiting."); return
    print(f"✅ Loaded {len(data_examples)} examples.")

    train_size = int(0.7 * len(data_examples)); 
    if len(data_examples) - train_size < 10 and len(data_examples) > 20 : # Ensure some dev examples if possible
        train_size = len(data_examples) - 10 
    if train_size <=0 and len(data_examples) > 0 : # ensure some trainset
        train_size = max(1, int(0.1 * len(data_examples))) # at least 1 or 10%
        
    trainset = data_examples[:train_size]; devset = data_examples[train_size:]
    
    if not trainset: print("❌ Training set is empty. Cannot proceed."); return
    if not devset: print("⚠️ Development set is empty. Evaluation will be skipped. MIPROv2 might use part of trainset for validation if evalset is not provided.")
    
    print(f"📊 Training set size: {len(trainset)}, Development set size: {len(devset)}")

    # --- 4. Define Validation Metric ---
    def validate_match(example, pred, trace=None):
        # Debugging prints to understand what 'pred' is
        # print(f"[METRIC DEBUG] Gold: {example.match}, Type of pred: {type(pred)}")
        # print(f"[METRIC DEBUG] Value of pred (first 100 chars): {str(pred)[:100]}")
        # print(f"[METRIC DEBUG] Attributes of pred: {dir(pred)}")

        if hasattr(pred, 'match'):
            return example.match == pred.match
        
        print(f"[METRIC WARNING] Prediction object missing 'match' attribute. Pred object: {pred}. Gold was: {example.match}")
        return False # Prediction failed or not structured as expected
    print("✅ Validation metric defined.")

    # --- 5. Optimize with MIPROv2 ---
    from dspy.teleprompt import MIPROv2 
    print("⚙️ Configuring MIPROv2 Teleprompter...")
    
    optimizer_config_mipro = dict(
        metric=validate_match,
        prompt_model=teacher_lm,  # OpenAI model
        task_model=student_lm,    # Local Mistral model
        auto="light"
    )
    
    compiled_program = None 
    try:
        # optimizer = MIPROv2(**optimizer_config_mipro) 
        
        # print("⚙️ Compiling program with MIPROv2 (Teacher: OpenAI, Student: Mistral)...")
        # # MODIFICATION: Following Drew's PDF strictly for tp.compile() arguments (matcher, trainset)
        # # And adding requires_permission_to_run=False to avoid hangs.
        # # Note: MIPROv2 might internally use a part of trainset for validation if evalset is not given.
        # # Some versions of MIPROv2 might accept evalset, but Drew's slide and user's error suggest not using it here.
        # compiled_program = optimizer.compile(
        #     student_program_template, # Pass the base module to compile
        #     trainset=trainset, 
        #     requires_permission_to_run=False 
        # )
        # print("✅ Program compiled successfully with MIPROv2.")
        # compiled_program.save(OPTIMIZED_MATCHER_PATH)
        # print(f"💾 Optimized matcher saved to {OPTIMIZED_MATCHER_PATH}")
        print("⚙️ Using BootstrapFewShot (much faster alternative to MIPROv2)...")
        from dspy.teleprompt import BootstrapFewShot

        # Much simpler and faster optimizer
        optimizer = BootstrapFewShot(
            metric=validate_match,
            max_bootstrapped_demos=3,  # Very small number
            max_labeled_demos=3,
            teacher_settings={'lm': teacher_lm}  # Use OpenAI for bootstrapping
        )

        try:
            compiled_program = optimizer.compile(
                student_program_template,
                trainset=trainset[:20]  # Use only 20 training examples
            )
            print("✅ Program compiled successfully with BootstrapFewShot.")
            compiled_program.save(OPTIMIZED_MATCHER_PATH)
            print(f"💾 Optimized matcher saved to {OPTIMIZED_MATCHER_PATH}")
        except Exception as e:
            print(f"❌ Error during compilation: {e}")
            return

    except ImportError: 
        print(f"❌ Error: Could not import MIPROv2.")
        return 
    except Exception as e: 
        print(f"❌ Error during MIPROv2 compilation: {e}")
        print("ℹ️ MIPROv2 compilation failed. You might want to try BootstrapFewShot if issues persist, or check your DSPy version / MIPROv2 parameters.")
        return # Exiting if MIPROv2 fails as per strict following of Drew's preferred optimizer


    # --- 6. Evaluate the Compiled Program (using the student_lm configured globally) ---
    if devset and compiled_program:
        print("📈 Evaluating compiled program on the devset (using Mistral Student LM)...")
        from dspy.evaluate.evaluate import Evaluate
        evaluator = Evaluate(devset=devset, metric=validate_match, num_threads=1, display_progress=True, display_table=5, provide_traceback=True)
        score = evaluator(compiled_program) # compiled_program will use dspy.settings.lm (Mistral)
        print(f"💯 Evaluation score on devset: {score}")
    elif not compiled_program:
        print("ℹ️ No compiled program to evaluate (compilation failed).")
    else: 
        print("ℹ️ No development set to evaluate on, but program was compiled.")

    # --- 7. Example of Loading and Using the Optimized Program ---
    if os.path.exists(OPTIMIZED_MATCHER_PATH) and compiled_program:
        print(f"\n🔍 Example: Loading and testing the optimized matcher from {OPTIMIZED_MATCHER_PATH}")
        
        loaded_matcher_module = dspy.Predict(PlaceMatcherSignature) 
        loaded_matcher_module.load(OPTIMIZED_MATCHER_PATH) 
        print("✅ Optimized matcher loaded (will use Mistral Student LM for inference).")
        
        if devset:
            test_example_input = devset[0].with_inputs("place_one", "place_two", "location_context")
            print(f"\nTest Example Inputs: place_one={test_example_input.place_one}, place_two={test_example_input.place_two}, context={test_example_input.location_context}")
            print(f"Gold label: match={devset[0].match}")
            try:
                # Ensure the loaded module uses the correct LM (Mistral, already set globally)
                prediction = loaded_matcher_module(
                    place_one=test_example_input.place_one, 
                    place_two=test_example_input.place_two, 
                    location_context=test_example_input.location_context
                )
                print(f"Prediction: match={prediction.match}, confidence={prediction.match_confidence}")
            except Exception as e:
                print(f"Error during test prediction with loaded matcher: {e}")
        else:
            print("No devset examples to test loaded matcher.")

    print(f"\n✨ DSPy Place Matching Program (Drew Breunig Inspired - OpenAI Teacher, Mistral Student) finished. ✨")

if __name__ == "__main__":
    main()
