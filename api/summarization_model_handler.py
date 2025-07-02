import torch
from transformers import pipeline, AutoTokenizer
import logging

from . import config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummarizationModelHandler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SummarizationModelHandler, cls).__new__(cls)
            cls._instance.summarizer_pipeline = None
            cls._instance.tokenizer = None
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """Initializing the summarization model and its tokenizer."""
        try:
            device = 0 if torch.cuda.is_available() else -1
            model_name = "google/pegasus-large"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=model_name,
                tokenizer=self.tokenizer,
                device=device
            )
            logger.info(f"Summarizer ({model_name}) loaded successfully on {'GPU' if device != -1 else 'CPU'}.")
        except Exception as e:
            logger.error(f"CRITICAL: Could not load summarization model: {e}.", exc_info=True)

    def summarize_text(self, text, params, ratio=0.5):
        """
        Summarizing text with specific parameters, with guaranteed truncation to prevent CUDA errors.
        """
        if not self.summarizer_pipeline:
            return "Error: Summarizer not initialized."

        # 1. Explicitly truncate the text to the model's max length (1024 tokens)
        max_model_length = 1024
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_model_length)
        
        # 2. Setting dynamic summary length based on the (potentially truncated) input
        num_input_tokens = len(inputs['input_ids'][0])
        target_token_count = int(num_input_tokens * ratio)
        
        # Setting reasonable min/max lengths for the output
        min_len = int(target_token_count * 0.7)
        max_len = int(target_token_count * 1.3)
        if min_len < 30: min_len = 30
        if max_len < 40: max_len = 40

        try:
            # 3. Passing the safe, truncated tokens to the model's generate function
            summary_ids = self.summarizer_pipeline.model.generate(
                inputs['input_ids'].to(self.summarizer_pipeline.device),
                min_length=min_len, 
                max_length=max_len, 
                **params
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Error during summarization with params {params}: {e}")
            return "Error: Could not generate summary."