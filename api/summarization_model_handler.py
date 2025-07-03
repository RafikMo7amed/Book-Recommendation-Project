import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging

from . import config
import os
os.environ['HUGGING_FACE_HUB_CACHE'] = '/app/cache'

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
        try:
            device = 0 if torch.cuda.is_available() else -1
            model_name = "google/pegasus-large"
            cache_path = "/app/model_cache"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_path)
            
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=model,
                tokenizer=self.tokenizer,
                device=device
            )
            logger.info(f"Summarizer ({model_name}) loaded successfully on {'GPU' if device != -1 else 'CPU'}.")
        except Exception as e:
            logger.error(f"CRITICAL: Could not load summarization model: {e}.", exc_info=True)

    def summarize_text(self, text, params, ratio=0.5):
        if not self.summarizer_pipeline:
            return "Error: Summarizer not initialized."

        max_model_length = 1024
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_model_length)
        
        num_input_tokens = len(inputs['input_ids'][0])
        target_token_count = int(num_input_tokens * ratio)
        
        min_len = int(target_token_count * 0.7)
        max_len = int(target_token_count * 1.3)
        if min_len < 30: min_len = 30
        if max_len < 40: max_len = 40

        try:
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