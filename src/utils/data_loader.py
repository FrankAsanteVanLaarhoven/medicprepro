import pandas as pd
import kaggle
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.setup_kaggle()
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_loader.log'),
                logging.StreamHandler()
            ]
        )

    def setup_kaggle(self):
        """Configure Kaggle credentials"""
        try:
            load_dotenv()
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_dir.mkdir(exist_ok=True)
            
            credentials = {
                'username': os.getenv('KAGGLE_USERNAME'),
                'key': os.getenv('KAGGLE_KEY')
            }
            
            with open(kaggle_dir / 'kaggle.json', 'w') as f:
                json.dump(credentials, f)
            
            os.chmod(kaggle_dir / 'kaggle.json', 0o600)
            
        except Exception as e:
            self.logger.error(f"Kaggle setup failed: {str(e)}")
            raise

    def load_data(self, force_reload=False):
        """Load data with caching"""
        data_file = self.data_dir / 'Medicine_Details.csv'
        
        if not data_file.exists() or force_reload:
            try:
                self.logger.info("Downloading dataset from Kaggle...")
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(
                    'singhnavjot2062001/11000-medicine-details',
                    path=str(self.data_dir),
                    unzip=True
                )
                self.logger.info("Dataset downloaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to download dataset: {str(e)}")
                raise

        try:
            df = pd.read_csv(data_file)
            self.logger.info(f"Loaded {len(df)} records")
            return self.validate_data(df)
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def validate_data(self, df):
        """Validate data schema and quality"""
        required_columns = [
            'medicine_name', 'composition', 'side_effects',
            'excellent_review_%', 'average_review_%', 'poor_review_%'
        ]
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Invalid data schema")
            
        # Validate review percentages
        for col in ['excellent_review_%', 'average_review_%', 'poor_review_%']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].clip(0, 100)
            
        return df
