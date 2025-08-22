import spacy
import os 
import sys
import pathlib
from ast import literal_eval
import pandas as pd
from nltk.tokenize import sent_tokenize
from utils import load_subtitles_dataset


class named_entity_recognizer:
    def __init__(self, model_name="en_core_web_md"):
        """
        Initialize the recognizer with a spaCy model.
        Default = en_core_web_md (make sure it's installed).
        """
        self.model_name = model_name
        self.nlp = spacy.load(model_name)  # Load once, reuse everywhere

        # Character normalization mapping
        self.character_mapping = {
            # Walter White variations
            'walt': 'Walter White',
            'walter': 'Walter White', 
            'white': 'Walter White',
            'walt white': 'Walter White',
            'walter white': 'Walter White',
            'mr white': 'Walter White',
            'heisenberg': 'Walter White',

            # Jesse Pinkman variations  
            'jesse': 'Jesse Pinkman',
            'pinkman': 'Jesse Pinkman',
            'jesse pinkman': 'Jesse Pinkman',

            # Skyler White variations
            'skyler': 'Skyler White',
            'sky': 'Skyler White',
            'skyler white': 'Skyler White',

            # Hank Schrader variations
            'hank': 'Hank Schrader',
            'schrader': 'Hank Schrader',
            'hank schrader': 'Hank Schrader',

            # Marie Schrader variations
            'marie': 'Marie Schrader',
            'marie schrader': 'Marie Schrader',

            # Walter Jr variations
            'walter jr': 'Walter Jr',
            'walt jr': 'Walter Jr',
            'flynn': 'Walter Jr',

            # Saul Goodman variations
            'saul': 'Saul Goodman',
            'goodman': 'Saul Goodman',
            'saul goodman': 'Saul Goodman',
            'jimmy': 'Saul Goodman',
            'jimmy mcgill': 'Saul Goodman',

            # Gus Fring variations
            'gus': 'Gus Fring',
            'fring': 'Gus Fring',
            'gus fring': 'Gus Fring',
            'gustavo': 'Gus Fring',

            # Mike Ehrmantraut variations
            'mike': 'Mike Ehrmantraut',
            'ehrmantraut': 'Mike Ehrmantraut',
            'mike ehrmantraut': 'Mike Ehrmantraut',
        }

    def get_ners_inference(self, script):
        """
        Extract character names from a script.
        Returns a list of sets, one set per sentence.
        """
        script_sentences = sent_tokenize(script)
        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text.strip().lower()

                    # Normalize using mapping
                    if full_name in self.character_mapping:
                        mapped_name = self.character_mapping[full_name]
                    else:
                        # fallback on first token
                        first_name = full_name.split(" ")[0]
                        mapped_name = self.character_mapping.get(first_name, entity.text.strip())
                    
                    ners.add(mapped_name)
            ner_output.append(ners)

        return ner_output

    def get_ners(self, dataset_path, save_path=None):
        """
        Run NER over dataset.
        If save_path exists, reload cached results.
        """
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # load dataset 
        df = load_subtitles_dataset(dataset_path)
        

        # Run Inference
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path, index=False)
        
        return df
