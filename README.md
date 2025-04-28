Vocal Bridge - Real-Time Multilingual Video Dubbing
Group 3 – Mark Fahim, Haifa Alnajjar, Tyler Glaze
Vocal Bridge is a Natural Language Processing (NLP) project designed to break down language barriers by enabling real-time video dubbing and subtitling. This system transforms a video clip of a person speaking in English into a dubbed video in a target language (Turkish, German, Italian, Spanish, or Hungarian), complete with synchronized audio and translated subtitles. By leveraging advanced machine translation, automated transcription, and text-to-speech technologies, Vocal Bridge facilitates rapid, scalable, and accessible multilingual content creation for applications in media localization, education, and global communication.
Project Overview
The Vocal Bridge pipeline integrates:

Audio Transcription: Extracts spoken content from the input video using tools like openai-whisper.
Machine Translation: Translates text from English to the target language using sequence-to-sequence (Seq2Seq) LSTM models (for Turkish) and transformer-based models (for Spanish, German, Italian, Hungarian).
Text-to-Speech (TTS): Generates natural-sounding dubbed audio in the target language using edge_tts.
Subtitle Generation: Produces synchronized subtitles in both source and target languages.
Audio-Video Synchronization: Aligns dubbed audio and subtitles with the original video, ensuring accurate timing and lip-sync.

The system is trained on datasets from Kaggle (English to Spanish, Turkish, Italian, German) and Tatoeba (English to Hungarian), achieving BLEU scores ranging from 37.23% (Hungarian) to 56.66% (Italian), indicating good to high-quality translations.
Prerequisites
To run the Vocal Bridge code, ensure the following are installed:

Python 3.x
Libraries:
numpy, pandas (data handling)
tensorflow, keras (Seq2Seq model)
transformers (Hugging Face for MarianMT models)
openai-whisper (transcription)
edge_tts (text-to-speech)
sklearn, nltk (data preprocessing and evaluation)
matplotlib (visualization)
pickle (saving tokenizers/models)


Hardware:
Access to a high-performance computing (HPC) system like ROAR Collab (recommended for training).
GPU support (optional but significantly reduces training time, e.g., for Hungarian models on M1 MacBook Pro with TensorFlow-metal).


Datasets:
Download from:
English-to-German, Spanish, Turkish, Italian
Turkish-to-English
English-to-Hungarian





Instructions to Run the Code
Step 1: Train the Machine Translation Models
The project uses two types of models: Seq2Seq (for Turkish) and transformers (for Spanish, German, Italian, Hungarian). Training is resource-intensive and requires scheduling on an HPC system like ROAR Collab.

Seq2Seq Model (English to Turkish):

Navigate to the seq2seq directory.
Run the SLURM job script:sbatch submit_job.slurm


This executes seq2seq_MT.py, training the model for ~25 hours on a ROAR Collab node.


Transformer Models (English to Spanish, German, Italian):

Navigate to the transformers directory.
Run the SLURM job script:sbatch submit_job-en-de-ita-spa.slurm


This executes transformer_en_de_ita_spa.py, training models for each language (~10 hours per language on a ROAR Collab node).


Transformer Model (English to Hungarian):

Run the Jupyter notebook:jupyter notebook Final-TranslationModel-Hung-Port.ipynb


This trains the Hungarian model (~1.9 hours for 50% dataset, ~1.84 hours for full dataset on an M1 MacBook Pro with GPU).



Note: Training only needs to be performed once. Save the trained models and tokenizers for reuse.
Step 2: Generate Dubbed Video
After training, use the main notebook to process a video and generate the dubbed output.

Open main_notebook.ipynb in Jupyter Notebook.
Follow the notebook instructions to:
Upload an English video.
Select the target language (Spanish, Hungarian, Italian, Turkish, German).
Generate transcribed text, translated text, dubbed audio, and subtitles.
Output a dubbed video with synchronized audio and subtitles.



Note: Ensure all dependencies (e.g., openai-whisper, edge_tts) are installed and datasets are accessible.
Directory Structure
VocalBridge/
├── seq2seq/
│   ├── submit_job.slurm
│   ├── seq2seq_MT.py
├── transformers/
│   ├── submit_job-en-de-ita-spa.slurm
│   ├── transformer_en_de_ita_spa.py
│   ├── Final-TranslationModel-Hung-Port.ipynb
├── main_notebook.ipynb
├── README.md

Expected Output

Input: A video clip of a person speaking in English.
Output: A dubbed video in the target language (Turkish, German, Italian, Spanish, or Hungarian) with:
Synchronized dubbed audio (synthetic voice via edge_tts).
Translated subtitles in the target language.
Preserved semantic content and natural-sounding translations.



Performance

Seq2Seq (Turkish): Trained for 25 hours, achieves ~91% validation accuracy, suitable for understandable translations.
Transformers:
Spanish: BLEU 53.36% (high quality).
German: BLEU 42.91% (good quality).
Italian: BLEU 56.66% (high quality).
Hungarian (50% dataset): BLEU 37.23% (understandable).
Hungarian (full dataset): BLEU 43.22% (good quality).


Challenges:
Overfitting observed in transformer models (validation loss increases after early epochs).
Long training times (mitigated with GPU support for Hungarian).
Synthetic voice naturalness could be improved.



Future Improvements

Optimize Beam Search: Reduce computational cost for faster real-time translation.
Expand Datasets: Collect more sentence pairs to boost translation quality (especially for Hungarian and German).
Enhance Voice Synthesis: Integrate voice cloning to preserve the original speaker’s voice characteristics.
Improve BLEU Metric: Explore alternative evaluation metrics to better reflect translation quality.
Real-Time Processing: Optimize pipelines for live video dubbing applications.

References

Kaggle Datasets: English-to-German, Spanish, Turkish, Italian, Turkish-to-English
Tatoeba: English-to-Hungarian
ArtSmart.ai. "Natural Language Processing (NLP) Statistics 2024." Link
IEEE. "Document 9084044." Link
ACM. "DOI: 10.1145/3641343.3641425." Link

Contact
For questions or contributions, contact:

Mark Fahim: mqf5675@psu.edu
Haifa Alnajjar: haa5622@psu.edu
Tyler Glaze: tjg5990@psu.edu

Thank you for exploring Vocal Bridge! Let’s bridge the language gap together.
