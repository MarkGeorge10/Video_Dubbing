Vocal Bridge – Group 3 – Mark Fahim & Haifa Alnajjar & Tyler Glaze

To run the code:

  •Run seq2seq/submit_job.slurm to schedule job for hpc ROAR to run seq2seq_MT.py to train machine translation from English to Turkish (this takes about 25 hours on a Roar Collab node)

  •Run transformers/submit_job-en-de-ita-spa.slurm to schedule job for hpc ROAR to run transformer_en_de_ita_spa.py to train machine translation from English to Italy, Spanish, German (this takes about 10 hours on a Roar Collab node for each language)

  •Run transformers/Final-TranslationModel-Hung-Port.ipynb to train machine translation from English to Hungrian (this takes about 10 hours on a Roar Collab node for each language)
 

Instructions to Run the Code:

Step 1:
    Run seq2seq/submit_job.slurm, transformers/submit_job-en-de-ita-spa.slurm, and transformers/Final-TranslationModel-Hung-Port.ipynb to generate models

Step 2:
    After step 1, run the main_notebook.ipynb notebook to proceed with generating video from english to target language(Spanish, Hungarian, Italian, Turkish, German) .

Note: you do not need to run this step 1 again
