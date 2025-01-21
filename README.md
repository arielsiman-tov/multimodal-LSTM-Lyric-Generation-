# Multimodal-LSTM-Lyric-Generation Model
 A multimodal LSTM-based model to generate song lyrics from melodies and lyrics combination.

**1.	Goal of the project** - implements a multimodal LSTM-based model to generate song lyrics from melodies and lyrics combination. We chose to implement an LSTM-based model due to its ability to effectively manage long-term dependencies. This capability makes LSTMs a promising choice for our objective, which is to generate coherent and contextually consistent text.

**2.	Pre-Processing:**  The preprocessing stage combines processing of two types of data: text (lyrics) and melody data (midi-files).

**Lyric Processing:** Lyrics are cleaned by normalizing file names (rename_files_to_lowercase function), removing punctuation and special characters (clean_text function), and tokenizing them into words. To create an informative representation of the lyrics for the  model, we used Word2Vec representations, retaining only the word that are in the Word2Vec vocabulary. (We tested embedding representations from different Word2Vec packages such as Text8, Google_News,   and GloVe and selected the one with the best performance.) To optimize the runtime of the song lyrics training process, the Word2Vec representation dictionary was reduced to include only words appearing in the songs. Tokens that occur less frequently (fewer than 10 times) in the song lyrics were excluded, and missing vectors were initialized randomly.
  	
**Melody Processing:** The melody processing stage involves extracting musical features from MIDI files to align with lyric segments for multimodal training. The extract_features_from_melody function defines a time window corresponding to a word's duration in the lyrics and analyses the MIDI content within this window. Key features were extracted. These features combined into a comprehensive feature vector that represent from now on the melody of each token. 
we implemented 2 methods for extracting features from the melody. The first is a simpler method that includes fewer features (23), and the second is a more extended method that includes additional features (153).


**3.	Data processing:** The extracted MIDI features are normalized and concatenated with the word embeddings of the current lyric, forming a combined tensor that represents both textual and musical information. This combined tensor is stored in the X_input list. The target Y_target is created by encoding the next word in the sequence as a one-hot vector, which serves as the model's prediction output. Thus, X_input represents input data that combines word embeddings and MIDI features, while Y_target contains the next word to predict in the sequence, guiding the model's learning process. We do this processing to the train-set and the test-set as well. 

**4.	Model architecture:**
To conveniently integrate the melody, we chose to create two components: LyricsRNN and MelodyRNN that function as follows:

**LyricsRNN** processes word embeddings representing song lyrics in a unidirectional LSTM setup.

**MelodyRNN** handles MIDI features of a song's melody in both unidirectional and bidirectional configurations, which will allow us to examine two types of models: (1) melodic-bidirectional and (2) melodic-unidirectional by adjusting the parameter bidirectional_melody=True or False. (The melody can be analyzed bidirectionally because it is not the component we are trying to generate). 

Because in Model (1)- melodic-bidirectional we learn the entire representation of the melody, which represents the entire song, this enables the model to incorporate future notes as context, potentially improving the understanding of complex musical patterns and transitions. We also expect Model (1) to allow for better generalization between sequences of words and a particular musical style.

**The CombinedModel** integrates the outputs of these two RNNs, taking both the lyrics and melody inputs, processing them through their respective RNNs, and then combining the outputs. 
