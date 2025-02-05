# Multimodal-LSTM-Lyric-Generation Model
A multimodal LSTM-based model to generate song lyrics from melodies and lyrics combination.

### 1.	Goal of the project 
implements a multimodal LSTM-based model to generate song lyrics from melodies and lyrics combination. We chose to implement an LSTM-based model due to its ability to effectively manage long-term dependencies. This capability makes LSTMs a promising choice for our objective, which is to generate coherent and contextually consistent text.

### 2.	Pre-Processing:   

The preprocessing stage combines processing of two types of data: text (lyrics) and melody data (midi-files).

**Lyric Processing:** Lyrics are cleaned by normalizing file names (rename_files_to_lowercase function), removing punctuation and special characters (clean_text function), and tokenizing them into words. To create an informative representation of the lyrics for the  model, we used Word2Vec representations, retaining only the word that are in the Word2Vec vocabulary. (We tested embedding representations from different Word2Vec packages such as Text8, Google_News,   and GloVe and selected the one with the best performance.) To optimize the runtime of the song lyrics training process, the Word2Vec representation dictionary was reduced to include only words appearing in the songs. Tokens that occur less frequently (fewer than 10 times) in the song lyrics were excluded, and missing vectors were initialized randomly.
  	
**Melody Processing:** The melody processing stage involves extracting musical features from MIDI files to align with lyric segments for multimodal training. The extract_features_from_melody function defines a time window corresponding to a word's duration in the lyrics and analyses the MIDI content within this window. Key features were extracted. These features combined into a comprehensive feature vector that represent from now on the melody of each token. 
we implemented 2 methods for extracting features from the melody. The first is a simpler method that includes fewer features (23), and the second is a more extended method that includes additional features (153).

![image](https://github.com/user-attachments/assets/f4fd1609-6892-4c93-87f1-9f14a5ada1d7)
**general .mid file information**

![image](https://github.com/user-attachments/assets/f3378689-217a-41f9-887e-74557f4b3b12)
**timing information of the analyzed file**


### 3.	Data processing: 

The extracted MIDI features are normalized and concatenated with the word embeddings of the current lyric, forming a combined tensor that represents both textual and musical information. This combined tensor is stored in the X_input list. The target Y_target is created by encoding the next word in the sequence as a one-hot vector, which serves as the model's prediction output. Thus, X_input represents input data that combines word embeddings and MIDI features, while Y_target contains the next word to predict in the sequence, guiding the model's learning process. We do this processing to the train-set and the test-set as well. 

### 4.	Model architecture:
To conveniently integrate the melody, we chose to create two components: LyricsRNN and MelodyRNN that function as follows:

- **LyricsRNN** processes word embeddings representing song lyrics in a unidirectional LSTM setup.

- **MelodyRNN** handles MIDI features of a song's melody in both unidirectional and bidirectional configurations, which will allow us to examine two types of models: (1) melodic-bidirectional and (2) melodic-unidirectional by adjusting the parameter bidirectional_melody=True or False. (The melody can be analyzed bidirectionally because it is not the component we are trying to generate). 

Because in Model (1)- melodic-bidirectional we learn the entire representation of the melody, which represents the entire song, this enables the model to incorporate future notes as context, potentially improving the understanding of complex musical patterns and transitions. We also expect Model (1) to allow for better generalization between sequences of words and a particular musical style.

- **The CombinedModel** integrates the outputs of these two RNNs, taking both the lyrics and melody inputs, processing them through their respective RNNs, and then combining the outputs. 

We preformed A hyperparameters search (by using hyperparameter_search function) to find the most optimal model for the task.  

**In Model(1) melodic-bidirectional we used: optim.Adam, lr=0.01, CrossEntropyLos, num_epochs = 8, batch_size=64, (bidirectional_melody=True -> Instantiate in the CombinedModel  parameters).**

**In Model(2) melodic-unidirectional we used: optim.Adam, lr=0.001, CrossEntropyLos, num_epochs =4, batch_size=64, (bidirectional_melody=False -> Instantiate in the CombinedModel  parameters).**

In both model’s train process, we used Early stopping condition -> (validation_loss - val_loss) < 1e-6

**Train loss model 1 vs model 2 (A screenshot from the TensorBoard framework)**
![image](https://github.com/user-attachments/assets/af5f4aa1-0132-4777-b3ce-c91559814779)

**Validation loss model 1 vs model 2 (A screenshot from the TensorBoard framework)**
![image](https://github.com/user-attachments/assets/b406fee5-9a33-49b1-b42f-281486c84c5a)

**Accuracy loss model 1 vs model 2 (A screenshot from the TensorBoard framework)**
![image](https://github.com/user-attachments/assets/edc2bb0d-1761-4922-8e16-2cfa1a52bd4e)

### 5.	Generate the text: 
The lyrics generation process begins with an initial word and its Word2Vec embedding (If the word is not in the vocabulary, a random embedding is chosen), processed alongside melodic features. for each melody feature:
•	Pass the word and melody features through their respective RNN layers.
•	Combine their outputs into a shared representation.
Use the combined representation to predict the next word. The predicted word's embedding is used as the input for the next iteration until all melody features are processed or a maximum length is reached. The result is a sequence of lyrics shaped by the melodic patterns of the input features.

To analyse linguistic coherence, the pipeline incorporates n-gram embedding generation, which creates tensor representations for word sequences of various lengths (1-grams to 4-grams). These embeddings capture relationships among consecutive words, allowing for detailed assessment of lyrical structure. 
Our assumption is that as we increase the value of the n-gram parameter, we will be able to better measure the context between the majorette text and the original song.

In order to compute the similarity, we use cosine similarity. The cosine similarity function evaluates how closely the generated lyrics resemble the original lyrics at multiple n-gram levels. By comparing embeddings of the generated and ground truth lyrics, the function measures similarity, providing an overall score for lyrical alignment.

### 6. Results:

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/586b54fc-afd6-4f5b-95f8-8987835df67e" />

### 7. Analysis: 

unlike our expectations, the melodic-unidirectional model with performed better in most cases. 
This may be due to the bidirectional approach's added complexity, which can lead to overfitting and optimization challenges. Additionally, the unidirectional model may better align with the natural sequential flow of lyrics, where words are understood in temporal order.

**Cosine similarity results (Midi feature processes method #1):**
![image](https://github.com/user-attachments/assets/035fbbdb-4797-407f-85f7-70ea194ec747)

Comparing the simulation results, Model 2 achieves higher similarity in the n-grams (1-4), which may indicate Model 2's ability to better capture the wider context of the song. Interestingly, this occurs despite our initial assumption that incorporating a bidirectional component into Model 1's melody would enhance context capturing. One possible explanation is that the bidirectional melody in Model 1 focuses less on the connection between the lyrics and the melody, as it prioritizes being more generative and less specific.

**Cosine similarity results (Midi feature processes method #2):**

![image](https://github.com/user-attachments/assets/bd1561d1-f559-4160-b23c-9dd4c4149d4c)

When comparing the simulation results for Midi feature processes method #2, it is not possible to unequivocally determine which model is better.

**Analysis conclusion of the cosine similarity results:**

For all cosine similarity evaluations, we observed that increasing the value of n consistently resulted in higher similarity scores. While the exact reason for this trend is unclear, we hypothesize that larger values of n lead to more overlap in the representation of an "average" word, thereby increasing the overall similarity.

**Analysis how the models effects the Generated Lyrics:**

When examining the generated lyrics for both the melodic-bidirectional (Model 1) and melodic-unidirectional (Model 2) LSTM approaches, several key patterns emerge that reflect each model’s handling of long-range context and melodic features.

Model 1, which processes melody bidirectionally, often integrates more forward-looking references-sometimes introducing words or phrases (“dead,” “wild,” “la la la”) that appear out of immediate lyrical context yet give the sense of scanning broader segments of the song. This can lead to flashes of creative or unexpected wording but occasionally causes disconnected or repetitive segments. 
In contrast, Model 2’s unidirectional melody processing tends to generate more thematically grounded passages, exhibiting fewer abrupt shifts in tone or unexpected insertions—although it can produce simpler, more repetitive phrases. 

Moreover, the bidirectional model may weave melodic elements more freely into the text, reflecting future musical cues, while the unidirectional model stays closer to a linear, time-forward progression. 
Across the sample songs, both methods capture fragments of the original style but display different balances between adventurous, sometimes extraneous lines (Model 1) and more straightforward, local coherence (Model 2).

