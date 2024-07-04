# Italian-Pluralisation

---
Extension of the Aragonese Pluralisation Problem

![Italian](https://github.com/AndrewBulata/Italian-Pluralisation/assets/64040990/35beb51f-c097-4117-abce-153deeca5571)



# Italian-Pluralisation

### Extension of the Aragonese Pluralisation Problem


This project extends the Aragonese Pluralisation problem to handle pluralisation in Italian. The primary goal is to develop a machine learning model that can accurately predict the plural forms of Italian nouns. The challenge is particularly interesting due to the complexity and variety of pluralisation rules in Italian, which include vowel mutation, suffixation, and consonant changes.

### Problem Statement

Languages such as Italian, Romanian, and others use different mechanisms to form plurals. Italian, for example, often relies on vowel changes and suffix modifications:

- **bambino - bambini** (child - children)
- **finestra - finestre** (window - windows)
- **casco - caschi** (helmet - helmets)
- **amico - amici** (masc. friend - friends)
- **collego - colleghi** (masc. colleague - colleagues)
- **soluzione - soluzioni** (solution - solutions)


Our task is to build a machine learning model that can learn these rules from a given dataset and accurately predict the plural forms of new words. This project focuses on Italian, but the methodology can be extended to other languages with similar pluralisation complexities.

### Approach

1. **Data Collection**: Gather a comprehensive list of Italian singular and plural word pairs.
2. **Feature Extraction**: Extract relevant features from the words, such as the last letter, vowel patterns, and word length.
3. **Model Training**: Use a Random Forest classifier to learn the pluralisation patterns from the training data.
4. **Prediction**: Implement a function to predict the plural form of new singular nouns based on the trained model.

### Data Format

The training data is stored in the text file `italian_word_pairs.txt` containing 500 entries, with each line containing a singular and plural pair separated by a comma, for example:

```
bambino,bambini
finestra,finestre
casco,caschi
```

### How to Use

1. **Prepare the Data**: Ensure you have a text file named `italian_word_pairs.txt` in the repository's root directory with your word pairs.

2. **Run the Script**: Execute the script to train the model and test the predictions.

### Example Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import re

# Function to read word pairs from a text file
def read_word_pairs(file_path):
    word_pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            singular, plural = line.strip().split(',')
            word_pairs.append((singular, plural))
    return word_pairs

# Define function to extract features
def extract_features(word):
    vowels = 'aeiou'
    last_vowel_index = max([word.rfind(v) for v in vowels])
    features = {
        'last_letter': word[-1],
        'last_two_letters': word[-2:],
        'last_three_letters': word[-3:],
        'length': len(word),
        'last_vowel': word[last_vowel_index] if last_vowel_index != -1 else '',
        'before_last_vowel': word[last_vowel_index-1] if last_vowel_index > 0 else ''
    }
    return features

# Define function to prepare and train the model
def train_pluralisation_model(word_pairs):
    df = pd.DataFrame(word_pairs, columns=['singular', 'plural'])

    df['features'] = df['singular'].apply(extract_features)
    df['suffix'] = df.apply(lambda row: row['plural'][len(row['singular']):], axis=1)

    X = df['features'].tolist()
    y = df['suffix']

    vectoriser = DictVectorizer(sparse=False)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=23)

    pipeline = Pipeline([
        ('vectoriser', vectoriser),
        ('classifier', classifier)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Function to predict plural forms
def predict_plural(model, singular_word):
    features = extract_features(singular_word)
    predicted_suffix = model.predict([features])[0]
    return singular_word[:-1] + predicted_suffix

# Read the word pairs from the text file
word_pairs = read_word_pairs('italian_word_pairs.txt')

# Train the model
model = train_pluralisation_model(word_pairs)

# Testing the function with new singular nouns
test_words = ['schermo', 'libro', 'coccodrillo', 'macchina', 'genitore', 'pulsante', 'università', 'palloncino', 'torta', 'luna']
predicted_plurals = [predict_plural(model, word) for word in test_words]

# Print the results
for singular, plural in zip(test_words, predicted_plurals):
    print(f"{singular} - {plural}")
```

And the output should be:

```
schermo - schermi
libro - libri
coccodrillo - coccodrilli
macchina - macchine
genitore - genitori
pulsante - pulsanti
università - università
palloncino - palloncini
torta - torte
luna - lune
```
### Conclusion

This project demonstrates how machine learning can be used to tackle the complex problem of pluralisation in natural languages. By extending the methodology used for Aragonese to Italian, we aim to create a versatile model capable of handling diverse linguistic patterns. Feel free to contribute by adding more word pairs, improving the feature extraction process, or extending the model to other languages.

---
