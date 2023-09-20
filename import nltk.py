import nltk
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download necessary resources if not already downloaded
nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# Load Moby Dick text from Gutenberg dataset
moby_dick = gutenberg.raw("melville-moby_dick.txt")

# Tokenization
tokens = word_tokenize(moby_dick)

# Stopwords filtering
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# POS frequency
pos_counts = Counter(tag for word, tag in pos_tags)
most_common_pos = pos_counts.most_common(5)

print("Most common parts of speech and their counts:")
for pos, count in most_common_pos:
    print(f"{pos}: {count}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
top_20_tokens = [word for word, _ in FreqDist(filtered_tokens).most_common(20)]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in top_20_tokens]

print("\nTop 20 tokens after lemmatization:")
print(lemmatized_tokens)

# Plotting frequency distribution of POS
pos_counts = dict(pos_counts)
plt.figure(figsize=(12, 6))
plt.bar(pos_counts.keys(), pos_counts.values())
plt.xlabel("Parts of Speech")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of Parts of Speech")
plt.xticks(rotation=45)
plt.show()
