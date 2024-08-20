import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier, MaxentClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data import data
import matplotlib.pyplot as plt

# Mikrofonu kullanarak metin alımı
def listen_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Dinliyorum...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='tr-TR')
        return text
    except sr.UnknownValueError:
        print("Anlayamadım.")
    except sr.RequestError as e:
        print(f"Ses hizmeti ile iletişim kurulamadı: {e}")
    return ""

# Özellik çıkarımı için metin işleme
def get_features(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('turkish'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return dict([(word, True) for word in filtered_words])

featuresets = [(get_features(text), label) for (text, label) in data]

classifiers = [
    NaiveBayesClassifier,
    DecisionTreeClassifier,
    MaxentClassifier
]

dt_params = {
    'criterion': 'entropy',
    'max_depth': 5,
    'min_samples_split': 10,
    'min_samples_leaf': 5
}

accuracies = {cls.__name__: [] for cls in classifiers}
for classifier_type in classifiers:
    for i in range(5):
        train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=i)
        classifier = classifier_type.train(train_set)
        accuracy = accuracy_score([label for (_, label) in test_set],
                                  [classifier.classify(features) for (features, label) in test_set])
        accuracies[classifier_type.__name__].append(accuracy)

# Grafik oluşturma
plt.figure(figsize=(10, 6))
for classifier_name, acc_values in accuracies.items():
    plt.plot(range(1, len(acc_values) + 1), acc_values, marker='o', label=classifier_name)
plt.title('Sınıflandırıcı Türlerine Göre Doğruluk Oranları')
plt.xlabel('Deneme Sayısı')
plt.ylabel('Doğruluk Oranı')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mikrofondan metni al ve en yüksek doğruluk oranına sahip olan sınıflandırıcı ile duygu analizi yap
text_to_analyze = listen_microphone()
if text_to_analyze:
    print(f"Metin: '{text_to_analyze}'")
    features = get_features(text_to_analyze)
    max_accuracy_classifier = max(accuracies, key=lambda k: max(accuracies[k]))
    classifier = None
    for cls in classifiers:
        if cls.__name__ == max_accuracy_classifier:
            classifier = cls
            break
    if classifier:
        trained_classifier = classifier.train(featuresets)
        sentiment = trained_classifier.classify(features)
        print(f"En yüksek doğruluk oranına sahip sınıflandırıcı: {max_accuracy_classifier}")
        print(f"Duygu: {sentiment}")
