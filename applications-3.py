import numpy as np
import os

# çıktıları sabit tutmak için
np.random.seed(42)

# grafik

#not defterine statik resimler çizmek
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12) #etiketler
mpl.rc('ytick', labelsize=12)

# rakamların kaydedileceği yer
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
#MNIST
    
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
    
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) #hedefleri string olarak döndürür
    sort_by_target(mnist) # fetch_openml() sıralanmış veri kümesi döndürür
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    mnist["data"], mnist["target"] #minst adlı değişkende rakamların ikilik sistemdeki hali mevcut


print("Data boyut = ",mnist.data.shape) #boyut

X, y = mnist["data"], mnist["target"]
print("rakamlar = ",X.shape) # x = Rakamlar

print(" y = etiketler ",y.shape)  #satir ve sütün boyutlarını ayrı ayrı yazdirdik

some_digit = X[68000] # rakam
some_digit_image = some_digit.reshape(28, 28) #pixel
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")


plt.show() # herhangi ibr rakamı ekrana çıkarttık

"""
70000 adet rakam var,
ve bu rakamlar 784 pixel ile depolanmıştır.
burada nupmy da bulunan reshape fonksiyonu ile 28x28 lik bir matris haline getiriyoruz.

"""

def plot_digit(data):
    image = data.reshape(28, 28) 
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    
"""
veri setinde tahminime göre ilk 7000 veri 0 sonraki 7000 veri 1 gibi ilerliyor

ve biz 0,1,2,3,4,5,6,7,8,9 rakamlarını ekrana alt alta yazdırdık

"""
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    
plt.figure(figsize=(9,20))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)

plt.show() # rakamları yazdırdık.

print("60000. rakamın veri etiketi",y[60000])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] #verileri test ve öğrnme için ayırıyoruz.

import numpy as np

shuffle_index = np.random.permutation(60000) #boyutu 60000 olan rastgele bir liste döndürdü.
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# ikili tabanda sınıflandırıcı

y_train_5 = (y_train == 5) #Tüm 5'ler için doğru, diğer tüm basamaklar için yanlış.
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier #sınıflandırıcı seçip eğitime başlıyoruz
"""
SGDC = bu sınıf, çok büyük veri setlerini verimli bir şekilde idare edebilir.
bu kısmendir çünkü: sgd eğitim örneklerini teker teker bağımsız olarak ele alır.
(buda sgd'yi çevrim içi öğrenme için çok uygun hale getirir)


"""

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42) #Tekrarlanabilir sonuçlar istiyorsanız,random_state parametresini ayarlamalıdır .
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

from sklearn.model_selection import cross_val_score #performans ölçüsü olarak çapraz doğrulama kullanıldı
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") 
"""

 elimizde bulunan veri setini istediğimiz sayıda eşit parçalara ayırıyoruz.
 örnek: 5 eşit parçaya ayıralım, bu 5 parça içerisinden 4 tane eğitim verisi, 1 tanede test verisi olarak ayırıyoruz.
 Her seferinde farklı test kümesi alacak şekilde eğitim ve sınıflandırma işlemini 5 kere gerçekleştiriyoruz.
 Sonunda her fazda elde ettiğimiz doğruluk değerinin ortalamasını alıyoruz. Sonuç bize sınıflandırma algoritmamızın doğruluk oranını verecektir.


"""

from sklearn.model_selection import StratifiedKFold #Tabakalı K-Katlama çapraz doğrulayıcı, Tren / test setlerindeki verileri bölmek için tren / test endeksleri sağlar.
from sklearn.base import clone #Aynı parametrelerle yeni bir tahminci oluşturur. 

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5): #seti parçalara ayırıyoruz.
    #her bir parçada 3 eğitim verisi ve 2test verisi olark ayırıyoruz.
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
    
    
from sklearn.base import BaseEstimator # tüm tahminciler için temel sınıf
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
    
never_5_clf = Never5Classifier()
print("cross val score",cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.model_selection import cross_val_predict # Her bir giriş veri noktası için çapraz doğrulanmış tahminler oluşturur

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) # x_train = eğitim verisi, y_train = eğitim verisinin etiketi

from sklearn.metrics import confusion_matrix # Bir sınıflamanın doğruluğunu değerlendirmek için karışıklık matrisini hesaplar.

confusion_matrix(y_train_5, y_train_pred) #karmaşıklık matrisi

y_train_perfect_predictions = y_train_5

confusion_matrix(y_train_5, y_train_perfect_predictions)

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred) 

recall_score(y_train_5, y_train_pred)

from sklearn.metrics import f1_score #İkili sınıflandırmanın istatistiksel analizinde F₁ skoru, bir testin doğruluğunun bir ölçüsüdür.
f1_score(y_train_5, y_train_pred)


y_scores = sgd_clf.decision_function([some_digit])
print("y_scores",y_scores)

threshold = 0
y_some_digit_pred = (y_scores > threshold)

threshold = 200000
y_some_digit_pred = (y_scores > threshold)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]
    
from sklearn.metrics import precision_recall_curve # tp / (tp + fp) , duyarlılık

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): # eksik yerleri tamamlama ve çizme
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])

plt.show()

print("test karşılaştırması",(y_train_pred == (y_scores > 0)).all())

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)

plt.show()

# ROC eğrileri

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
"""
Tanı testlerinin sonuçları eğer var/yok, yüksek/normal veya düşük/normal şeklinde iki sonuçlu ise testlerin başarısı; sensitivite, spesifisite, pozitif prediktif değer, negatif prediktif değer ölçütleri değerlendirilir. Ancak test sonuçları sürekli (Hemoglobin değeri, WBC sayısı, plazma glukoz konsantrasyonu, PSA değeri gibi)  ya da sıralı sayısal (BI-RADS gibi) sonuçlar şeklinde ise, bu testlerin tanı koymadaki başarılarını değerlendirmek biraz daha karmaşık olabilmektedir.


"""

def plot_roc_curve(fpr, tpr, label=None): #grafik
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)

plt.show()

from sklearn.metrics import roc_auc_score #Hesaplama Alanı, tahmin skorlarından Alıcı Çalışma Karakteristik Eğrisi (ROC AUC)

roc_auc_score(y_train_5, y_scores)

from sklearn.ensemble import RandomForestClassifier #karar ağacı sınıflandırıcısına uyan ve tahmin doğruluğunu iyileştirmek ve aşırı uyumu kontrol etmek için ortalamayı kullanan bir meta tahmincidir.
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # puan = pozitif sınıf proba
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)

plt.show()

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)

# Çok sınıflı sınıflandırma

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

some_digit_scores = sgd_clf.decision_function([some_digit])

from sklearn.multiclass import OneVsOneClassifier # Bu strateji, her sınıf çifti için bir sınıflandırıcı yerleştirmekten oluşur. Tahmin zamanında en çok oyu alan sınıf seçilir.
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=-np.infty, random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

"""
Ortalamayı kaldırarak ve birim varyansına ölçeklendirerek özellikleri standartlaştırma

Bir numunenin standart skoru xşu şekilde hesaplanır:

z = (x - u) / s


"""

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)

def plot_confusion_matrix(matrix):
    """renk çubuğu tercihi için."""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    
plot_confusion_matrix(conf_mx)

plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
#karışıklık matrisi hataları
plt.show()

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# hata anlizleri basamağının grafiği
plt.show()

# Çok etiketli sınıflandırma

from sklearn.neighbors import KNeighborsClassifier # k en yakın komşu algoritması

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# **Warning**: aşığıdaki kod bloğu uzun sürebilir. donanımınıza bağlı.

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1) # çapraz tahmin
f1_score(y_multilabel, y_train_knn_pred, average="macro")

# Çoklu çıkış sınıflandırması -------

noise = np.random.randint(0, 100, (len(X_train), 784)) # rakam üzerinde gürültü yaratma 
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
# gürültülü rakamıb resmi
plt.show()

knn_clf.fit(X_train_mod, y_train_mod) # x test verileri , y etiket test verileri
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)









