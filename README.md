#  Machine learning applications 3 Classification 

Bu örnekte MINS (lise öğrencilerinin el yazılarıyla yazdıkları rakamlar)(70000 veri) veri kümesini kullanarak bir sınıflandırma işlemi gerçekleştireceğiz.

Kodu incelemeden önce aşağıdaki bilgileri okuyunuz(Bilgi amaçlı).

# Performans ölçüsü (çapraz doğrulama)

SGDC = bu sınıf, çok büyük veri setlerini verimli bir şekilde idare edebilir.
bu kısmendir çünkü: sgd eğitim örneklerini teker teker bağımsız olarak ele alır.
(buda sgd'yi çevrim içi öğrenme için çok uygun hale getirir)

# Eğitim ve Test verileri

Elimizde bulunan veri setini istediğimiz sayıda eşit parçalara ayırıyoruz.
örnek: 5 eşit parçaya ayıralım, bu 5 parça içerisinden 4 tane eğitim verisi, 1 tanede test verisi olarak ayırıyoruz.
Her seferinde farklı test kümesi alacak şekilde eğitim ve sınıflandırma işlemini 5 kere gerçekleştiriyoruz.
Sonunda her fazda elde ettiğimiz doğruluk değerinin ortalamasını alıyoruz. Sonuç bize sınıflandırma algoritmamızın doğruluk oranını verecektir.
Bu yöntem ile Eğitim ve Test verilerini parçalamış olduk.

![1280px-Çapraz_doğrulama_diyagramı svg](https://user-images.githubusercontent.com/54184905/75623207-dee6d000-5bb8-11ea-8576-53a5df2475b8.png)

# Karmaşıklık Matrisi

Karmaşıklık matrisi tahminlerin doğruluğu hakkında bilgi veren bir ölçüm aracıdır.
Arkasında yatan mantık aslında basit, ama ölçümün doğruluğu hakkında anlaşılması kolay bilgiler sağladığı için özellikle sınıflandırma algoritmalarında sıklıkla kullanılıyor.

![cost1](https://user-images.githubusercontent.com/54184905/75623289-8bc14d00-5bb9-11ea-9715-e83d301ac81a.png)

# F1 skoru

F 1 puan (aynı zamanda F-skor ya da F ölçü ) bir testin doğruluğunu bir ölçüsüdür. Hem gördüğü hassas p ve hatırlama r puanı hesaplamak için test: p sınıflandırıcı tarafından döndürülen tüm olumlu sonuçların sayısına bölünerek doğru pozitif sonuç sayısıdır ve r bölü doğru pozitif sonuç sayısıdır tüm ilgili örneklerin sayısı (pozitif olarak tanımlanması gereken tüm örnekler). F 1 skoru harmonik ortalamadırbir hassasiyet ve hatırlama bir F, 1 puanı en iyi 1 de değerini (mükemmel hassasiyet ve geri çağırma) ve 0 en kötü ulaşır.

![image](https://user-images.githubusercontent.com/54184905/75623349-39346080-5bba-11ea-8762-3aabee086f51.png)

# ROC Eğrisi

ROC eğrisinde ise (grafik) tüm eşik değerlere göre elde edilen sensitivite y ekseninde, spesifite ise (aslen 1- spesifite olarak) x ekseninde noktalar halinde belirtilerek en sonunda birleştirilir ve ortaya ROC eğrisi çıkar. Analiz sonunda AUC olarak belirtilen değer ‘Eğri Altında Kalan Alanı (EAA)’ temsil eder ve 1’e yaklaştıkça tanı değeri yükselir. %100’lük tanı gücünde EAA 1’e eşit olur. EAA ve %95 güven aralığı belirlenir. 0.50 değeri (fark yoktur) güven aralığı dışında ise istatistiksel olarak anlamlı sonuç elde edilir. Pratik olarak eşik değer eğri üzerinde sol üst köşeye (0;1 koordinatlarına) en yakın olan noktadır.

![image](https://user-images.githubusercontent.com/54184905/75623411-b9f35c80-5bba-11ea-90f0-f404dd3bd120.png)

# K-En Yakın Komşu Algoritması (Knn)

k=x olarak aldık. Bu durumda en yakın x komşuya göre sınıflandırma yapılacaktır. Örnek veri setine katılacak olan yeni verinin, mevcut verilere göre uzaklığı tek tek hesaplanır. İlgili uzaklık fonksiyonları yardımıyla. İlgili uzaklılardan en yakın k komşu ele alınır. Öznitelik değerlerine göre k komşu veya komşuların sınıfına atanır. veri setlerini internet üzerinden alıp sınıflandırma yapmaktadır.

![image](https://user-images.githubusercontent.com/54184905/75623497-8ebd3d00-5bbb-11ea-8bc5-6a611ccbfc15.png)



