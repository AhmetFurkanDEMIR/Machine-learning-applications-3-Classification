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






