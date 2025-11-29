# Artificial-Neural-Networks-Course-Midterm
I developed an artificial neural network-based classifier on a dataset loaded from scikit-learn, applied data preprocessing techniques, perform hyperparameter optimization, and interpret models using XAI (Explainable AI) methods.
__________________________________

# Yapay Sinir Ağları Dersi

**Ara Sınav Ödevi – MLP Sınıflandırma Projesi + XAI (SHAP) Analizi** 

Bu ara sınav ödevinde, scikit-learn’den yüklenen bir veri seti üzerinde yapay sinir ağı tabanlı bir sınıflandırıcı geliştirecek, veri ön işleme tekniklerini uygulayacak, hiperparametre optimizasyonu gerçekleştirecek ve modelleri XAI (Explainable AI) yöntemleri ile yorumlayacaksınız.

Aşağıdaki adımları eksiksiz tamamlamanız ve her bölüm sonunda kısa yorumlar eklemeniz gerekmektedir.

# 1. Veri Setinin Yüklenmesi
   
**1.1 scikit-learn’den Veri Seti Yükleme**

Kullanılabilecek veri setleri:

Breast Cancer Wisconsin

Wine Classification (seçtim)

**1.2 Veri Çerçevesi Oluşturma**

X (özellikler) ve y (hedef) değişkenlerini ayırın.

pandas DataFrame formatına dönüştürünüz.

İlk 5 satırı görüntüleyiniz.

# 2. Veri Seti Kalite Kontrolleri

**2.1 Eksik Değer Analizi**

Her sütunda missing value kontrolü yapınız.

Eksik değer varsa uygun yöntemle doldurunuz.

**2.2 Aykırı Değer (Outlier) Analizi**

Aşağıdakilerden en az birini uygulayınız:
IQR yöntemi

Z-score analizi

Boxplot incelemesi

**2.3 Veri Tipi ve Dağılım İncelemesi**

Sayısal / kategorik değişken sayılarını raporlayın.

Sütunların dtype bilgilerini gösterin.

# 3. Keşifsel Veri Analizi (EDA)

**3.1 İstatistiksel Özellikler**

Her sütun için aşağıdaki değerleri hesaplayın:

Mean

Median

Min–Max

Std

Q1–Q3

**3.2 Korelasyon Matrisi**
Pearson korelasyon matrisi oluşturun.
Heatmap ile görselleştirin.
En yüksek korelasyonlu 3 çift sütunu yorumlayın.
**3.3 Boxplot Analizi**
Tüm özellikler için boxplot çiziniz.
Aykırı değerleri yorumlayın.

# 4. Veri Ölçeklendirme (Scaling)
Aşağıdaki yaklaşımlardan biri kullanılabilir:
StandardScaler (önerilen)
MinMaxScaler
RobustScaler
Ölçeklendirilmiş veriyi X_scaled olarak kaydediniz.

# 5. Veri Setinin Bölünmesi
Veri şu şekilde bölünecektir:

%70 Training

%10 Validation

%20 Test

Not: Validation için ikinci bir train_test_split kullanılabilir.

# 6. Farklı MLP Modellerinin Kurulması

Aşağıdaki parametre kombinasyonlarıyla 5 farklı MLP modeli oluşturulacaktır:
