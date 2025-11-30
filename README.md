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

![Снимок экрана 2025-11-30 103317](https://github.com/user-attachments/assets/13c5f786-b732-48ca-9799-ac6b5d89cef0)

# 2. Veri Seti Kalite Kontrolleri

**2.1 Eksik Değer Analizi**

Her sütunda missing value kontrolü yapınız.

Eksik değer varsa uygun yöntemle doldurunuz.

![Снимок экрана 2025-11-30 103450](https://github.com/user-attachments/assets/dbf5f6e2-a335-4d25-a103-7fe9815efe92)

Scikit-learn'deki yerleşik Wine veri kümesinde eksik değer yoktur.

**2.2 Aykırı Değer (Outlier) Analizi**

Aşağıdakilerden en az birini uygulayınız:

IQR yöntemi (seçtim)

Z-score analizi

Boxplot incelemesi

Outlier analysis

📊 Column: alcohol

   Q1: 12.36, Q3: 13.68, IQR: 1.32
   
   Bound: [10.39, 15.65]
   
   Outliers: 0 (0.0%)

📊 Column: malic_acid

   Q1: 1.60, Q3: 3.08, IQR: 1.48
   
   Bound: [-0.62, 5.30]
   
   Outliers: 3 (1.7%)

📊 Column: ash
   Q1: 2.21, Q3: 2.56, IQR: 0.35
   
   Bound: [1.69, 3.08]
   
   Outliers: 3 (1.7%)

📊 Column: alcalinity_of_ash

   Q1: 17.20, Q3: 21.50, IQR: 4.30
   
   Bound: [10.75, 27.95]
   
   Outliers: 4 (2.2%)

📊 Column: magnesium
   Q1: 88.00, Q3: 107.00, IQR: 19.00
   
   Bound: [59.50, 135.50]
   
   Outliers: 4 (2.2%)

📊 Column: total_phenols
   Q1: 1.74, Q3: 2.80, IQR: 1.06
   
   Bound: [0.16, 4.39]
   
   Outliers: 0 (0.0%)

📊 Column: flavanoids
   Q1: 1.21, Q3: 2.88, IQR: 1.67
   
   Bound: [-1.30, 5.38]
   
   Outliers: 0 (0.0%)

📊 Column: nonflavanoid_phenols

   Q1: 0.27, Q3: 0.44, IQR: 0.17
   
   Bound: [0.02, 0.69]
   
   Outliers: 0 (0.0%)

📊 Column: proanthocyanins

   Q1: 1.25, Q3: 1.95, IQR: 0.70
   
   Bound: [0.20, 3.00]
   
   Outliers: 2 (1.1%)

📊 Column: color_intensity

   Q1: 3.22, Q3: 6.20, IQR: 2.98
   
   Bound: [-1.25, 10.67]
   
   Outliers: 4 (2.2%)

📊 Column: hue

   Q1: 0.78, Q3: 1.12, IQR: 0.34
   
   Bound: [0.28, 1.63]
   
   Outliers: 1 (0.6%)

📊 Column: od280/od315_of_diluted_wines

   Q1: 1.94, Q3: 3.17, IQR: 1.23
   
   Bound: [0.09, 5.02]
   
   Outliers: 0 (0.0%)

📊 Column: proline

   Q1: 500.50, Q3: 985.00, IQR: 484.50
   
   Bound: [-226.25, 1711.75]
   
   Outliers: 0 (0.0%)

📊 Column: target

   Q1: 0.00, Q3: 2.00, IQR: 2.00
   
   Bound: [-3.00, 5.00]
   
   Outliers: 0 (0.0%)

   <img width="1489" height="790" alt="2 3" src="https://github.com/user-attachments/assets/f25f3232-e9d7-4759-a139-f8a2d4c4d2e3" />

Neredeyse tüm özellikler, alkolden (alcohol) od280/od315_of_diluted_wines'a kadar, dikey eksende sıfıra yakın dar bir dağılıma sahiptir ve bunların IQR'leri (kutu) çok küçüktür veya zar zor fark edilebilir.

Proline özelliği, diğer tüm özelliklerden keskin bir şekilde ayrılır ve grafikte hakimiyet kurar.

1. Proline ve diğer tüm özellikler arasındaki büyük ölçek farkı.

2. Özellik Ölçeklendirmesi'ni (Feature Scaling) zorunlu olarak yapmak (gereklidir).

**2.3 Veri Tipi ve Dağılım İncelemesi**

Sayısal / kategorik değişken sayılarını raporlayın.

Sütunların dtype bilgilerini gösterin.

![Снимок экрана 2025-11-30 104113](https://github.com/user-attachments/assets/a012e46a-b783-4c7b-bc60-f16df9ce4e56)

# 3. Keşifsel Veri Analizi (EDA)

**3.1 İstatistiksel Özellikler**

Her sütun için aşağıdaki değerleri hesaplayın:

Mean

Median

Min–Max

Std

Q1–Q3

Sonuç

![Снимок экрана 2025-11-30 104322](https://github.com/user-attachments/assets/fe26a9b7-9747-4de1-a62e-0289e85ab878)

**3.2 Korelasyon Matrisi**

Pearson korelasyon matrisi oluşturun.

Heatmap ile görselleştirin.

En yüksek korelasyonlu 3 çift sütunu yorumlayın.

<img width="1117" height="882" alt="3 2" src="https://github.com/user-attachments/assets/cd145073-872f-43a4-b6b7-1a57b7549873" />

Top 3 most correlated pairs:

flavanoids - total_phenols: 0.865

total_phenols - flavanoids: 0.865

target - flavanoids: -0.847

İlk 3 çift, birbirleriyle en güçlü şekilde ilişkili olan işaretleri gösterir. Pozitif korelasyon, her iki işaretin birlikte büyüdüğü anlamına gelirken, negatif korelasyon birinin büyüdüğü, diğerinin azaldığı anlamına gelir.

**3.3 Boxplot Analizi**

Tüm özellikler için boxplot çiziniz.

Aykırı değerleri yorumlayın.

<img width="1490" height="790" alt="3 3" src="https://github.com/user-attachments/assets/05643fff-1523-4ca3-b33c-e013e85704c4" />
<img width="1779" height="590" alt="3 3 1" src="https://github.com/user-attachments/assets/30ef2fc3-9064-4095-8119-e45a4b8627fb" />

Analiz

1. Prolin (proline)
   
*   Sınıf 0, en yüksek medyan proline değerine ve aynı zamanda en büyük dağılıma (en yüksek kutu ve en uzun bıyıklar/aykırı değerler) sahiptir. Medyan yaklaşık 800-1000 civarındadır.

*   Sınıf 1, en düşük medyan proline değerine (yaklaşık 350-400) ve daha küçük bir dağılıma sahiptir.

*   Sınıf 2, Sınıf 0'a kıyasla orta düzeyde bir medyan değere (yaklaşık 600-700) ve daha küçük bir dağılıma sahiptir.

Sonuç: Proline özelliği, dağılımları minimum düzeyde örtüştüğü ve medyanları çok farklı olduğu için üç sınıf arasındaki en güçlü ayırıcı faktördür.

2. Magnezyum (magnesium)
   
Sınıf 1, açıkça en yüksek medyan magnesium değerine (yaklaşık 100) ve belirgin aykırı değerlerle birlikte büyük bir dağılıma sahiptir.

*   Sınıf 0 ve Sınıf 2, çok benzer ve daha düşük medyan değerlerine (yaklaşık 80-90) ve daha küçük bir dağılıma sahiptir.

*   Sonuç: Magnesium özelliği, Sınıf 1'i Sınıf 0 ve 2'den iyi bir şekilde ayırır.

3. Külün Alkalinitesi (alkalinity_of_ash)
   
*   Sınıf 1, Sınıf 0 ve Sınıf 2'den (medyanlar yaklaşık 15-17) daha yüksek bir kül alkalinitesine (medyan yaklaşık 20) eğilimlidir.

*   Sonuç: Alkalinity_of_ash özelliği, özellikle Sınıf 1'in ayrılmasına katkıda bulunur, ancak dağılımları büyük ölçüde örtüşmektedir.

4. Diğer Özellikler
   
*   Geriye kalan özelliklerin çoğu (örneğin, alcohol, malic_acid, flavanoids, color_intensity, vb.), çok düşük değerlere ve sınıflar arasında yüksek düzeyde örtüşmeye sahiptir, bu da onları sınıfları tek başına açıkça ayırmak için daha az etkili kılar. Örneğin, alcohol ve malic_acid, her üç sınıfta da çok yakın medyanlara ve dağılımlara sahiptir ve bu ölçekte zorlukla ayırt edilebilirler.

Grafikler, Sınıf 0, 1 ve 2'nin kimyasal profillerinde istatistiksel olarak anlamlı farklılıklar olduğunu ve bunun onların sınıflandırılmasına olanak tanıdığını göstermektedir.

*   Prolin, üç sınıfın tamamını ayırmak için anahtar özelliktir.

*   Magnezyum ve külün alkalinitesi, Sınıf 1'i diğer ikisinden ayırmaya yardımcı olur.

# 4. Veri Ölçeklendirme (Scaling)
Aşağıdaki yaklaşımlardan biri kullanılabilir:

StandardScaler (önerilen) (sectim)

MinMaxScaler

RobustScaler

Ölçeklendirilmiş veriyi X_scaled olarak kaydediniz.

# 5. Veri Setinin Bölünmesi

Veri şu şekilde bölünecektir:

%70 Training

%10 Validation

%20 Test

Not: Validation için ikinci bir train_test_split kullanılabilir.

```python

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_scaled = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
print("\nX_scaled shape:", X_scaled.shape)
```
X_scaled shape: (178, 13)

# 6. Farklı MLP Modellerinin Kurulması

Aşağıdaki parametre kombinasyonlarıyla 5 farklı MLP modeli oluşturulacaktır:

**Model 1 – Basit**

```python
- hidden_layer_sizes=(16,)
- activation="relu"
- learning_rate_init=0.001
  ```

**Model 2 – Orta**

```python
- hidden_layer_sizes=(32, 16)
- activation="relu"
- learning_rate_init=0.005
  ```

**Model 3 – Geniş**

```python
- hidden_layer_sizes=(64, 64)
- activation="tanh"
- learning_rate_init=0.001
  ```

**Model 4 – Derin**

```python
- hidden_layer_sizes=(128, 64, 32)
- activation="relu"
- learning_rate_init=0.0005
  ```

**Model 5 – Düşük Öğrenme Oranlı**

```python
- hidden_layer_sizes=(32,)
- activation="relu"
- learning_rate_init=0.0001
  ```

# 7. Validation Performanslarının Ölçülmesi

**Her model validation seti üzerinde aşağıdaki metrikler ile değerlendirilecektir:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Tüm modellerin performanslarını bir tablo hâlinde karşılaştırınız.

![Снимок экрана 2025-11-30 105126](https://github.com/user-attachments/assets/0293e50a-ebf0-4f53-9b10-9e59b8154bee)

# 8. En İyi Modelin Test Üzerinde Değerlendirilmesi

**Validation sonuçlarına göre en iyi modeli seçiniz.**

**Bu model için test seti üzerinde:**

**8.1 Performans Metrikleri**

- Accuracy
  
- Precision
  
- Recall
  
- F1-score
  
- ROC-AUC

![Снимок экрана 2025-11-30 105426](https://github.com/user-attachments/assets/15f83897-f2ee-432e-ac99-fc375adcdc57)

**8.2 Confusion Matrix**

- seaborn heatmap ile çiziniz.

 <img width="733" height="590" alt="8 2" src="https://github.com/user-attachments/assets/5230a158-33e0-4ab6-a42f-e178a223a682" />

**Sınıflandırma Sonuçları**

Model 5 Low Learning Rate bu veri kümesinde mükemmel bir performans sergilemiştir, çünkü diyagonal olmayan tüm öğeler sıfıra eşittir, bu da sınıflandırma hatası olmadığı anlamına gelir.

Sınıf 0 (class_0):

Gerçekte Sınıf 0'a ait olanlar: 12

Sınıf 0 olarak Doğru Tahmin Edilenler: 12

Sınıf 1 veya 2 olarak Yanlış Tahmin Edilenler: 0

Sınıf 1 (class_1):

Gerçekte Sınıf 1'e ait olanlar: 14

Sınıf 1 olarak Doğru Tahmin Edilenler: 14

Sınıf 0 veya 2 olarak Yanlış Tahmin Edilenler: 0

Sınıf 2 (class_2):

Gerçekte Sınıf 2'ye ait olanlar: 10

Sınıf 2 olarak Doğru Tahmin Edilenler: 10

Sınıf 0 veya 1 olarak Yanlış Tahmin Edilenler: 0

Veri kümesindeki toplam nesne sayısı tüm doğru tahminlerin toplamına eşittir: 12 + 14 + 10 = 36 nesne.

**8.3 ROC Eğrisi**

- ROC curve + AUC değeri

- Eşik değerinin performansa etkisini yorumlayın.

<img width="989" height="790" alt="8 3" src="https://github.com/user-attachments/assets/78e911d6-fd25-449c-872f-67076ede815a" />
<img width="989" height="590" alt="8 3 1" src="https://github.com/user-attachments/assets/ff1d9d86-2d59-46d2-95bb-e80b4a406ba6" />

Açıklaması

**ROC Curve**

Üç sınıfın (class_0, class_1, class_2) ROC eğrileri, grafiğin üst sınırında mükemmel bir şekilde yer almaktadır; yani, FPR=0 iken TPR=1.
AUC (Eğri Altındaki Alan) değerinin 1.00 olması, ideal sınıflandırma anlamına gelir. Model, her bir sınıf için pozitif nesneleri negatif nesnelerden kusursuz bir şekilde ayırabilir. Bu sonuç, Karışıklık Matrisinden (Confusion Matrix) çıkarılan, modelin bu veri kümesinde sıfır sınıflandırma hatası yaptığı sonucunu doğrulamaktadır.

**Learning Curve**

Grafik, iterasyonların başlangıcındaki maksimum değerden (yaklaşık 1.0) başlayarak ve eğitimin sonuna (1500+ iterasyon) doğru sıfıra (yaklaşık 0.05) yaklaşarak, kayıp fonksiyonunda (Loss) hızlı ve istikrarlı bir düşüş göstermektedir. Algoritma, düşük öğrenme hızıyla (Low LR) iyi bir yakınsama (convergence) sergilemektedir.

# 9. Optuna ile Hiperparametre Optimizasyonu (150 Deneme)

**9.1 Optuna Study Tanımı**

- direction="maximize"

- metric: validation accuracy veya F1-score

**9.2 Optuna Arama Aralıkları**

```python
hidden_layer_sizes:    (trial.suggest_int(16, 256), trial.suggest_int(8, 128))
learning_rate_init:    trial.suggest_loguniform(1e-5, 1e-1)
alpha:    trial.suggest_loguniform(1e-6, 1e-2)
activation:    trial.suggest_categorical(["relu", "tanh"])
solver:    trial.suggest_categorical(["adam", "sgd"])
batch_size:    trial.suggest_categorical([16, 32, 64, 128])
```

**Kullanılan model**

```python
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
```

**Parametrelerin etkisi**

- hidden_layer_sizes: Bu parametre, sinir ağının yapısını doğrudan belirler. Gizli katman sayısını ve her katmandaki nöron sayısını ayarlarsınız. Bu sayıları artırmak, modeli daha karmaşık hale getirir ve verilerdeki doğrusal olmayan ve karmaşık ilişkileri öğrenme yeteneğini artırır. Ancak bu aynı zamanda aşırı öğrenme (overfitting) riskini ve eğitim süresini de artırır.

- activation: Bu, gizli katmanlardaki her nöronun çıkışına uygulanan aktivasyon fonksiyonudur. Modele doğrusal olmayanlık katar ve ağın karmaşık fonksiyonları modellemesine olanak tanır. Örneğin, 'relu' (Rectified Linear Unit), eğitimi hızlandırdığı ve gradyanın kaybolması (vanishing gradient) sorununu çözmeye yardımcı olduğu için en popüler varsayılan seçimdir.

- solver: Bu, hatayı (kayıp fonksiyonunu) en aza indirmek için ağın ağırlıklarını optimize etmekte kullanılan algoritmadır.

- 'adam' – Çoğu görev için mükemmel bir varsayılan seçimdir; her ağırlık için öğrenme oranını uyarlanabilir şekilde ayarlayarak hızlı yakınsama sağlar.

- 'sgd' – Öğrenme oranının manuel olarak ayarlanmasını gerektirir, ancak bazen daha iyi genelleme performansı sunabilir.

- learning_rate_init: Başlangıç öğrenme oranıdır – optimizasyon sırasında ağırlıkların ne kadar büyük adımlarla ayarlanacağını belirler. Çok büyük bir değer, eğitimin dengesiz olmasına ve optimum noktanın "atlanmasına" neden olabilir. Çok küçük bir değer ise eğitimin çok yavaş ilerlemesine yol açar.

- batch_size: Her eğitim adımında gradyanı hesaplamak için kullanılan veri paketinin (kümesinin) boyutudur. Daha büyük paket boyutu, daha doğru gradyan tahmini sağlar, ancak yerel minimumlarda takılıp kalmaya yol açabilir. Daha küçük paket boyutu, sürece "gürültü" katarak ağın yerel minimumlardan kurtulmasına ve genelleme yeteneğini artırmasına yardımcı olabilir.

- max_iter: Maksimum epoch sayısıdır, yani tüm eğitim veri kümesi üzerinde yapılan tam geçiş sayısıdır. Modelin eğitime harcayacağı sürenin üst sınırını belirler.

- alpha: Bu, L2-düzenlileştirme katsayısıdır. Kayıp fonksiyonuna, ağırlıkların büyüklüğünün karesiyle orantılı bir ceza ekler. Bu cezanın amacı: ağırlıkların daha küçük olmaya zorlanmasıdır. Bu, modeli basitleştirmeye ve böylece aşırı öğrenmeyi önlemeye yardımcı olur, modeli eğitim verilerindeki gürültüye karşı daha az duyarlı hale getirir.

- early_stopping: Bu parametrenin True olarak ayarlanması, aşırı öğrenmeyle mücadele etmenin en iyi mekanizmalarından birini etkinleştirir. Model, doğrulama (validation) kümesindeki performansı iyileşmeyi durdurursa otomatik olarak eğitimi durdurur.

- validation_fraction: Erken durdurma kararını vermek için kullanılan doğrulama kümesi olarak otomatik olarak ayrılacak eğitim verisi oranını belirler. Modelin kalitesindeki iyileşme bu küme üzerinden izlenir.

- n_iter_no_change: Modelin "sabır" süresini ayarlar. Erken durdurmanın etkinleştirilmesinden önce, doğrulama kümesinde iyileşme olmaması gereken maksimum epoch (iterasyon) sayısıdır.

- random_state: Rastgele sayı üretecisini sabitler ve böylece ağın başlangıç ağırlıklarının ve veri bölmelerinin her çalıştırmada aynı olmasını sağlar. Bu, deneyinizin tekrarlanabilirliğini garanti eder.

Best F1-score: 1.0

Best params: {'layer1_size': 242, 'layer2_size': 123, 'learning_rate_init': 0.04565176372646328, 'alpha': 3.0243691612726458e-05, 'activation': 'tanh', 'solver': 'sgd', 'batch_size': 16}

![Снимок экрана 2025-11-30 105724](https://github.com/user-attachments/assets/faa6d3d5-efaf-4614-9395-a31f25d5ee4e)

**9.3 Eğitim Döngüsü**

- Her trial bir MLPClassifier modeli kurup eğitir.

- Validation skorunu geri döndürür.

  Optuna çatısı kullanılarak yapılan otomatik hiperparametre optimizasyonu sürecinde. Bu denemelerin (trial) her birinin içinde aşağıdaki eylem dizisi gerçekleştirilir:

**Modelin İnşası ve Eğitimi**

model.fit(X_train_scaled, y_train)

**Modelin Doğrulama Kümesinde Değerlendirilmesi**

y_val_pred = model.predict(X_val_scaled)

**İki temel performans metriği hesaplanır:**

val_accuracy ve val_f1

Optuna bağlamında ve genel olarak hiperparametre optimizasyonu görevlerinde, farklı denemeler (trial'lar) arasında "ince ayar" (fine-tuning) veya "eğitime devam etme" (continuation of training) kullanılmaz.

** Optuna Neden Eğitime Devam Etmez?**

Optuna'nın Amacı Karşılaştırmadır: Optuna'nın ana görevi, en iyi hiperparametre kombinasyonunu (örneğin, alpha, learning_rate_init, hidden_layer_sizes) bulmaktır. Örneğin, A kombinasyonunu (küçük alpha ile) B kombinasyonuyla (büyük alpha ile) dürüstçe karşılaştırmak için, her modelin sıfırdan ve bağımsız olarak eğitilmesi gerekir.

**9.4 En İyi Trial’ın Raporlanması**

- En iyi parametre setini yazdırınız.

- Validation metriklerini gösteriniz.

![Снимок экрана 2025-11-30 105724](https://github.com/user-attachments/assets/cfc3f4d3-180b-4a88-b8f0-b49f562b1b66)

![Снимок экрана 2025-11-30 110900](https://github.com/user-attachments/assets/e7f9cafb-76ec-45ff-879a-58f59e066f6d)

**Sonuç:**

Hiperparametre optimizasyonu başarılı bir şekilde tamamlanmıştır ve doğrulama kümesinde son derece yüksek bir metrik (F1-Score: 1.0000) sağlayan parametre kombinasyonlarını bulmaya olanak tanımıştır.

Ancak, en iyi denemenin metrikleri ile bu parametrelere sahip eğitilmiş modelin (muhtemelen başka veya nihai) doğrulama/test kümesindeki metriklerinin karşılaştırılması, potansiyel aşırı uydurma (overfitting) olduğunu göstermektedir.

# 10. XAI – SHAP Açıklanabilirlik Analizi (Zorunlu)

Bu bölümde modellerinizin nasıl karar verdiğini açıklayacaksınız.

**10.1 Beş MLP Modelinden validasyon başarısına göre seçilen model için SHAP Analizi**

En iyi  model için:

- SHAP Explainer oluşturun

- summary_plot gösterin

- bar_plot (feature importance) çizdirin

- En baskın özellikleri yorumlayın

- Model performansı ve SHAP önem sıralaması arasındaki ilişkiyi tartışın

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic

Bu bölümün analizi, iki anahtar model üzerinde gerçekleştirilecektir: doğrulama (validation) metriğine göre ilk beş MLPClassifier modeli arasından seçilen en iyi model (Model 5) ve Optuna kullanılarak yapılan otomatik hiperparametre optimizasyonu sonucunda elde edilen küresel olarak en iyi model.

# Model 5

**summary_plot gösterin**

<img width="584" height="680" alt="10 1" src="https://github.com/user-attachments/assets/fc691996-f691-4ade-8526-82b290e3ff0c" />

Grafik, üç özellik (alcohol, malic_acid ve ash) arasındaki etkileşimi göstermektedir.

**Etkileşimin Yorumlanması (Sütunlara Göre)**

**Köşegen Hücreler (Diagonal Cells)**

Bu hücreler teknik olarak "ana etkiyi" (main effect) gösterir, saf etkileşimi değil. Ancak bu tür grafiklerde genellikle özelliğin SHAP değerini göstermek için kullanılırlar.

alcohol (alcohol satırı, alcohol sütunu): Yüksek alcohol değeri (kırmızı noktalar), genellikle pozitif bir SHAP değeri (sağa kayma) ile ilişkilidir; düşük değerler (mavi) ise negatif bir SHAP değeri (sola kayma) ile ilişkilidir. Bu, yüksek alcohol seviyesinin tek başına tahmini artırdığı (ve tersi) anlamına gelir.

**alcohol ve malic_acid Etkileşimi (alcohol-malic_acid)**

İlgili hücreye bakıyoruz: alcohol satırı, malic_acid sütunu.

Noktaların çoğunlukla 0 civarında kümelendiği görülüyor.

Az sayıda kırmızı nokta (yüksek malic_acid), 0'ın sağında yer alıyor. Bu, hem alcohol hem de malic_acid yüksek değerlere sahip olduğunda, ortak etkilerinin hafifçe pozitif olabileceğini (tahmini güçlendirdiğini) gösterebilir.

**ash ve alcohol Etkileşimi (ash-alcohol)**

İlgili hücreye bakıyoruz: ash satırı, alcohol sütunu.

Burada da noktaların çoğu 0 civarındadır.

Bazı kırmızı noktalar (yüksek alcohol) ve mavi noktalar (düşük alcohol), küçük bir negatif veya pozitif SHAP etkileşim değerine sahiptir. alcohol düşükken (mavi) ve ash değişirken, etkileşim biraz daha negatif (sola kayma) gibi görünüyor, ancak genel etkileşim etkisi küçüktür.

**malic_acid ve ash Etkileşimi (malic_acid-ash)**

İlgili hücreye bakıyoruz: malic_acid satırı, ash sütunu.

Noktaların büyük bir kısmı sıfırda yoğunlaşmıştır. Bu, malic_acid ve ash arasındaki etkileşimin modelin tahmini üzerinde minimum düzeyde etkiye sahip olduğunu gösterir.

**Genel Sonuç**

Grafik şunu gösteriyor:

Özelliklerin ana etkisi (köşegen) en belirgin olanıdır: alcohol, malic_acid ve ash özellikleri tek başlarına tahmin üzerinde önemli bir etkiye sahiptir.

Özellik çiftleri arasındaki etkileşimler (alcohol-malic_acid, alcohol-ash, malic_acid-ash) ana etkilerine kıyasla zayıftır (noktaların çoğu 0'a çok yakındır). Bu, bir özelliğin tahmin üzerindeki etkisinin, o çiftteki diğer özelliğin değerine büyük ölçüde bağlı olmadığı anlamına gelir.

**bar_plot (feature importance) çizdirin**

<img width="754" height="299" alt="10 1 1" src="https://github.com/user-attachments/assets/070a72fc-aed3-48cf-9547-96d729e5d78a" />

0th alcohol olarak adlandırılan özellik, bu üç kategori/sınıf arasında en önemli olanıdır veya tahmin üzerinde en büyük ortalama etkiye sahiptir; buna karşılık 2nd alcohol ise en az etkiye sahiptir.

<img width="789" height="660" alt="10 1 2" src="https://github.com/user-attachments/assets/4c5e6963-a687-4283-b4bd-c9f38b91f9cd" />

Özelliklerin Genel Önemi (Çubuk Uzunluğu)
Modele en büyük genel etkiyi yapan en önemli özellikler (en uzun çubuklar) şunlardır:

alcohol: Modeldeki açık ara en önemli özelliktir.

ash

od280/od315_of_diluted_wines (Muhtemelen şarabın saflığının bir ölçüsüdür)

proline

En az önemli özellikler: magnesium, proanthocyanins, nonflavanoid_phenols.

Model, şarapları sınıflandırmak için temel olarak alcohol, ash ve od280/od315_of_diluted_wines özelliklerine dayanmaktadır. Bu özelliklerin etkisi farklı sınıflar için aynı değildir: örneğin, Class 0'ı ayırt etmek için model çoğunlukla alcohol seviyesine bakarken, Class 2'yi ayırt etmek için od280/od315 göstergesine bakar.

**10.2 Optuna En İyi Model için SHAP Analizi**

Aşağıdaki SHAP görselleştirmeleri yapılacaktır:

- summary_plot (dots) → tüm verideki önem sıralaması

- bar_plot → ortalama etki büyüklükleri

- force_plot → tek bir örnek için karar açıklaması

- decision_plot → karar yolunun özellere göre katkısı

# best_model

**summary_plot (dots) → tüm verideki önem sıralaması**

<img width="578" height="680" alt="10 2" src="https://github.com/user-attachments/assets/28a7461f-6f91-4ce8-bdf1-9824999a469f" />

Optuna kullanılarak optimize edilmiş model, alcohol ve malic_acid özellikleri arasında olağanüstü güçlü bir etkileşim sergilemektedir.

Bu, alcohol ve malic_acid'in etkisinin ayrı ayrı analiz edilemeyeceği anlamına gelir. Model, bu iki özelliğin kombinasyonunu güçlü bir öngörücü olarak kullanmayı öğrenmiştir; burada bir özelliğin etkisi, diğerinin değeri tarafından ya güçlendirilmekte ya da zayıflatılmaktadır.

**bar_plot → ortalama etki büyüklükleri**

<img width="754" height="299" alt="10 2 1" src="https://github.com/user-attachments/assets/a90770c1-451d-40ec-b3b7-89ec08c0cb60" />

Bu grafik, optimize edilmiş modelde 'alcohol' özelliğinin üç farklı sınıf/kategori (0th, 1st, 2nd) üzerindeki tahmin için olan ortalama mutlak etkisini (mean(|SHAP value|)) göstermektedir.

**Yorumlama**

- 0th alcohol: En büyük ortalama mutlak etkiye (+0.05) sahiptir. Bu, alcohol özelliğinin Sınıf 0'ı ayırt etme veya tahmin etme girişiminde model için en kritik olduğu anlamına gelir.

- 1st alcohol: Neredeyse aynı büyük ortalama mutlak etkiye (+0.05) sahiptir. Bu, alcohol özelliğinin Sınıf 1'i ayırt etmek veya tahmin etmek için de çok önemli olduğu anlamına gelir.

- 2nd alcohol: En düşük ortalama mutlak etkiye (+0.02) sahiptir. Bu, Sınıf 2 tahmininde (diğer sınıflara kıyasla) modelin ortalama olarak alcohol özelliğine daha az güvendiğini gösterir.

**X Ekseni Değerleri ile Sağdaki Etiketler Arasındaki Uyumsuzluk**

**Grafiğe dikkatlice bakıldığında:**

- 0th alcohol" çubuğu, +0.05 değerine karşılık gelecek şekilde X ekseninin en sonuna ulaşır. Burada etiket ve çubuk uyumludur.

- 1st alcohol" çubuğu, "0th alcohol" çubuğundan açıkça daha kısadır ve yaklaşık olarak 0.045 seviyesinde durmaktadır. Ancak sağdaki etiket +0.05 göstermektedir.

- 2nd alcohol" çubuğu yaklaşık 0.02'de durur ve bu da +0.02 etiketiyle uyumludur.

**Sonuç:**

1st alcohol etkisinin sayısal olarak +0.05 olarak yuvarlanmasına rağmen, görsel olarak 0th alcohol'ün etkisinden (örneğin 0.051 olabilir) biraz daha küçüktür. Bununla birlikte, yorumlama amacıyla 0th alcohol ve 1st alcohol'ü eşit derecede önemli özellikler olarak kabul ederiz, çünkü ortalama etkileri aynı yuvarlanmış değer içindedir ve her ikisi de modeldeki baskın faktörlerdir.

**force_plot → tek bir örnek için karar açıklaması**

![Снимок экрана 2025-11-30 115102](https://github.com/user-attachments/assets/4b5f41c2-81ed-40b4-8332-9730e9cff8f5)

Bu grafik, sınıflandırma görevindeki tek bir somut örnek (instance) (indeks 169) ve tek bir belirli sınıf (Sınıf 1) için bir SHAP Kuvvet Grafiğini (Force Plot) temsil etmektedir.

Amacı, bu spesifik örnekteki her bir özelliğin, modelin tahminini Sınıf 1 için Temel Değerden (Base Value) Nihai Tahmine (Output Value) doğru nasıl kaydırdığını açıklamaktır.

**Anahtar Öğeler**

- Temel Değer (Base Value): Ortada gösterilir (yaklaşık 0.4033). Bu, eğitim veri setinin tamamı için Sınıf 1'in ortalama beklenen model çıktısı değeridir (log-odds veya olasılıklar cinsinden).

- Çıktı Değeri (Output Value): Yatay ölçekte sol tarafta gösterilir (100% işaretinin bulunduğu kırmızı/mavi alanın bitiş noktası). Bu, modelin bu örnek ve Sınıf 1 için yaptığı fiili tahmindir (yaklaşık -0.1967).

- Özellikler (Features):

  - Kırmızı Oklar/Etiketler: Tahmini artıran (sağa doğru kaydıran, yani -0.1998'den temel değer 0.4002'ye doğru iten) özelliklerdir.

  - Mavi Oklar/Etiketler: Tahmini azaltan (sola doğru kaydıran) özelliklerdir.

**Yorumlama**

Yatay çizgi, Temel Değerden nihai tahmine giden "yolu" gösterir:

- Temel Değer (Base Value) 0.4033'dir.

- Bu örnek için Fiili Tahmin (Sınıf 1) -0.1967'de sona ermektedir.

Model, nihai değer (-0.1967) Temel Değerden (0.4033) belirgin şekilde düşük olduğu için bu örneğin Sınıf 1'e ait olmadığını tahmin etmektedir.

**Özellik Katkısı:**

Tahmini azaltmaya en çok katkıda bulunan özellikler (mavi, sola çekenler):

flavanoids = 1.101 (En güçlü azaltıcı etkiyi gösterir).

proanthocyanins

alcohol = 12.83

hue = 1.23

Tahmini artırmaya en çok katkıda bulunan özellikler (kırmızı, sağa çekenler):

alcalinity_of_ash = 15.1 (Güçlü artırıcı etki).

color_intensity = 1.478

ash = 1.62

**Sonuç:**

Bu özel örnek için:

Yüksek flavanoids, proanthocyanins ve alcohol değerleri, Sınıf 1'e ait olma olasılığını güçlü bir şekilde düşürmüştür.

alcalinity_of_ash, color_intensity ve ash'in belirli değerleri, Sınıf 1'e ait olma olasılığını yükseltmeye çalışmıştır.

Ancak, tahmini azaltan özelliklerin toplam etkisi belirgin şekilde daha güçlü olmuş ve sonuç olarak Sınıf 1 için nihai tahmin (-0.1967) ortalama temel değerin oldukça altında kalmıştır.


**force_plot → tek bir örnek için karar açıklaması**

![Снимок экрана 2025-11-30 115129](https://github.com/user-attachments/assets/9bbb7c7e-0633-40d0-af10-e246b8c12adc)

**Özellik Katkısı:**

Tahmini azaltmaya en çok katkıda bulunan özellikler (mavi, sola çekenler):

malic_acid = 1.736: Çok güçlü bir azaltıcı etki gösterir.

hue = -1.675 (Bu negatif değer, ya verilerdeki ya da kodlamadaki bir hataya işaret ediyor olabilir ya da sadece hue özelliğinin çok düşük bir değerde olduğunu gösteriyordur).

flavanoids = 1.207

ash = 0.06168

**Tahmini artırmaya en çok katkıda bulunan özellikler (kırmızı, sağa çekenler):**

proline = -1.6745 (Dikkat edin, proline'ın düşük değeri tahmini artırıyor. Bu, düşük proline seviyesinin Sınıf 1 için pozitif bir öngörücü olduğu anlamına gelir).

alcohol = 1.094

color_intensity = 1.693

**Sonuç:**

Tahmini azaltan özelliklerin (özellikle malic_acid) kümülatif etkisi, artıran özelliklerin etkisinden çok daha ağır bastı. Bu durum, Sınıf 1 için nihai tahminin çok düşük olmasına yol açmıştır.

![Снимок экрана 2025-11-30 115206](https://github.com/user-attachments/assets/cb319e1e-7ce1-40ca-a00d-24312f016be1)

Bu, SHAP Kuvvet Grafiğinin dinamik veya özetlenmiş bir görünümüdür (SHAP Force Plot). Modelin tahminlerinin çok sayıda veri örneği üzerinde nasıl değiştiğini gösterir.

**Anahtar Öğeler**

Yatay Eksen (X): Tahminlerin benzerliğine göre sıralanmış örneklerin indeksi (genellikle tahmin edilen f(x) değerinin azalan veya artan sırasına göre).

Dikey Eksen (Y): f(x). Bu, her örnek için modelin fiili tahmin edilen değeridir (log-odds veya olasılıklar cinsinden).

Renkler ve Katmanlar: Farklı renkler, farklı özelliklerin tahmine olan katkısını temsil eder.

Kırmızı Alanlar: f(x) tahminini artıran özellikler (pozitif katkı).

Mavi Alanlar: f(x) tahminini azaltan özellikler (negatif katkı).

Katman Genişliği: Belirli bir renkteki katman ne kadar geniş (kalın) ise, o özelliğin tahmine katkısı o kadar büyüktür.

Feature 0 ve Feature 12 Etiketleri: Katkıları genel resmi oluşturan en önemli özelliklere işaret eder.

**Yorumlama**

1. Tahminlerin Genel Dinamiği

Sol Kısım (Örnekler 0 ila 11): f(x) tahmini (dikey eksen) 0.7305 ile 0.1303 aralığındadır. Burada mavi katmanlar (azaltıcı özellikler) baskındır veya önemli bir genişliğe sahiptir, bu da bu örneklerin çoğu için özelliklerin belirgin bir negatif katkı sağladığını gösterir.

Sağ Kısım (Örnekler 12 ila 16): f(x) tahmini keskin bir şekilde yükselir (1.131'e kadar). Bu aralıkta kırmızı katmanlar (artırıcı özellikler) baskın ve geniş hale gelir.

2. Belirli Özelliklerin Katkısı

Feature 0 (Mavi Katman): Grafiğin sol kısmında (düşük tahmin), Feature 0 (önceki grafiklere dayanarak muhtemelen alcohol veya flavanoids) güçlü bir negatif katkı (mavi renk) yapar. Katkısı, tahmini düşük seviyede tutar.

Feature 12 (Kırmızı Katman): Grafiğin sağ kısmında (yüksek tahmin), Feature 12 çok güçlü bir pozitif katkı (kırmızı renk) yapar. Bu, diğer katkıları keskin bir şekilde geride bırakarak son örnekler için tahmini hızla yukarı çeken baskın faktördür.

**Fareyle Üzerine Gelindiğindeki Bilgiler (İndeks ~6)**

İndeks 6 çevresindeki alana fareyle gelindiğinde şu değerler görünür:

Feature 7 = -1.431

Feature 8 = 0.23

Feature 9 = -1.227

Bu etiketler, o noktadaki belirli örnek için bu özelliklerin değerlerini gösterir ve bu üç özelliğin, bu bölge için tahmine en önemli katkıyı yapanlar olduğunu belirtir.

**decision_plot → karar yolunun özellere göre katkısı**

<img width="792" height="659" alt="10 2 3" src="https://github.com/user-attachments/assets/59105a3a-0fda-4472-8a1a-857c1342aa6f" />
<img width="790" height="659" alt="10 2 4" src="https://github.com/user-attachments/assets/28e5f47e-a3dc-4197-be9e-9df3975273e0" />
<img width="791" height="659" alt="10 2 5" src="https://github.com/user-attachments/assets/b7b853b5-c6f2-47da-a970-0c5f578e3ca0" />

Bu üç grafik, sırasıyla Sınıf 0, Sınıf 1 ve Sınıf 2 için SHAP Karar Grafikleridir. Bunlar, modelin seçilen beş örnek (Samples 0-4) için her bir sınıfa yönelik tahmin kararını nasıl oluşturduğunu görselleştirir.

**Öğeler**

1. Yatay Eksen (X): Model Çıktı Değeri (Model Output Value). Bu, tahminin değeridir (olasılıklar veya log-odds). Ortadaki gri dikey çizgi Temel Değeri (Beklenen Değer) gösterir.

2. Dikey Eksen (Y): Katkılarına göre sıralanmış özelliklerin listesi.

3. Çizgiler: Her çizgi (farklı renkler/stiller), bir örneğin (Sample 0, 1, 2, 3, 4) tahminini temsil eder.

Çizgi, en alttaki etiketten (Temel Değerden başlar) ve her bir özellik yukarı doğru dikkate alındıkça sola/sağa doğru kayar.

4. Kayma (Katkı):

Sağa Kayma (Kırmızı Renk): Özellik, o sınıf için tahmini artırır (pozitif katkı).

Sola Kayma (Mavi Renk): Özellik, o sınıf için tahmini azaltır (negatif katkı).

**Genel Yorumlama (Sınıf Karşılaştırması)**

1. Karar Grafiği – Sınıf 0

Temel Değer: Yaklaşık 0.35.

Dinamik: Sample 1 ve Sample 4 (kırmızı çizgiler) çok yüksekte (yaklaşık 0.9–1.0) sonlanır, bu da modelin bunların Sınıf 0'a ait olduğuna dair güçlü bir güvene sahip olduğunu gösterir.

Tahmini artıran temel faktörler (sağa kayma): alcohol, proline, ash.

Sample 0, 2, 3 (mavi/mor çizgiler) düşükte (yaklaşık 0.0–0.2) sonlanır, yani büyük olasılıkla Sınıf 0'a ait değillerdir.

Tahmini azaltan temel faktörler (sola kayma): od280/od315_of_diluted_wines, alcalinity_of_ash.

2. Karar Grafiği – Sınıf 1

Temel Değer: Yaklaşık 0.45.

Dinamik: Sample 2 ve Sample 3 (kırmızı/mor çizgiler) yüksekte (yaklaşık 0.7–0.9) sonlanır.

Tahmini artıran temel faktörler: ash, proline, color_intensity, hue.

Sample 0, 1, 4 (mavi çizgiler) düşükte (yaklaşık 0.0–0.2) sonlanır, yani Sınıf 1'e ait değillerdir.

Tahmini azaltan temel faktörler: alcohol, flavanoids, malic_acid.

3. Karar Grafiği – Sınıf 2

Temel Değer: Yaklaşık 0.30.

Dinamik: Sample 0 ve Sample 4 (kırmızı/kesikli çizgiler) yüksekte (yaklaşık 0.9–1.0) sonlanır.

Tahmini artıran temel faktörler: od280/od315_of_diluted_wines, flavanoids, malic_acid.

Sample 1, 2, 3 (mavi/mor çizgiler) çok düşükte (yaklaşık -0.4) sonlanır, bu da Sınıf 2'ye ait olmadıklarına dair yüksek bir güven olduğunu gösterir.

Tahmini azaltan temel faktörler: alcohol, total_phenols, ash.

**Özet Çıkarım**

Karar grafikleri, SHAP Özellik Önem Grafiğinde (image_46f2c1.png) gördüğümüz sınıfa özgü özellik önemini doğrular:

alcohol ve proline, Sınıf 0 için güçlü öngörücülerdir.

od280/od315_of_diluted_wines ve flavanoids, Sınıf 2 için güçlü öngörücülerdir.

Bir sınıf için tahmini azaltan özellikler (örneğin, Sınıf 2 için alcohol), genellikle başka bir sınıf için tahmini artıran özelliklerdir (örneğin, Sınıf 0 için alcohol).



**Sonuçları yorumlayınız:**

- Hangi özellikler kararları belirledi?

- Optuna’nın bulduğu model hangi özelliklere daha duyarlı?

- MLP modellerindeki ortak ve farklı SHAP paternleri neler?

# Sonuçların Yorumlanması

**1. Hangi özellikler kararları belirledi?**

Modelin (şarap sınıflandırması) kararları, genel olarak SHAP Özellik Önem Grafiğine (image_46f2c1.png) göre aşağıdaki özellikler tarafından belirlenmiştir:

- En Önemli Özellikler (Temel Öngörücüler):

  1. alcohol (Alkol)

  2. ash (Kül)

  3. od280/od315_of_diluted_wines (Muhtemelen saflık veya fenolik bileşen göstergesi)

- Sınıfa Özgü Kararlar: Özelliklerin farklı sınıfları farklı şekilde etkilediğini belirtmek önemlidir (Karar Grafikleri/Decision Plots):

  1. Sınıf 0: Büyük ölçüde yüksek alcohol ve proline değerleri tarafından belirlenir.

  2. Sınıf 1: ash ve proline pozitif katkı sağlar.

  3. Sınıf 2: Büyük ölçüde yüksek od280/od315_of_diluted_wines ve flavanoids değerleri tarafından belirlenir.

**2. Optuna’nın bulduğu model hangi özelliklere daha duyarlı?**

Optimize edilmiş modelin sonuçlarını inceleyerek bu soruyu yanıtlayabiliriz:

- 'alcohol' Etkisinin Sınıflara Göre Karşılaştırılması:

  - Optimize edilmiş (Optuna) modelde, Sınıf 1 (1st alcohol) için 'alcohol'ün önemi +0.04'ten +0.05'e yükselmiştir. Bu, optimizasyondan sonra modelin Sınıf 1 ile ilgili kararlar alırken alcohol özelliğine karşı daha duyarlı hale geldiğini gösterir.

- Güçlü Etkileşim :

  - Optimize edilmiş model, alcohol ve malic_acid arasında olağanüstü güçlü bir etkileşim sergilemektedir. Bu, modelin sadece bu özelliklerin tek tek değerlerine değil, aynı zamanda bunların kombinasyonuna (etkileşimine) karşı da çok duyarlı hale geldiği anlamına gelir. Bir özelliğin etkisi, diğerinin değerine artık güçlü bir şekilde bağlıdır; bu durum orijinal modelde (image_47063d.png) gözlenmemiştir.

Sonuç: Optuna tarafından bulunan model, alkole (özellikle Sınıf 1 için) ve kritik olarak alcohol ile malic_acid'in ortak etkisine (etkileşimine) karşı daha duyarlı hale gelmiştir.

**3. MLP modellerindeki ortak ve farklı SHAP paternleri neler**

Yalnızca tek bir (veya bir kez optimize edilmiş) modelin grafiklerini sağladığınız için, farklı MLP modelleri arasındaki kalıpların doğrudan karşılaştırmasını yapamayız.

Ancak, sunulan verilere dayanarak, MLP ve diğer karmaşık modeller için tipik olan yorumlanabilirlik kalıplarını listeleyebiliriz:

-Özellik Kalıbı: Doğrusalsızlık ve Etkileşim

  - Durum: Ortak

  - Gözlemlenen Kanıt: Optuna modelinde alcohol ve malic_acid arasında güçlü SHAP Etkileşimi mevcuttur. MLP modelleri, doğal olarak doğrusal olmayan ilişkileri ve etkileşimleri yakalama yeteneğine sahiptir.

- Özellik Kalıbı: Sınıfa Özgü Önem

  - Durum: Ortak

  - Gözlemlenen Kanıt: Bir sınıf için tahmini keskin bir şekilde artıran (örneğin, Sınıf 0 için alcohol) özelliklerin, başka bir sınıf için tahmini keskin bir şekilde azalttığı (Karar Grafikleri/Decision Plots üzerinde görülür, örneğin Sınıf 2 için alcohol).

- Özellik Kalıbı: Karmaşık Karar Yolları

  - Durum: Ortak

  - Gözlemlenen Kanıt: Karar Grafikleri (Decision Plots), nihai tahmine giden yolun monoton olmadığını gösterir; özellik katkıları, nihai değere ulaşılana kadar genellikle birbirini iptal eder veya güçlendirir.
