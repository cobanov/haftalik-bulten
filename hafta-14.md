# Cobanov Haftalik Makale Bulteni - AI Week 14

<p align="center"><img src="assets/poster-week14.jpeg" alt="cloud ruins by mert cobnoav, 2022" width="650"><p>

Yoğun gelen mesajlardan sonra hepimizin derlenmiş, seçilmiş ve özellikle internet çöplüğünden ayıklanmış kaliteli içeriklere ihtiyacı olduğunu hissetmeye başladım. Zaten günlük rutinimde bu makaleleri sabahtan akşama kadar araştırıyorum, beğendiklerimi bir kenara not alıyordum. Sonra dedim ki, neden bu bulduklarımı sizinle paylaşmayayım? Hem siz de vakit kaybetmeden işinize yarayacak, kaliteli içeriklere ulaşmış olursunuz.

Yaklaşık olarak günde 3, haftada en az 20 makale not alıyorum (hesaplarıma göre yaklaşık 300 makaleye hızlı bir bakış attıktan sonra), bunlardan en can alıcılarını, en yararlı bulduklarımı özetleyip, ilgili anahtar taglerle birlikte sizinle paylaşmaya karar verdim.

Twitter’da @mertcobanov olarak bulabilirsiniz, düzenli içerik üretmeye 6 yıldır devam ediyorum. Umarım bu haftalık bülten, size zaman kazandırıp, iyi bir kısa yol sunacaktır.

**Cobanov**

---

<div style="page-break-after: always;"></div>

## 1. How to Generate Instruction Datasets from Any Documents for LLM Fine-Tuning

### Source

> <https://towardsdatascience.com/how-to-generate-instruction-datasets-from-any-documents-for-llm-fine-tuning-abb319a05d91>

### Tags

- Large Language Models
- Programming and Compiling
- Application Development

### Summary
*Click to expand*
<details><summary>
Makale, büyük dil modellerini (LLM) belirli alanlara özgü bilgilerle uyumlu hale getirmenin zorluklarından bahsediyor, çünkü bu modeller genellikle bu bilgilere sahip olmuyor. 
</summary>
<br>
Yazar, kullanıcının özel verilerine uyum sağlamak için yapay talimat veri kümeleri üretebilen, açık kaynaklı ve hafif bir kütüphane olan Bonito adında, maliyeti düşük bir çözüm öneriyor. Bonito, özel görevler oluşturmak için tasarlanmış ve LLM'leri daha iyi hale getirmek için gerekli veri kümelerini oluşturmak için kullanılabilir. Yazar, istenilen sonuçlara ulaşmak için net talimatların verilmesinin önemini vurguluyor, çünkü talimatlar tartışmayı yönlendirip, ilgili ve kullanıcı beklentileriyle uyumlu olmasını sağlıyor.

Makale, Bonito kullanmanın faydalarını, açık kaynak olması, hafif tasarımı ve yapay veri kümelerini ekonomik bir şekilde üretebilme yeteneğini öne çıkarıyor. Yazar ayrıca, LLM'leri kullanarak talimat veri kümeleri oluşturmanın pahalı ve zaman alıcı olabileceğini, bu yüzden Bonito'nun uygun bir alternatif olduğunu belirtiyor. Ek olarak, makale, talimatların ne olduğunu ve istenilen sonuçlara ulaşmak için tartışmayı nasıl yönlendirdiğini anlamanın önemine değiniyor.

Genel olarak, makale, LLM'leri daha iyi hale getirmek için talimat veri kümeleri üretme potansiyeli olan Bonito hakkında detaylı bir bakış sunuyor ve net talimatların istenilen sonuçlara ulaşmak için önemini vurguluyor.

</details>

---

<div style="page-break-after: always;"></div>

## 2. Build Your Own RAG and Run It Locally on Your Laptop: ColBERT + DSPy + Streamlit

### Source

> <https://medium.com/towards-data-science/rag-on-your-laptop-colbert-dspy-streamlit-c206ea92188f>

### Tags

- Large Language Models
- Programming and Compiling
- Application Development

### Summary
*Click to expand*
<details>
<summary>
Shuyi Yang tarafından yazılan "Kendi RAG'inizi Yapın ve Yerel Olarak Laptop'unuzda Çalıştırın: ColBERT + DSPy + Streamlit" başlıklı makale, başlangıç seviyesindeki kişilere basit bir Alın Yazılarını Geliştirme (Retrieval Augmented Generation - RAG) sistemi kurma ve yerel olarak çalıştırma adımlarını anlatır.
</summary><br>

Yazar, yazma görevlerinde yardımcı olan akıllı bir asistan olan RAG kavramını ve bunun ColBERT, DSPy ve Streamlit kullanılarak bir laptop üzerinde nasıl kurulup çalıştırılabileceğini açıklar.

Ders, RAG'in temellerini ve çeşitli endüstrilerdeki potansiyel uygulamalarını anlamanın önemini açıklayarak başlar. Yazar daha sonra RAG sistemi kurmak için gerekli olan ColBERT, DSPy ve Streamlit gibi araçlar ve kütüphanelerin bir listesini sunar.

Makalenin bir sonraki bölümü, gerekli araçların ve kütüphanelerin kurulumu ve ayarlanmasına odaklanır. Yazar, ColBERT, DSPy ve Streamlit'in nasıl kurulacağı ve optimal performans için nasıl yapılandırılacağı konusunda detaylı talimatlar sağlar.

Araçlar ve kütüphaneler kurulduktan sonra, yazar ColBERT ve DSPy kullanarak RAG sisteminin nasıl kurulacağını açıklar. Bu, kullanıcıların isteklerini girebilecekleri ve üretilen yanıtları alabilecekleri basit bir metin tabanlı arayüz oluşturmayı içerir. Yazar, bu özellikleri nasıl uygulayacaklarına dair kod örnekleri ve açıklamalar sağlar.

Makalenin son bölümü, RAG sisteminin Streamlit kullanılarak kullanıcının laptop'unda yerel olarak nasıl dağıtılacağını kapsar. Yazar, herhangi bir harici bağımlılık veya sunucu gerektirmeyen, yerel olarak çalıştırılabilecek bir Streamlit uygulaması oluşturmayı açıklar.

Tüm ders boyunca, yazar başlangıç seviyesindekilere yönelik yararlı ipuçları ve püf noktaları sunar, örneğin yaygın hataların nasıl ele alınacağı ve RAG sisteminin daha iyi performans için nasıl optimize edileceği gibi. Makale, anahtar noktaların özetini sunarak ve okuyucuları sağlanan kod örneklerini kullanarak kendi RAG sistemlerini kurmaya teşvik ederek sona erer.

Genel olarak, makale, başlangıç seviyesindeki kişilere laptop'larında yerel olarak basit bir RAG sistemi kurup çalıştırmak isteyenler için kapsamlı ve erişilebilir bir ders sunar. Yazarın açık ve öz dil kullanımı, detaylı kod örnekleriyle birlikte, okuyucuların dersi takip etmesini ve sunulan kavramları anlamasını kolaylaştırır.

</details>

---

<div style="page-break-after: always;"></div>

## 3. Vector Embeddings Explained for Developers!

### Source

> <https://medium.com/gitconnected/vector-embeddings-explained-for-developers-6bd9800d3635>

### Tags

- Large Language Models
- Programming and Compiling
- Application Development

### Summary
*Click to expand*
<details>

<summary> 
Bu makalede, vektör gömülülerini ve makine öğrenimi ile veri bilimindeki önemlerini tartıştık. 
</summary>
<br>

Ayrıca, OpenAI, Cohere ve HuggingFace gibi popüler gömülü modelleri kullanarak vektör gömülülerini nasıl oluşturabileceğimizi ve bunları SingleStore veritabanında nasıl saklayabileceğimizi keşfettik.

Makalede ele alınan ana noktalar şunlardır:

1. Vektör gömülüler, metin, görüntü ve diğer veri formlarının etkili bir şekilde işlenmesi ve analiz edilmesini sağlayarak makine öğrenimi ve veri biliminde hayati bir bileşendir.
2. Veri noktalarını, yüksek boyutlu bir uzayda vektörler olarak temsil ederek, boyut indirgeme yoluyla daha düşük boyutlu bir uzaya indirgeyerek ve benzerliğe göre kümeleme veya sınıflandırma yaparak vektör gömülülerinin nasıl çalıştığını açıkladık.
3. Farklı özellikler ve yetenekler sunan OpenAI, Cohere ve HuggingFace gibi popüler gömülü modellerden bahsettik.
4. Bu modelleri kullanarak vektör gömülülerini nasıl oluşturacağımızı gösterirken, modellerle etkileşim kurmak için Python kütüphaneleri ve API anahtarlarının nasıl kullanılacağını gösterdik.
5. Oluşturulduktan sonra, vektör gömülülerini bir SingleStore veritabanında nasıl saklayacağımızı gösterdik. SQL Düzenleyici aracılığıyla bunu yapmanın kolaylığına ve indeksli yaklaşık-en-yakın-komşu (ANN) araması yapma yeteneğine dikkat çektik.
6. Son olarak, vektör gömülülerinin potansiyelinden ve AI'da, sohbet robotlarından içerik öneri sistemlerine kadar uygulamalarından bahsettik.

Geliştiriciler, vektör gömülülerinin nasıl çalıştığını ve nasıl oluşturulup saklanacağını anlayarak, daha sofistike makine öğrenimi modelleri ve uygulamaları geliştirmek için bu güçlü araçları kullanabilirler.

</details>

---

<div style="page-break-after: always;"></div>

## 4. Intro to DSPy: Goodbye Prompting, Hello Programming!

### Source

> <https://medium.com/towards-data-science/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9>

### Tags

- DSPy Framework
- Large Language Models
- Programming and Compiling
- Application Development
- Retrieval-Augmented Generation

### Summary
*Click to expand*
<details> <summary> 
Bu makalede, Leonie Monigatti, büyük dil modelleri (LLM) tabanlı uygulamalar geliştirmek için DSPy çerçevesinin kullanımının avantajlarını inceliyor. 
</summary> <br>
Geleneksel yöntemlerle yapılan LLM tabanlı uygulamaların genellikle karmaşık ve kırılgan olduğunu, bu durumun da dağıtım sırasında sorunlara neden olduğunu açıklıyor. Bu problemleri çözmek için DSPy, sorular sorma yerine programlama ve derlemeyi önererek daha sağlam bir çözüm sunuyor.

Makale, DSPy'nin ne olduğunu ve diğer çerçevelerden nasıl farklı olduğunu tanımlayarak başlıyor. Ardından, LLM tabanlı uygulamalar geliştirirken karşılaşılan güncel zorluklar, özellikle de kırılganlık problemi üzerine bir genel bakış sunuyor. Bu sorunu çözmek için DSPy, soru sorma yerine programlama ve derleme içeren yeni bir yaklaşım sunar. Bu yaklaşım, işlem akışı üzerinde daha fazla kontrol sağlar ve dağıtım sırasında hata riskini azaltır.

Yazar, Weaviate veri setini kullanarak Retrieval-Augmented Generation (RAG) için baştan sona bir DSPy iş akışı örneği veriyor. Bu iş akışı, DSPy'nin karmaşık LLM tabanlı uygulamaları daha kolay ve güvenilir bir şekilde nasıl oluşturabileceğini gösteriyor.

Makale boyunca Monigatti, DSPy'de programlamanın ve derlemenin önemini vurguluyor ve bu yaklaşımın geleneksel soru sorma yöntemlerine göre avantajlarını belirtiyor. Ayrıca, DSPy ile başlamak için pratik ipuçları sunuyor ve daha fazla öğrenim için kaynaklar öneriyor.

Sonuç olarak, makale, DSPy'ye ve LLM tabanlı uygulamaları geliştirme şeklimizi devrim niteliğinde değiştirebilecek potansiyeline geniş bir giriş sağlıyor. Soru sorma yerine programlama ve derlemeyi tercih ederek, DSPy, karmaşık uygulamalar oluşturmak için daha sağlam ve güvenilir bir çözüm sunuyor.

</details>

---

<div style="page-break-after: always;"></div>
