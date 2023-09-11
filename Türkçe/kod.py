
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# Adım-1: Verinin Anlaşılması
#############################################

# persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd
pd.set_option("display.max_rows", None)
df = pd.read_csv('datasets/persona.csv')
df.head()
df.shape
df.info()

# Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")


# Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE": "sum"})

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="sum")

# SOURCE türlerine göre göre satış sayıları nedir?
df["SOURCE"].value_counts()

# Ülkelere göre PRICE ortalamaları nedir?
df.groupby(by=['COUNTRY']).agg({"PRICE": "mean"})

# SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby(by=['SOURCE']).agg({"PRICE": "mean"})

# COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})

#############################################
# Adım-2: Verinin Hazırlanması
#############################################

df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()
# Bu işlem sayesinde "COUNTRY", 'SOURCE', "SEX", "AGE" değişkenlerinin tüm kombinasyonları gözlenmiş olur.
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()
agg_df = agg_df.reset_index()


#AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.

# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

# AGE değişkeninin nerelerden bölüneceğini belirtelim:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Bölünen noktalara karşılık isimlendirmelerin ne olacağını ifade edelim:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_+']

# age'i bölelim:
agg_df["AGE_INTERVAL"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()


#############################################
# Adım-3: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# CUSTOMER_LEVEL adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile CUSTOMER_LEVEL değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.


agg_df['CUSTOMER_LEVEL'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_INTERVAL']].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.head()


# Gereksiz değişkenleri çıkaralım:
agg_df = agg_df[["CUSTOMER_LEVEL", "PRICE"]]
agg_df.head()

for i in agg_df["CUSTOMER_LEVEL"].values:
    print(i.split("_"))


# Amacımıza bir adım daha yaklaştık.
# Burada ufak bir problem var. Birçok aynı segment olacak.
# örneğin USA_ANDROID_MALE_0_18 segmentinden birçok sayıda olabilir.
# kontrol edelim:
agg_df["CUSTOMER_LEVEL"].value_counts()

# Bu sebeple segmentlere göre groupby yaptıktan sonra price ortalamalarını almalı ve segmentleri tekilleştirmeliyiz.
agg_df = agg_df.groupby("CUSTOMER_LEVEL").agg({"PRICE": "mean"})

# CUSTOMER_LEVEL index'te yer almaktadır. Bunu değişkene çevirelim.
agg_df = agg_df.reset_index()
agg_df.head()

# kontrol edelim. her bir persona'nın 1 tane olmasını bekleriz:
agg_df["CUSTOMER_LEVEL"].value_counts()
agg_df.head()


#############################################
# Segmentleri oluşturunuz.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})


#############################################
# Adım-4: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMER_LEVEL"] == new_user]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["CUSTOMER_LEVEL"] == new_user]
