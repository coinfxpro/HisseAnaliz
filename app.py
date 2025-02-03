import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Streamlit sayfa yapılandırması
st.set_page_config(page_title="Hisse Senedi Analizi", layout="wide")

# Ana başlık
st.title("Hisse Senedi Analizi Uygulaması")

# Kullanıcıdan hisse adını al
hisse_adi = st.text_input("Analiz edilecek hisse adını girin (örn: SISE):", "").upper()

# Dosya yükleme alanı
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type=['csv'])

if uploaded_file is not None:
    # Dosya adını kontrol et
    if not uploaded_file.name.startswith(hisse_adi):
        st.error(f"Lütfen {hisse_adi} ile başlayan bir CSV dosyası yükleyin!")
    else:
        # CSV dosyasını oku
        df = pd.read_csv(uploaded_file)
        
        # Tarih sütununu düzenle
        df['time'] = pd.to_datetime(df['time'], unit='s')  # Unix timestamp'i datetime'a çevir
        df.set_index('time', inplace=True)

        # Temel istatistikler
        st.header("Temel İstatistikler")
        st.dataframe(df.describe())

        # Grafik
        st.header("Hisse Fiyat Grafiği")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['close'], label='Kapanış Fiyatı')
        ax.set_title(f"{hisse_adi} Hisse Senedi Fiyat Grafiği")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Fiyat")
        ax.legend()
        st.pyplot(fig)

        # Teknik Analiz
        st.header("Teknik Analiz")
        
        # RSI hesaplama
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD hesaplama
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Teknik göstergeleri göster
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RSI Göstergesi")
            fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
            ax_rsi.plot(df.index, df['RSI'])
            ax_rsi.axhline(y=70, color='r', linestyle='--')
            ax_rsi.axhline(y=30, color='g', linestyle='--')
            ax_rsi.set_title("RSI")
            st.pyplot(fig_rsi)

        with col2:
            st.subheader("MACD Göstergesi")
            fig_macd, ax_macd = plt.subplots(figsize=(10, 4))
            ax_macd.plot(df.index, df['MACD'], label='MACD')
            ax_macd.plot(df.index, df['Signal Line'], label='Signal Line')
            ax_macd.legend()
            ax_macd.set_title("MACD")
            st.pyplot(fig_macd)

        # Gelecek tahminleri
        st.header("Gelecek Tahminleri")
        
        features = ['open', 'high', 'low', 'close', 'volume']
        
        # Son 30 günlük tahminler
        df['Next_Close'] = df['close'].shift(-1)
        df_pred = df.dropna()
        X = df_pred[features]
        y = df_pred['Next_Close']
        
        # Model eğitimi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Son verilerle tahmin
        last_data = df[features].iloc[-1:]
        last_data_scaled = scaler.transform(last_data)
        next_day_pred = model.predict(last_data_scaled)[0]
        
        st.subheader("Yarının Tahmini Kapanış Fiyatı")
        st.write(f"Tahmin edilen fiyat: {next_day_pred:.2f} TL")
        
        # Tahmin doğruluğu
        y_pred = model.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        st.write(f"Model RMSE: {rmse:.2f} TL")

else:
    st.info(f"Lütfen önce hisse adını girin ve ardından {hisse_adi if hisse_adi else 'hisse adı'} ile başlayan CSV dosyasını yükleyin.")
