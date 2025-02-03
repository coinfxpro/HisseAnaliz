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

# Yardımcı fonksiyonlar
def calculate_risk_metrics(returns):
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    return {
        'Volatilite': volatility,
        'Sharpe Oranı': sharpe_ratio,
        'VaR (95%)': var_95,
        'VaR (99%)': var_99
    }

def calculate_fibonacci_levels(high, low):
    diff = high - low
    levels = {
        'Seviye 0.0': low,
        'Seviye 0.236': low + 0.236 * diff,
        'Seviye 0.382': low + 0.382 * diff,
        'Seviye 0.5': low + 0.5 * diff,
        'Seviye 0.618': low + 0.618 * diff,
        'Seviye 0.786': low + 0.786 * diff,
        'Seviye 1.0': high
    }
    return levels

def perform_statistical_analysis(df):
    # Günlük getiriler
    returns = df['close'].pct_change().dropna()
    
    # Normallik testi
    stat, p_value = stats.normaltest(returns)
    
    # Durağanlık testi
    adf_result = adfuller(df['close'].dropna())
    
    return {
        'Normallik Testi p-değeri': p_value,
        'ADF Test İstatistiği': adf_result[0],
        'ADF p-değeri': adf_result[1]
    }

def predict_next_day_values(df):
    features = ['open', 'high', 'low', 'close', 'Volume']
    scaler = StandardScaler()
    
    # Tahmin için veri hazırlama
    df['Next_Close'] = df['close'].shift(-1)
    df_pred = df.dropna()
    
    X = df_pred[features]
    y = df_pred['Next_Close']
    
    # Model eğitimi
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Son gün verisiyle tahmin
    last_data = df[features].iloc[-1:].copy()
    last_data_scaled = scaler.transform(last_data)
    next_day_pred = model.predict(last_data_scaled)[0]
    
    return next_day_pred

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
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Ana metrikler
        st.header("Temel Metrikler")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Son Kapanış", f"₺{df['close'].iloc[-1]:.2f}")
        with col2:
            daily_return = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
            st.metric("Günlük Değişim", f"%{daily_return:.2f}")
        with col3:
            volume_change = ((df['Volume'].iloc[-1] / df['Volume'].iloc[-2]) - 1) * 100
            st.metric("Hacim Değişimi", f"%{volume_change:.2f}")
        with col4:
            st.metric("Günlük İşlem Hacmi", f"₺{df['Volume'].iloc[-1]:,.0f}")

        # Teknik Analiz
        st.header("Teknik Analiz")
        
        # Fiyat Grafiği
        fig_price = plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Kapanış Fiyatı')
        plt.title(f"{hisse_adi} Hisse Senedi Fiyat Grafiği")
        plt.xlabel("Tarih")
        plt.ylabel("Fiyat (₺)")
        plt.legend()
        st.pyplot(fig_price)
        
        # MACD ve RSI
        col1, col2 = st.columns(2)
        
        with col1:
            # MACD hesaplama
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            fig_macd = plt.figure(figsize=(10, 4))
            plt.plot(df.index, macd, label='MACD')
            plt.plot(df.index, signal, label='Sinyal')
            plt.title('MACD Göstergesi')
            plt.legend()
            st.pyplot(fig_macd)
            
        with col2:
            # RSI hesaplama
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig_rsi = plt.figure(figsize=(10, 4))
            plt.plot(df.index, rsi)
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
            plt.title('RSI Göstergesi')
            st.pyplot(fig_rsi)

        # Risk Metrikleri
        st.header("Risk Analizi")
        returns = df['close'].pct_change().dropna()
        risk_metrics = calculate_risk_metrics(returns)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volatilite", f"%{risk_metrics['Volatilite']*100:.2f}")
        with col2:
            st.metric("Sharpe Oranı", f"{risk_metrics['Sharpe Oranı']:.2f}")
        with col3:
            st.metric("VaR (95%)", f"%{risk_metrics['VaR (95%)']*100:.2f}")
        with col4:
            st.metric("VaR (99%)", f"%{risk_metrics['VaR (99%)']*100:.2f}")

        # Fibonacci Seviyeleri
        st.header("Fibonacci Seviyeleri")
        fib_levels = calculate_fibonacci_levels(df['high'].max(), df['low'].min())
        
        col1, col2 = st.columns(2)
        with col1:
            for level, value in fib_levels.items():
                st.write(f"{level}: ₺{value:.2f}")
                
        with col2:
            fig_fib = plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['close'])
            for value in fib_levels.values():
                plt.axhline(y=value, color='r', linestyle='--', alpha=0.3)
            plt.title('Fibonacci Seviyeleri')
            st.pyplot(fig_fib)

        # İstatistiksel Analiz
        st.header("İstatistiksel Analiz")
        stats_results = perform_statistical_analysis(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Normallik Testi p-değeri", f"{stats_results['Normallik Testi p-değeri']:.4f}")
        with col2:
            st.metric("ADF Test İstatistiği", f"{stats_results['ADF Test İstatistiği']:.4f}")
        with col3:
            st.metric("ADF p-değeri", f"{stats_results['ADF p-değeri']:.4f}")

        # Gelecek Tahminleri
        st.header("Gelecek Tahminleri")
        next_day_price = predict_next_day_values(df)
        current_price = df['close'].iloc[-1]
        price_change = ((next_day_price / current_price) - 1) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mevcut Fiyat", f"₺{current_price:.2f}")
        with col2:
            st.metric("Tahmini Sonraki Gün Fiyatı", 
                     f"₺{next_day_price:.2f}",
                     f"%{price_change:.2f}")

        # Korelasyon Analizi
        st.header("Korelasyon Analizi")
        corr_matrix = df[['open', 'high', 'low', 'close', 'Volume']].corr()
        
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Korelasyon Matrisi')
        st.pyplot(fig_corr)

else:
    st.info(f"Lütfen önce hisse adını girin ve ardından {hisse_adi if hisse_adi else 'hisse adı'} ile başlayan CSV dosyasını yükleyin.")
