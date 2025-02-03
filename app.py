import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Yardımcı fonksiyonlar
def calculate_technical_indicators(df):
    # Temel hesaplamalar
    df['Daily_Return'] = df['close'].pct_change() * 100
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    
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
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2*df['close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2*df['close'].rolling(window=20).std()
    
    return df

def calculate_risk_metrics(returns):
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    return {
        'Volatilite': volatility,
        'Sharpe Oranı': sharpe_ratio,
        'VaR (95%)': var_95,
        'VaR (99%)': var_99,
        'Çarpıklık': skewness,
        'Basıklık': kurtosis
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
    
    # Durağanlık testi
    adf_result = adfuller(df['close'].dropna())
    
    # Normallik testi
    stat, p_value = stats.normaltest(returns)
    
    # Otokorelasyon
    autocorr = returns.autocorr()
    
    # ARIMA modeli
    try:
        model = ARIMA(df['close'], order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=1)[0]
    except:
        forecast = None
    
    return {
        'Normallik Testi p-değeri': p_value,
        'ADF Test İstatistiği': adf_result[0],
        'ADF p-değeri': adf_result[1],
        'Otokorelasyon': autocorr,
        'ARIMA Tahmini': forecast
    }

def predict_next_day_values(df):
    features = ['open', 'high', 'low', 'close', 'Volume', 'Daily_Return', 'Volatility', 'RSI']
    target_cols = ['close', 'high', 'low']
    predictions = {}
    
    for target in target_cols:
        # Veri hazırlama
        df[f'Next_{target}'] = df[target].shift(-1)
        df_pred = df.dropna()
        
        X = df_pred[features]
        y = df_pred[f'Next_{target}']
        
        # Model eğitimi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_scaled, y)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)
        
        # Son veriyle tahmin
        last_data = df[features].iloc[-1:].copy()
        last_data_scaled = scaler.transform(last_data)
        
        gb_pred = gb_model.predict(last_data_scaled)[0]
        rf_pred = rf_model.predict(last_data_scaled)[0]
        
        # Ensemble tahmin
        predictions[target] = (gb_pred + rf_pred) / 2
    
    return predictions

# Streamlit sayfa yapılandırması
st.set_page_config(page_title="Hisse Senedi Analizi", layout="wide")

# Ana başlık
st.title("Detaylı Hisse Senedi Analizi Raporu")

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
        
        # Teknik göstergeleri hesapla
        df = calculate_technical_indicators(df)

        # 1. ÖZET BİLGİLER
        st.header("1. ÖZET BİLGİLER")
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

        # 2. TEKNİK ANALİZ
        st.header("2. TEKNİK ANALİZ")
        
        # 2.1 Fiyat ve Hacim Grafiği
        st.subheader("2.1 Fiyat ve Hacim Analizi")
        fig = plt.figure(figsize=(12, 8))
        
        # Üst grafik - Fiyat ve Hareketli Ortalamalar
        ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
        ax1.plot(df.index, df['close'], label='Kapanış', linewidth=2)
        ax1.plot(df.index, df['MA20'], label='20 Günlük HO', alpha=0.7)
        ax1.plot(df.index, df['MA50'], label='50 Günlük HO', alpha=0.7)
        ax1.plot(df.index, df['MA200'], label='200 Günlük HO', alpha=0.7)
        ax1.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.1)
        ax1.set_title(f"{hisse_adi} Fiyat Grafiği")
        ax1.legend()
        ax1.grid(True)
        
        # Alt grafik - Hacim
        ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1)
        ax2.bar(df.index, df['Volume'], label='Hacim')
        ax2.set_title("Hacim")
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 2.2 Teknik Göstergeler
        st.subheader("2.2 Teknik Göstergeler")
        col1, col2 = st.columns(2)
        
        with col1:
            # MACD
            fig_macd = plt.figure(figsize=(10, 4))
            plt.plot(df.index, df['MACD'], label='MACD')
            plt.plot(df.index, df['Signal_Line'], label='Sinyal')
            plt.fill_between(df.index, df['MACD'] - df['Signal_Line'], color='gray', alpha=0.3)
            plt.title('MACD Göstergesi')
            plt.legend()
            plt.grid(True)
            st.pyplot(fig_macd)
            
        with col2:
            # RSI
            fig_rsi = plt.figure(figsize=(10, 4))
            plt.plot(df.index, df['RSI'])
            plt.axhline(y=70, color='r', linestyle='--')
            plt.axhline(y=30, color='g', linestyle='--')
            plt.fill_between(df.index, df['RSI'], 70, where=(df['RSI']>=70), color='red', alpha=0.3)
            plt.fill_between(df.index, df['RSI'], 30, where=(df['RSI']<=30), color='green', alpha=0.3)
            plt.title('RSI Göstergesi')
            plt.grid(True)
            st.pyplot(fig_rsi)

        # 3. İSTATİSTİKSEL ANALİZ
        st.header("3. İSTATİSTİKSEL ANALİZ")
        
        # 3.1 Tanımlayıcı İstatistikler
        st.subheader("3.1 Tanımlayıcı İstatistikler")
        desc_stats = df[['close', 'Daily_Return', 'Volume']].describe()
        st.dataframe(desc_stats)
        
        # 3.2 Risk Metrikleri
        st.subheader("3.2 Risk Analizi")
        returns = df['Daily_Return'].dropna() / 100  # Yüzdelik değeri ondalığa çevir
        risk_metrics = calculate_risk_metrics(returns)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Yıllık Volatilite", f"%{risk_metrics['Volatilite']*100:.2f}")
            st.metric("Çarpıklık", f"{risk_metrics['Çarpıklık']:.2f}")
        with col2:
            st.metric("Sharpe Oranı", f"{risk_metrics['Sharpe Oranı']:.2f}")
            st.metric("Basıklık", f"{risk_metrics['Basıklık']:.2f}")
        with col3:
            st.metric("VaR (95%)", f"%{-risk_metrics['VaR (95%)']*100:.2f}")
            st.metric("VaR (99%)", f"%{-risk_metrics['VaR (99%)']*100:.2f}")
            
        # Getiri Dağılımı
        fig_dist = plt.figure(figsize=(10, 6))
        sns.histplot(returns, kde=True)
        plt.title("Getiri Dağılımı")
        st.pyplot(fig_dist)

        # 3.3 İstatistiksel Testler
        st.subheader("3.3 İstatistiksel Testler")
        stats_results = perform_statistical_analysis(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normallik Testi p-değeri", f"{stats_results['Normallik Testi p-değeri']:.4f}")
            st.metric("Otokorelasyon", f"{stats_results['Otokorelasyon']:.4f}")
        with col2:
            st.metric("ADF Test İstatistiği", f"{stats_results['ADF Test İstatistiği']:.4f}")
            st.metric("ADF p-değeri", f"{stats_results['ADF p-değeri']:.4f}")

        # 4. FİBONACCİ SEVİYELERİ
        st.header("4. FİBONACCİ SEVİYELERİ")
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
            plt.grid(True)
            st.pyplot(fig_fib)

        # 5. GELECEK TAHMİNLERİ
        st.header("5. GELECEK TAHMİNLERİ")
        predictions = predict_next_day_values(df)
        
        col1, col2, col3 = st.columns(3)
        current_close = df['close'].iloc[-1]
        
        with col1:
            pred_change = ((predictions['close'] / current_close) - 1) * 100
            st.metric("Tahmini Kapanış", 
                     f"₺{predictions['close']:.2f}",
                     f"%{pred_change:.2f}")
        
        with col2:
            st.metric("Tahmini En Yüksek", 
                     f"₺{predictions['high']:.2f}")
        
        with col3:
            st.metric("Tahmini En Düşük", 
                     f"₺{predictions['low']:.2f}")
            
        if stats_results['ARIMA Tahmini'] is not None:
            st.metric("ARIMA Model Tahmini", 
                     f"₺{stats_results['ARIMA Tahmini']:.2f}")

        # 6. KORELASYON ANALİZİ
        st.header("6. KORELASYON ANALİZİ")
        corr_matrix = df[['open', 'high', 'low', 'close', 'Volume', 'Daily_Return', 'RSI']].corr()
        
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Korelasyon Matrisi')
        st.pyplot(fig_corr)

else:
    st.info(f"Lütfen önce hisse adını girin ve ardından {hisse_adi if hisse_adi else 'hisse adı'} ile başlayan CSV dosyasını yükleyin.")
