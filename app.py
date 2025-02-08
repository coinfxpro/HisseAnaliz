import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
import yfinance as yf
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Fonksiyon tanımlamaları
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

def calculate_rsi(prices, period=14):
    """RSI (Göreceli Güç Endeksi) hesaplar"""
    # Fiyat değişimlerini hesapla
    delta = prices.diff()
    
    # Pozitif ve negatif değişimleri ayır
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # RS hesapla
    rs = gain / loss
    
    # RSI hesapla
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_risk_metrics(df):
    returns = df['Daily_Return'].dropna() / 100  # Yüzdeyi ondalığa çevir
    
    # Volatilite (yıllık)
    volatility = returns.std() * np.sqrt(252)
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Sharpe Ratio (Risk-free rate olarak %5 varsayıyoruz)
    risk_free_rate = 0.05
    excess_returns = returns - risk_free_rate/252  # Günlük risk-free rate
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns/rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Beta (Piyasa verisi olmadığı için varsayılan 1)
    beta = 1.0
    
    return {
        'Volatilite': volatility,
        'VaR_95': var_95,
        'VaR_99': var_99,
        'Sharpe Oranı': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Beta': beta
    }

def perform_statistical_analysis(df):
    # Durağanlık testi (ADF)
    adf_result = adfuller(df['close'].dropna())
    
    # Normallik testi
    returns = df['Daily_Return'].dropna()
    stat, p_value = stats.normaltest(returns)
    
    # Otokorelasyon
    autocorr = returns.autocorr()
    
    # Çarpıklık ve Basıklık
    skew = returns.skew()
    kurtosis = returns.kurtosis()
    
    # ARIMA modeli
    try:
        model = ARIMA(df['close'], order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=1)[0]
    except:
        forecast = None
    
    # Mevsimsellik analizi
    try:
        seasonal_result = seasonal_decompose(df['close'], period=30)
        seasonality = seasonal_result.seasonal[-1]
    except:
        seasonality = None
    
    return {
        'ADF p-değeri': adf_result[1],
        'Normallik p-değeri': p_value,
        'Otokorelasyon': autocorr,
        'Çarpıklık': skew,
        'Basıklık': kurtosis,
        'ARIMA Tahmini': forecast,
        'Mevsimsellik': seasonality
    }

def predict_next_day_values(df):
    """Gelecek gün tahminlerini hesaplar"""
    try:
        # Feature'ları hazırla
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['close'])
        
        # NaN değerleri temizle
        df = df.dropna()
        
        # Feature'ları ve hedef değişkeni ayarla
        features = ['close', 'volume', 'MA5', 'MA20', 'RSI']  # Volume büyük harfle
        X = df[features].values
        y_close = df['close'].values
        
        # Veriyi ölçeklendir
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train = X_scaled[:-1]  # Son günü test için ayır
        X_test = X_scaled[-1:]   # Son gün
        y_train = y_close[:-1]
        
        # Model eğitimi
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Tahmin
        next_day_pred = model.predict(X_test)[0]
        
        # Tahmin sonuçlarını hazırla
        predictions = {
            'Tahmin Edilen Kapanış': next_day_pred,
            'Son Kapanış': df['close'].iloc[-1],
            'Değişim': (next_day_pred - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100
        }
        
        return predictions
    except Exception as e:
        st.error(f"Tahmin hesaplanırken bir hata oluştu: {str(e)}")
        # Hata durumunda varsayılan tahminler
        return {
            'Tahmin Edilen Kapanış': df['close'].iloc[-1] * 1.001,  # Çok küçük bir artış
            'Son Kapanış': df['close'].iloc[-1],
            'Değişim': 0.1
        }

def generate_alternative_scenarios(df, predictions):
    """Alternatif senaryolar oluşturur"""
    try:
        # Hacim durumu analizi
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        
        volume_status = "Düşük Hacim" if volume_change < -25 else "Yüksek Hacim" if volume_change > 25 else "Normal Hacim"
        
        # Yüksek hacim senaryosu
        yuksek_hacim = {
            'Tahmin Edilen Kapanış': predictions['Tahmin Edilen Kapanış'] * 1.02,  # %2 daha yüksek
            'Son Kapanış': predictions['Son Kapanış'],
            'Değişim': ((predictions['Tahmin Edilen Kapanış'] * 1.02 - predictions['Son Kapanış']) / predictions['Son Kapanış']) * 100
        }
        
        # Düşük hacim senaryosu
        dusuk_hacim = {
            'Tahmin Edilen Kapanış': predictions['Tahmin Edilen Kapanış'] * 0.98,  # %2 daha düşük
            'Son Kapanış': predictions['Son Kapanış'],
            'Değişim': ((predictions['Tahmin Edilen Kapanış'] * 0.98 - predictions['Son Kapanış']) / predictions['Son Kapanış']) * 100
        }
        
        return {
            'Yüksek_Hacim': yuksek_hacim,
            'Düşük_Hacim': dusuk_hacim,
            'Hacim_Durumu': {
                'Durum': volume_status,
                'Değişim': volume_change
            }
        }
    except Exception as e:
        st.error(f"Senaryo hesaplanırken bir hata oluştu: {str(e)}")
        # Hata durumunda varsayılan senaryolar
        return {
            'Yüksek_Hacim': {
                'Tahmin Edilen Kapanış': predictions['Son Kapanış'] * 1.02,
                'Son Kapanış': predictions['Son Kapanış'],
                'Değişim': 2.0
            },
            'Düşük_Hacim': {
                'Tahmin Edilen Kapanış': predictions['Son Kapanış'] * 0.98,
                'Son Kapanış': predictions['Son Kapanış'],
                'Değişim': -2.0
            },
            'Hacim_Durumu': {
                'Durum': 'Normal Hacim',
                'Değişim': 0.0
            }
        }

def analyze_volume_scenarios(df, predictions):
    """Hacim senaryolarını analiz eder"""
    try:
        # Hacim durumu analizi
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        
        # Hacim durumu belirleme
        if volume_change < -25:
            volume_status = "Düşük Hacim"
        elif volume_change > 25:
            volume_status = "Yüksek Hacim"
        else:
            volume_status = "Normal Hacim"
        
        return {
            'Durum': volume_status,
            'Değişim': volume_change
        }
    except Exception as e:
        st.error(f"Hacim analizi hesaplanırken bir hata oluştu: {str(e)}")
        return {
            'Durum': "Normal Hacim",
            'Değişim': 0.0
        }

def generate_analysis_summary(df, predictions, risk_metrics, stats_results):
    """Analiz özetini ve yorumları oluşturur."""
    
    # Genel trend analizi
    current_trend = "YÜKSELİŞ" if df['close'].iloc[-1] > df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else \
                   "YÜKSELİŞ" if df['close'].iloc[-1] > df['MA20'].iloc[-1] else \
                   "DÜŞÜŞ" if df['close'].iloc[-1] < df['MA20'].iloc[-1] < df['MA50'].iloc[-1] else \
                   "DÜŞÜŞ" if df['close'].iloc[-1] < df['MA20'].iloc[-1] else "YATAY"
    
    # RSI durumu
    rsi_status = "AŞIRI ALIM 🔴" if df['RSI'].iloc[-1] > 70 else \
                 "AŞIRI SATIM 🟢" if df['RSI'].iloc[-1] < 30 else \
                 "NÖTR ⚪"
    
    # Volatilite durumu
    volatility_status = "YÜKSEK ⚠️" if risk_metrics['Volatilite'] > 0.3 else \
                       "NORMAL ✅" if risk_metrics['Volatilite'] > 0.15 else \
                       "DÜŞÜK 💤"
    
    # Durağanlık durumu
    stationarity = "DURAĞAN ✅" if stats_results['ADF p-değeri'] < 0.05 else "DURAĞAN DEĞİL ⚠️"
    
    # Hareketli ortalamalar
    ma_status = {
        "MA20": f"{'⬆️' if df['close'].iloc[-1] > df['MA20'].iloc[-1] else '⬇️'} {df['MA20'].iloc[-1]:.2f}",
        "MA50": f"{'⬆️' if df['close'].iloc[-1] > df['MA50'].iloc[-1] else '⬇️'} {df['MA50'].iloc[-1]:.2f}",
        "MA200": f"{'⬆️' if df['close'].iloc[-1] > df['MA200'].iloc[-1] else '⬇️'} {df['MA200'].iloc[-1]:.2f}"
    }
    
    # MACD durumu
    macd_signal = "AL 🟢" if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] else "SAT 🔴"
    
    # Bollinger durumu
    if df['close'].iloc[-1] > df['BB_upper'].iloc[-1]:
        bb_status = "AŞIRI ALINIM ⚠️"
    elif df['close'].iloc[-1] < df['BB_lower'].iloc[-1]:
        bb_status = "AŞIRI SATIM 🔔"
    else:
        bb_status = "NORMAL ✅"
    
    # Hacim analizi
    volume_avg = df['volume'].mean()
    current_volume = df['volume'].iloc[-1]
    volume_status = "YÜKSEK 💪" if current_volume > volume_avg * 1.5 else \
                   "DÜŞÜK 👎" if current_volume < volume_avg * 0.5 else \
                   "NORMAL 👍"
    
    # Risk durumu
    risk_status = "YÜKSEK RİSK ⚠️" if risk_metrics['Volatilite'] > 0.3 or risk_metrics['VaR_95'] < -0.03 else \
                 "ORTA RİSK ⚡" if risk_metrics['Volatilite'] > 0.2 or risk_metrics['VaR_95'] < -0.02 else \
                 "DÜŞÜK RİSK ✅"
    
    return {
        'Genel Trend': f"{current_trend} {'📈' if current_trend == 'YÜKSELİŞ' else '📉' if current_trend == 'DÜŞÜŞ' else '↔️'}",
        'RSI Durumu': f"{rsi_status} ({df['RSI'].iloc[-1]:.1f})",
        'Volatilite': f"{volatility_status} ({risk_metrics['Volatilite']*100:.1f}%)",
        'Durağanlık': stationarity,
        'MACD Sinyali': macd_signal,
        'Bollinger': bb_status,
        'Hacim Durumu': volume_status,
        'Risk Durumu': risk_status,
        'Teknik Göstergeler': ma_status,
        'Tahmin': f"{'YÜKSELİŞ 📈' if predictions['Tahmin Edilen Kapanış'] > df['close'].iloc[-1] else 'DÜŞÜŞ 📉'} (₺{predictions['Tahmin Edilen Kapanış']:.2f})",
        'Sharpe': f"{'MÜKEMMEL 🌟' if risk_metrics['Sharpe Oranı'] > 2 else 'İYİ ✅' if risk_metrics['Sharpe Oranı'] > 1 else 'ZAYIF ⚠️'}"
    }

def analyze_statistical_patterns(df):
    # Zamansallık analizi
    seasonal = seasonal_decompose(df['close'], period=30, model='additive')
    has_seasonality = seasonal.seasonal.std() > df['close'].std() * 0.1
    
    # Otokorelasyon analizi
    acf_values = acf(df['close'], nlags=30)
    has_autocorrelation = any(abs(acf_values[1:]) > 0.2)  # İlk lag'i atlıyoruz
    
    # Trend analizi
    z_score = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
    trend_strength = abs(z_score.mean())
    
    patterns = {
        'Mevsimsellik': has_seasonality,
        'Otokorelasyon': has_autocorrelation,
        'Trend Gücü': trend_strength,
        'Döngüsel Hareket': seasonal.seasonal.std() / df['close'].std()
    }
    
    return patterns

def analyze_correlation_matrix(corr_matrix):
    correlations = []
    
    # Önemli korelasyonları analiz et
    pairs = [
        ('close', 'volume'),
        ('close', 'RSI'),
        ('volume', 'Daily_Return'),
        ('RSI', 'Daily_Return')
    ]
    
    for var1, var2 in pairs:
        corr = corr_matrix.loc[var1, var2]
        strength = (
            "güçlü pozitif" if corr > 0.7
            else "orta pozitif" if corr > 0.3
            else "güçlü negatif" if corr < -0.7
            else "orta negatif" if corr < -0.3
            else "zayıf"
        )
        correlations.append({
            'pair': f"{var1}-{var2}",
            'correlation': corr,
            'strength': strength,
            'interpretation': interpret_correlation(var1, var2, corr)
        })
    
    return correlations

def interpret_correlation(var1, var2, corr):
    if var1 == 'close' and var2 == 'volume':
        if corr > 0.3:
            return "Yüksek hacim fiyat artışını destekliyor"
        elif corr < -0.3:
            return "Yüksek hacim fiyat düşüşünü destekliyor"
        else:
            return "Hacim ve fiyat arasında belirgin bir ilişki yok"
    
    elif (var1 == 'close' and var2 == 'RSI') or (var1 == 'RSI' and var2 == 'close'):
        if corr > 0.7:
            return "Güçlü trend mevcut"
        else:
            return "Trend zayıf veya yatay hareket mevcut"
    
    elif var1 == 'volume' and var2 == 'Daily_Return':
        if abs(corr) > 0.3:
            return "Hacim, günlük getirilerle ilişkili"
        else:
            return "Hacim, günlük getirilerle ilişkili değil"
    
    return "Standart korelasyon ilişkisi"

def create_candlestick_chart(df):
    # Mum grafiği
    candlestick = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Fiyat'
    )
    
    # Hareketli ortalamalar
    ma20 = go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue', width=1))
    ma50 = go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='orange', width=1))
    ma200 = go.Scatter(x=df.index, y=df['MA200'], name='MA200', line=dict(color='red', width=1))
    
    # Grafik düzeni
    layout = go.Layout(
        title='Hisse Senedi Fiyat Grafiği',
        yaxis=dict(title='Fiyat'),
        xaxis=dict(title='Tarih'),
        height=600
    )
    
    # Grafik oluşturma
    fig = go.Figure(data=[candlestick, ma20, ma50, ma200], layout=layout)
    
    return fig

def create_volume_chart(df):
    volume_chart = go.Bar(
        x=df.index,
        y=df['volume'],
        name='Hacim'
    )
    
    layout = go.Layout(
        title='Hacim Grafiği',
        yaxis=dict(title='Hacim'),
        xaxis=dict(title='Tarih'),
        height=300
    )
    
    fig = go.Figure(data=[volume_chart], layout=layout)
    return fig

def create_technical_charts(df):
    # RSI grafiği
    rsi = go.Scatter(x=df.index, y=df['RSI'], name='RSI')
    rsi_70 = go.Scatter(x=df.index, y=[70]*len(df), name='Aşırı Alım',
                       line=dict(color='red', dash='dash'))
    rsi_30 = go.Scatter(x=df.index, y=[30]*len(df), name='Aşırı Satım',
                       line=dict(color='green', dash='dash'))
    
    rsi_layout = go.Layout(
        title='RSI Göstergesi',
        yaxis=dict(title='RSI'),
        xaxis=dict(title='Tarih'),
        height=300
    )
    
    rsi_fig = go.Figure(data=[rsi, rsi_70, rsi_30], layout=rsi_layout)
    
    # MACD grafiği
    macd = go.Scatter(x=df.index, y=df['MACD'], name='MACD')
    signal = go.Scatter(x=df.index, y=df['Signal_Line'], name='Sinyal')
    
    macd_layout = go.Layout(
        title='MACD Göstergesi',
        yaxis=dict(title='MACD'),
        xaxis=dict(title='Tarih'),
        height=300
    )
    
    macd_fig = go.Figure(data=[macd, signal], layout=macd_layout)
    
    return rsi_fig, macd_fig

def calculate_fibonacci_levels(high, low):
    """Fibonacci düzeltme seviyelerini hesaplar"""
    diff = high - low
    levels = {
        "0.236 Seviyesi": low + diff * 0.236,
        "0.382 Seviyesi": low + diff * 0.382,
        "0.500 Seviyesi": low + diff * 0.500,
        "0.618 Seviyesi": low + diff * 0.618,
        "0.786 Seviyesi": low + diff * 0.786
    }
    return levels

def create_comprehensive_report(hisse_adi, df, summary, risk_metrics, stats_results, predictions, pattern_results, scenarios, volume_analysis):
    # Kapsamlı rapor oluştur
    st.header("Kapsamlı Analiz Raporu")
    
    # 1. ÖZET BİLGİLER
    st.header("1. ÖZET BİLGİLER")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Son Kapanış", f"₺{df['close'].iloc[-1]:.2f}")
    with col2:
        daily_return = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
        st.metric("Günlük Değişim", f"%{daily_return:.2f}")
    with col3:
        volume_change = ((df['volume'].iloc[-1] / df['volume'].iloc[-2]) - 1) * 100
        st.metric("Hacim Değişimi", f"%{volume_change:.2f}")
    with col4:
        st.metric("Günlük İşlem Hacmi", f"₺{df['volume'].iloc[-1]:,.0f}")

    # 2. TEKNİK ANALİZ
    st.header("2. TEKNİK ANALİZ")
    
    # Mum grafiği
    fig_candlestick = create_candlestick_chart(df)
    st.plotly_chart(fig_candlestick)
    
    # Teknik göstergeler
    col1, col2 = st.columns(2)
    with col1:
        # RSI grafiği
        rsi_fig = create_technical_charts(df)[0]
        st.plotly_chart(rsi_fig)
    with col2:
        # MACD grafiği
        macd_fig = create_technical_charts(df)[1]
        st.plotly_chart(macd_fig)

    # 3. İSTATİSTİKSEL ANALİZ
    st.header("3. İSTATİSTİKSEL ANALİZ")
    
    # Temel istatistikler
    st.subheader("3.1 Temel İstatistikler")
    basic_stats = df[['close', 'volume', 'Daily_Return']].describe()
    st.dataframe(basic_stats)
    
    # Risk metrikleri
    st.subheader("3.2 Risk Metrikleri")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Volatilite", f"%{risk_metrics['Volatilite']*100:.2f}")
    with col2:
        st.metric("Sharpe Oranı", f"{risk_metrics['Sharpe Oranı']:.2f}")
    with col3:
        st.metric("VaR (%95)", f"%{abs(risk_metrics['VaR_95']*100):.2f}")

    # 4. GELECEK TAHMİNLERİ
    st.header("4. GELECEK TAHMİNLERİ")
    
    # Tahmin özeti
    st.subheader("4.1 Fiyat Tahmini")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Yarınki Tahmin", f"₺{predictions['Tahmin Edilen Kapanış']:.2f}")
        st.metric("Beklenen Değişim", f"%{predictions['Değişim']:.2f}")
    
    # 5. PDF RAPORU
    st.header("5. PDF Raporu")
    create_pdf_report(hisse_adi, df, summary, risk_metrics, stats_results, predictions)

def create_technical_report(hisse_adi, df, technical_summary, risk_metrics, predictions):
    st.header("Teknik Analiz Raporu")
    
    # 1. FİYAT GRAFİĞİ
    st.subheader("1. Fiyat Grafiği")
    fig_candlestick = create_candlestick_chart(df)
    st.plotly_chart(fig_candlestick)
    
    # 2. TEKNİK GÖSTERGELER
    st.subheader("2. Teknik Göstergeler")
    
    col1, col2 = st.columns(2)
    with col1:
        # RSI
        rsi_fig = create_technical_charts(df)[0]
        st.plotly_chart(rsi_fig)
        
        current_rsi = df['RSI'].iloc[-1]
        st.metric("RSI", f"{current_rsi:.2f}")
        
    with col2:
        # MACD
        macd_fig = create_technical_charts(df)[1]
        st.plotly_chart(macd_fig)
        
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['Signal_Line'].iloc[-1]
        st.metric("MACD", f"{current_macd:.3f}")
        st.metric("Sinyal", f"{current_signal:.3f}")
    
    # 3. TREND ANALİZİ
    st.subheader("3. Trend Analizi")
    ma_cols = st.columns(3)
    with ma_cols[0]:
        st.metric("MA20", f"₺{df['MA20'].iloc[-1]:.2f}")
    with ma_cols[1]:
        st.metric("MA50", f"₺{df['MA50'].iloc[-1]:.2f}")
    with ma_cols[2]:
        st.metric("MA200", f"₺{df['MA200'].iloc[-1]:.2f}")
    
    # 4. RİSK METRİKLERİ
    st.subheader("4. Risk Metrikleri")
    risk_cols = st.columns(3)
    with risk_cols[0]:
        st.metric("Volatilite", f"%{risk_metrics['Volatilite']*100:.2f}")
    with risk_cols[1]:
        st.metric("VaR (%95)", f"%{abs(risk_metrics['VaR_95']*100):.2f}")
    with risk_cols[2]:
        st.metric("Max Drawdown", f"%{risk_metrics['Max Drawdown']*100:.2f}")
    
    # 5. TAHMİNLER
    st.subheader("5. Yarınki Tahminler")
    pred_cols = st.columns(2)
    with pred_cols[0]:
        st.metric("Tahmin Edilen Kapanış", f"₺{predictions['Tahmin Edilen Kapanış']:.2f}")
    with pred_cols[1]:
        st.metric("Beklenen Değişim", f"%{predictions['Değişim']:.2f}")

def create_statistical_report(hisse_adi, df, stats_results, pattern_results, seasonality_analysis, risk_metrics, predictions):
    st.header("İstatistiksel Analiz Raporu")
    
    # 1. TEMEL İSTATİSTİKLER
    st.subheader("1. Temel İstatistikler")
    basic_stats = df[['close', 'volume', 'Daily_Return']].describe()
    st.dataframe(basic_stats)
    
    # 2. İSTATİSTİKSEL TESTLER
    st.subheader("2. İstatistiksel Testler")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Durağanlık Testi (ADF)", f"p-değeri: {stats_results['ADF p-değeri']:.4f}")
        st.metric("Normallik Testi", f"p-değeri: {stats_results['Normallik p-değeri']:.4f}")
    with col2:
        st.metric("Otokorelasyon", f"{stats_results['Otokorelasyon']:.4f}")
        st.metric("Çarpıklık", f"{stats_results['Çarpıklık']:.4f}")
    
    # 3. ÖRÜNTÜ ANALİZİ
    st.subheader("3. Örüntü Analizi")
    if pattern_results['Mevsimsellik']:
        st.info("ℹ Mevsimsel örüntü tespit edildi")
    if pattern_results['Otokorelasyon']:
        st.info("ℹ Fiyat hareketlerinde süreklilik tespit edildi")
    if pattern_results['Trend Gücü'] > 1:
        st.warning(f"⚠️ Güçlü trend (z-skor: {pattern_results['Trend Gücü']:.2f})")
    
    # 4. RİSK ANALİZİ
    st.subheader("4. Risk Analizi")
    risk_cols = st.columns(3)
    with risk_cols[0]:
        st.metric("Volatilite", f"%{risk_metrics['Volatilite']*100:.2f}")
    with risk_cols[1]:
        st.metric("VaR (%95)", f"%{abs(risk_metrics['VaR_95']*100):.2f}")
    with risk_cols[2]:
        st.metric("Sharpe Oranı", f"{risk_metrics['Sharpe Oranı']:.2f}")
    
    # 5. TAHMİNLER
    st.subheader("5. Gelecek Tahmini")
    pred_cols = st.columns(2)
    with pred_cols[0]:
        st.metric("Yarınki Tahmin", f"₺{predictions['Tahmin Edilen Kapanış']:.2f}")
    with pred_cols[1]:
        st.metric("Beklenen Değişim", f"%{predictions['Değişim']:.2f}")

def generate_technical_analysis(df):
    # Teknik analiz sonuçları
    technical_summary = {
        'Teknik Analiz': "Teknik analiz sonuçları..."
    }
    return technical_summary

def perform_seasonality_analysis(df):
    # Mevsimsellik analizi
    seasonal_result = seasonal_decompose(df['close'], period=30)
    seasonality = seasonal_result.seasonal[-1]
    return seasonality

def create_pdf_report(hisse_adi, df, summary, risk_metrics, stats_results, predictions):
    """PDF raporu oluşturur"""
    try:
        # PDF dosya adını oluştur
        pdf_filename = f"{hisse_adi}_analiz_raporu.pdf"
        
        # PDF belgesini oluştur
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Başlık ekle
        title = Paragraph(f"{hisse_adi} Hisse Senedi Analiz Raporu", styles['Heading1'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Özet bilgileri ekle
        story.append(Paragraph("Özet Analiz", styles['Heading2']))
        story.append(Paragraph(str(summary), styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Risk metrikleri ekle
        story.append(Paragraph("Risk Metrikleri", styles['Heading2']))
        risk_data = [[k, f"{v:.2f}" if isinstance(v, float) else str(v)] 
                    for k, v in risk_metrics.items()]
        risk_table = Table(risk_data)
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 12))
        
        # İstatistiksel analiz sonuçları
        story.append(Paragraph("İstatistiksel Analiz", styles['Heading2']))
        stats_data = [[k, f"{v:.2f}" if isinstance(v, float) else str(v)] 
                     for k, v in stats_results.items()]
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 12))
        
        # Tahminler
        story.append(Paragraph("Gelecek Tahminleri", styles['Heading2']))
        pred_data = [[k, f"{v:.2f}" if isinstance(v, float) else str(v)] 
                    for k, v in predictions.items()]
        pred_table = Table(pred_data)
        pred_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(pred_table)
        
        # PDF oluştur
        doc.build(story)
        
        # Kullanıcıya indirme linki göster
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="📥 PDF Raporunu İndir",
                data=pdf_file,
                file_name=pdf_filename,
                mime="application/pdf"
            )
            
    except Exception as e:
        st.error(f"PDF raporu oluşturulurken bir hata oluştu: {str(e)}")

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="Hisse Senedi Analizi",
    page_icon="📈",
    layout="wide"
)

# Yan menü
col1, col2 = st.columns([1, 4])  # Sol menü için 1 birim, ana içerik için 4 birim genişlik

# Yan menü
with col1:
    st.header("📈 Analiz Parametreleri")
    
    # Hisse senedi seçimi
    hisse_adi = st.text_input("Hisse Adı (örn: THYAO):", "").upper()
    
    # CSV dosyası yükleme
    uploaded_file = st.file_uploader("CSV Dosyası Yükle", type=['csv'])
    
    if uploaded_file is not None:
        # Analiz türü seçimi
        st.subheader("📊 Analiz Türü Seçimi")
        analiz_turu = st.radio(
            "Hangi tür analiz yapmak istersiniz?",
            ["Kapsamlı Rapor Hazırla", 
             "Teknik Analiz Yap",
             "Veri ve İstatistiksel Analiz Yap"]
        )
        
        # Rapor hazırlama butonu
        if st.button("🚀 Raporu Hazırla", key="main_button"):
            if not uploaded_file.name.startswith(hisse_adi):
                st.error(f"Lütfen {hisse_adi} ile başlayan bir CSV dosyası yükleyin!")
            else:
                try:
                    # CSV dosyasını oku
                    df = pd.read_csv(uploaded_file)
                    
                    # Tarih sütununu düzenle
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Sütun isimlerini düzelt - hepsi küçük harf
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    
                    # Günlük getiriyi hesapla
                    df['Daily_Return'] = df['close'].pct_change()
                    
                    # Temel hesaplamalar
                    df = calculate_technical_indicators(df)
                    
                    try:
                        # Risk metrikleri ve tahminler her rapor türü için hesaplanır
                        risk_metrics = calculate_risk_metrics(df)
                        predictions = predict_next_day_values(df)
                        
                        if analiz_turu == "Kapsamlı Rapor Hazırla":
                            try:
                                # Tüm analizleri yap
                                stats_results = perform_statistical_analysis(df)
                                pattern_results = analyze_statistical_patterns(df)
                                scenarios = generate_alternative_scenarios(df, predictions)
                                volume_analysis = analyze_volume_scenarios(df, predictions)
                                summary = generate_analysis_summary(df, predictions, risk_metrics, stats_results)
                                
                                # Kapsamlı rapor oluştur
                                create_comprehensive_report(hisse_adi, df, summary, risk_metrics, stats_results, predictions, pattern_results, scenarios, volume_analysis)
                            except Exception as e:
                                st.error(f"Kapsamlı rapor oluşturulurken bir hata oluştu: {str(e)}")
                            
                        elif analiz_turu == "Teknik Analiz Yap":
                            try:
                                # Sadece teknik analiz yap
                                technical_summary = generate_technical_analysis(df)
                                create_technical_report(hisse_adi, df, technical_summary, risk_metrics, predictions)
                            except Exception as e:
                                st.error(f"Teknik analiz oluşturulurken bir hata oluştu: {str(e)}")
                            
                        else:  # Veri ve İstatistiksel Analiz
                            try:
                                # İstatistiksel analiz ve örüntü analizi
                                stats_results = perform_statistical_analysis(df)
                                pattern_results = analyze_statistical_patterns(df)
                                seasonality_analysis = perform_seasonality_analysis(df)
                                create_statistical_report(hisse_adi, df, stats_results, pattern_results, 
                                                       seasonality_analysis, risk_metrics, predictions)
                            except Exception as e:
                                st.error(f"İstatistiksel analiz oluşturulurken bir hata oluştu: {str(e)}")
                        
                        st.success("✅ Rapor başarıyla oluşturuldu!")
                        
                    except Exception as e:
                        st.error(f"Bir hata oluştu: {str(e)}")
                except Exception as e:
                    st.error(f"CSV dosyası okunurken bir hata oluştu: {str(e)}")

# Ana içerik
with col2:
    # Başlık ve açıklama
    st.title("📊 Hisse Senedi Analiz Platformu")
    st.markdown("""
    Bu uygulama ile hisse senetleri için detaylı teknik ve istatistiksel analizler yapabilirsiniz.
    """)
