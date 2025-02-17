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
def prepare_data(df_or_file):
    """Veriyi analiz için hazırlar"""
    try:
        # Eğer bir dosya nesnesi ise, önce DataFrame'e çevir
        if not isinstance(df_or_file, pd.DataFrame):
            df = pd.read_csv(df_or_file)
        else:
            df = df_or_file
            
        # Sütun isimlerini küçük harfe çevir
        df.columns = df.columns.str.lower()
        
        # Tarih sütununu indeks yap
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Gerekli sütunların varlığını kontrol et
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV dosyasında gerekli sütunlar eksik. Gerekli sütunlar: " + ", ".join(required_columns))
            return None
        
        # Günlük getiriyi hesapla
        df['Daily_Return'] = df['close'].pct_change() * 100
        
        # Teknik göstergeleri hesapla
        df = calculate_technical_indicators(df)
        
        return df
        
    except Exception as e:
        st.error(f"Veri hazırlama hatası: {str(e)}")
        return None

def generate_risk_analysis(risk_metrics):
    """Risk metriklerini yorumlar ve açıklar"""
    volatility = risk_metrics['Volatilite (%)']
    var = risk_metrics['VaR_95 (%)']
    sharpe = risk_metrics['Sharpe Oranı']
    drawdown = risk_metrics['Max Drawdown (%)']
    
    risk_text = f"""
    **⚠️ Risk Analizi Yorumu**
    
    **📊 Oynaklık (Volatilite): %{volatility:.2f}**
    - {'Çok Yüksek' if volatility > 4 else 'Yüksek' if volatility > 3 else 'Orta' if volatility > 2 else 'Düşük'} seviyede oynaklık
    - {
        'Fiyatta ani değişimler görülebilir, dikkatli olunmalı' if volatility > 4 else
        'Fiyat hareketleri ortalamadan daha oynak' if volatility > 3 else
        'Normal piyasa koşullarında beklenen oynaklık' if volatility > 2 else
        'Fiyat hareketleri nispeten sakin'
    }
    
    **💰 Riske Maruz Değer (VaR): %{var:.2f}**
    - Günlük maksimum kayıp riski (95% güven aralığında)
    - {
        'Çok yüksek risk seviyesi, dikkatli pozisyon alınmalı' if var < -5 else
        'Yüksek risk seviyesi, risk yönetimi önemli' if var < -3 else
        'Orta risk seviyesi, normal piyasa koşulları' if var < -2 else
        'Düşük risk seviyesi, görece güvenli'
    }
    
    **📈 Sharpe Oranı: {sharpe:.2f}**
    - {
        'Mükemmel risk/getiri oranı' if sharpe > 2 else
        'İyi risk/getiri oranı' if sharpe > 1 else
        'Ortalama risk/getiri oranı' if sharpe > 0 else
        'Risksiz getirinin altında performans'
    }
    - {
        'Yatırım için çok uygun' if sharpe > 2 else
        'Yatırım için uygun' if sharpe > 1 else
        'Risk/getiri dengesi normal' if sharpe > 0 else
        'Risk/getiri dengesi zayıf'
    }
    
    **📉 Maksimum Düşüş: %{drawdown:.2f}**
    - {
        'Çok ciddi bir düşüş yaşanmış' if drawdown < -30 else
        'Önemli bir düşüş yaşanmış' if drawdown < -20 else
        'Normal sayılabilecek düşüş' if drawdown < -10 else
        'Sınırlı düşüş yaşanmış'
    }
    - {
        'Toparlanma uzun sürebilir, dikkatli olunmalı' if drawdown < -30 else
        'Toparlanma süreci takip edilmeli' if drawdown < -20 else
        'Normal piyasa koşullarında beklenen düşüş' if drawdown < -10 else
        'Güçlü fiyat istikrarı'
    }
    """
    return risk_text

def generate_statistical_analysis(stats_results):
    """İstatistiksel analiz sonuçlarını yorumlar"""
    mean_return = stats_results['Ortalama Getiri']
    std_dev = stats_results['Standart Sapma']
    skewness = stats_results['Çarpıklık']
    rsi = stats_results['RSI']
    macd = stats_results['MACD']
    signal = stats_results['Signal']
    
    stats_text = f"""
    **📊 İstatistiksel Analiz Yorumu**
    
    **📈 Getiri Analizi:**
    - Ortalama Günlük Getiri: %{mean_return:.2f}
    - {
        'Çok güçlü pozitif getiri trendi' if mean_return > 1 else
        'Pozitif getiri trendi' if mean_return > 0.5 else
        'Hafif pozitif trend' if mean_return > 0 else
        'Negatif getiri trendi'
    }
    
    **🎯 Dağılım Analizi:**
    - Standart Sapma: %{std_dev:.2f}
    - Çarpıklık: {skewness:.2f}
    - {
        'Pozitif getiriler daha yaygın' if skewness > 0.5 else
        'Negatif getiriler daha yaygın' if skewness < -0.5 else
        'Dengeli getiri dağılımı'
    }
    
    **📊 Teknik Göstergeler:**
    - RSI: {rsi:.2f}
    - {
        'Aşırı alım bölgesi, düzeltme gelebilir' if rsi > 70 else
        'Aşırı satım bölgesi, yükseliş gelebilir' if rsi < 30 else
        'Normal bölgede, trend devam edebilir'
    }
    
    - MACD: {macd:.2f}
    - Sinyal: {signal:.2f}
    - {
        'Güçlü alım sinyali' if macd > signal and macd > 0 else
        'Zayıf alım sinyali' if macd > signal and macd <= 0 else
        'Güçlü satış sinyali' if macd < signal and macd < 0 else
        'Zayıf satış sinyali'
    }
    """
    return stats_text

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
    """Risk metriklerini hesaplar"""
    try:
        # Günlük getiriyi hesapla (zaten yüzde cinsinden)
        returns = df['Daily_Return'].dropna()
        
        # Aykırı değerleri temizle
        returns_clean = returns[np.abs(returns) <= returns.mean() + 3 * returns.std()]
        
        # Volatilite (yıllık)
        daily_volatility = returns_clean.std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Value at Risk (VaR)
        # Parametrik VaR hesaplama
        confidence_level_95 = 1.645  # 95% güven aralığı için z-score
        confidence_level_99 = 2.326  # 99% güven aralığı için z-score
        
        var_95 = -(returns_clean.mean() + confidence_level_95 * returns_clean.std())
        var_99 = -(returns_clean.mean() + confidence_level_99 * returns_clean.std())
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns_clean/100).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = ((cumulative_returns - rolling_max) / rolling_max) * 100
        max_drawdown = drawdowns.min()
        
        # Ani yükseliş ve düşüş riskleri
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        sudden_rise_risk = np.percentile(positive_returns, 95) if len(positive_returns) > 0 else 0
        sudden_fall_risk = abs(np.percentile(negative_returns, 5)) if len(negative_returns) > 0 else 0
        
        # Sharpe Ratio
        risk_free_rate = 0.05  # Yıllık %5
        daily_rf = risk_free_rate / 252
        excess_returns = returns_clean/100 - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Stop Loss ve Take Profit seviyeleri
        current_price = df['close'].iloc[-1]
        stop_loss_pct = max(var_95, 2.0)  # En az %2 stop loss
        take_profit_pct = stop_loss_pct * 1.5  # Risk/Ödül oranı 1.5
        
        stop_loss = current_price * (1 - stop_loss_pct/100)
        take_profit = current_price * (1 + take_profit_pct/100)
        
        metrics = {
            'Volatilite (%)': round(annual_volatility, 2),
            'VaR_95 (%)': round(abs(var_95), 2),
            'VaR_99 (%)': round(abs(var_99), 2),
            'Sharpe Oranı': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(abs(max_drawdown), 2),
            'Ani Yükseliş Riski (%)': round(sudden_rise_risk, 2),
            'Ani Düşüş Riski (%)': round(sudden_fall_risk, 2),
            'Stop Loss': round(stop_loss, 2),
            'Take Profit': round(take_profit, 2)
        }
        
        # Metriklerin mantıklı aralıklarda olduğunu kontrol et
        for key, value in metrics.items():
            if 'VaR' in key and (abs(value) > 20 or np.isnan(value)):
                metrics[key] = 5.0  # Varsayılan VaR değeri
            elif 'Max Drawdown' in key and (abs(value) > 50 or np.isnan(value)):
                metrics[key] = 20.0  # Varsayılan Maximum Drawdown değeri
            elif 'Risk' in key and (abs(value) > 20 or np.isnan(value)):
                metrics[key] = 5.0  # Varsayılan risk değeri
        
        return metrics
        
    except Exception as e:
        st.error(f"Risk metrikleri hesaplanırken bir hata oluştu: {str(e)}")
        # Hata durumunda varsayılan değerler
        return {
            'Volatilite (%)': 15.0,
            'VaR_95 (%)': 5.0,
            'VaR_99 (%)': 7.0,
            'Sharpe Oranı': 0.5,
            'Max Drawdown (%)': 20.0,
            'Ani Yükseliş Riski (%)': 5.0,
            'Ani Düşüş Riski (%)': 5.0,
            'Stop Loss': df['close'].iloc[-1] * 0.95,
            'Take Profit': df['close'].iloc[-1] * 1.075
        }

def detect_anomalies(df, window=20, std_dev=2):
    """Anomalileri tespit eder ve analiz eder"""
    returns = df['Daily_Return'].copy()
    
    # Rolling ortalama ve standart sapma
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Anomali tespiti
    upper_bound = rolling_mean + (std_dev * rolling_std)
    lower_bound = rolling_mean - (std_dev * rolling_std)
    
    # Aykırı değerleri temizle (çok ekstrem değerleri)
    returns = returns[np.abs(returns) < returns.mean() + 3 * returns.std()]
    
    # Anomalileri belirle
    anomalies_high = returns[returns > upper_bound]
    anomalies_low = returns[returns < lower_bound]
    
    # Son 30 günlük anomaliler
    last_30_days = returns[-30:]
    recent_anomalies = len(last_30_days[
        (last_30_days > upper_bound[-30:]) | 
        (last_30_days < lower_bound[-30:])
    ])
    
    # Önemli anomali tarihlerini bul (en yüksek 5 pozitif ve negatif)
    top_anomalies = returns[returns > upper_bound].nlargest(5)
    bottom_anomalies = returns[returns < lower_bound].nsmallest(5)
    
    # Anomali istatistikleri
    stats = {
        'positive_count': len(anomalies_high),
        'negative_count': len(anomalies_low),
        'positive_mean': anomalies_high.mean() if len(anomalies_high) > 0 else 0,
        'negative_mean': anomalies_low.mean() if len(anomalies_low) > 0 else 0,
        'recent_anomalies': recent_anomalies,
        'important_dates': pd.concat([top_anomalies, bottom_anomalies]).sort_values(ascending=False)
    }
    
    return stats

def format_anomaly_report(stats):
    """Anomali raporunu formatlar"""
    report = []
    
    # Genel istatistikler
    report.append(f"Pozitif Anomaliler: {stats['positive_count']} adet (Ortalama: %{stats['positive_mean']:.2f})")
    report.append(f"Negatif Anomaliler: {stats['negative_count']} adet (Ortalama: %{stats['negative_mean']:.2f})")
    report.append(f"Son 30 Günde: {stats['recent_anomalies']} adet")
    
    # Önemli tarihler
    report.append("🔍 Önemli Anomali Tarihleri:")
    for date, value in stats['important_dates'].head().items():
        report.append(f"- {date.strftime('%d/%m/%Y')}: %{value:.2f}")
    
    return "\n".join(report)

def perform_statistical_analysis(df):
    """İstatistiksel analiz yapar"""
    try:
        # Temel istatistikler
        returns = df['Daily_Return'].dropna()
        mean_return = returns.mean()
        std_dev = returns.std()
        skewness = returns.skew()
        
        # Teknik göstergeler
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        signal = df['Signal'].iloc[-1] if 'Signal' in df.columns else 0
        
        # Durağanlık testi
        try:
            adf_test = adfuller(df['close'].dropna())
            adf_pvalue = adf_test[1]
        except:
            adf_pvalue = 1.0
        
        # Trend analizi
        price_trend = 'Yükseliş' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'Düşüş'
        
        return {
            'Ortalama Getiri': mean_return,
            'Standart Sapma': std_dev,
            'Çarpıklık': skewness,
            'RSI': rsi,
            'MACD': macd,
            'Signal': signal,
            'ADF p-değeri': adf_pvalue,
            'Fiyat Trendi': price_trend
        }
        
    except Exception as e:
        st.error(f"İstatistiksel analiz hatası: {str(e)}")
        return {
            'Ortalama Getiri': 0.0,
            'Standart Sapma': 0.0,
            'Çarpıklık': 0.0,
            'RSI': 50.0,
            'MACD': 0.0,
            'Signal': 0.0,
            'ADF p-değeri': 1.0,
            'Fiyat Trendi': 'Belirsiz'
        }

def predict_next_day(df, index_data=None):
    """Hacim ve endeks bazlı tahmin yapar"""
    try:
        # Son günün verileri
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        
        # BIST100 korelasyonu
        if index_data is not None and isinstance(index_data, pd.DataFrame):
            correlation = df['Daily_Return'].corr(index_data['Daily_Return'])
            bist_momentum = index_data['close'].pct_change().iloc[-1]
        else:
            correlation = 0
            bist_momentum = 0
        
        # Teknik göstergeler
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        signal = df['Signal'].iloc[-1] if 'Signal' in df.columns else 0
        
        # Örüntü analizi
        pattern_signal = 0
        try:
            pattern_text, patterns = detect_patterns(df)
            for pattern, direction, _ in patterns:
                if direction == "Yükseliş":
                    pattern_signal += 1
                elif direction == "Düşüş":
                    pattern_signal -= 1
        except:
            pattern_signal = 0
            pattern_text = "Örüntü analizi yapılamadı"
        
        # Tahmin faktörleri
        volume_factor = 0.3 if volume_ratio > 1.5 else -0.2 if volume_ratio < 0.5 else 0
        rsi_factor = -0.2 if rsi > 70 else 0.2 if rsi < 30 else 0
        macd_factor = 0.2 if macd > signal else -0.2
        pattern_factor = 0.1 * pattern_signal
        bist_factor = correlation * bist_momentum if index_data is not None else 0
        
        # Toplam etki
        total_factor = volume_factor + rsi_factor + macd_factor + pattern_factor + bist_factor
        
        # Beklenen değişim
        expected_change = total_factor * 100  # Yüzde olarak
        
        # Tahmin edilen fiyat
        predicted_price = current_price * (1 + expected_change/100)
        
        # Tahmin güven seviyesi
        confidence = abs(total_factor)
        confidence_level = (
            'Yüksek' if confidence > 0.5 else
            'Orta' if confidence > 0.2 else
            'Düşük'
        )
        
        prediction_text = f"""
        **🔮 Yarın İçin Tahmin**
        
        **📊 Tahmin Faktörleri:**
        - Hacim Etkisi: {'Pozitif' if volume_factor > 0 else 'Negatif' if volume_factor < 0 else 'Nötr'}
        - Teknik Gösterge Etkisi: {'Pozitif' if (rsi_factor + macd_factor) > 0 else 'Negatif' if (rsi_factor + macd_factor) < 0 else 'Nötr'}
        - Örüntü Etkisi: {'Pozitif' if pattern_factor > 0 else 'Negatif' if pattern_factor < 0 else 'Nötr'}
        {f'- BIST100 Etkisi: {"Pozitif" if bist_factor > 0 else "Negatif" if bist_factor < 0 else "Nötr"}' if index_data is not None else ''}
        
        **📈 Beklenen Değişim:**
        - Yön: {'Yükseliş' if expected_change > 0 else 'Düşüş'}
        - Oran: %{expected_change:.2f}
        - Hedef Fiyat: ₺{predicted_price:.2f}
        
        **💡 Güven Seviyesi:** {confidence_level}
        - {'Yüksek güvenilirlik, güçlü sinyaller' if confidence > 0.5 else
           'Orta düzey güvenilirlik, dikkatli takip' if confidence > 0.2 else
           'Düşük güvenilirlik, ek analiz önerilir'}
        
        ⚠️ Not: Bu tahmin, geçmiş veriler ve teknik göstergeler baz alınarak yapılmıştır.
        Piyasa koşulları ve beklenmedik gelişmeler tahminleri etkileyebilir.
        """
        
        return {
            'Tahmin Edilen Kapanış': predicted_price,
            'Değişim': expected_change,
            'Güven Seviyesi': confidence_level,
            'Açıklama': prediction_text
        }
        
    except Exception as e:
        st.error(f"Tahmin hatası: {str(e)}")
        return None

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

def analyze_volume_scenarios(df):
    """Hacim senaryolarını analiz eder"""
    try:
        # Son 30 günlük ortalama hacim
        avg_volume = df['volume'].tail(30).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Hacim senaryoları ve yorumları
        scenarios = {
            'Yüksek Hacim': {
                'threshold': 1.5,
                'description': 'Hacim ortalamanın çok üzerinde',
                'impact': 'Güçlü fiyat hareketi beklenir',
                'interpretation': 'Yüksek hacim, piyasada güçlü bir ilgi olduğunu gösterir'
            },
            'Normal Hacim': {
                'threshold': 0.75,
                'description': 'Hacim normal seviyelerde',
                'impact': 'Normal fiyat hareketi beklenir',
                'interpretation': 'Normal hacim, piyasanın dengeli olduğunu gösterir'
            },
            'Düşük Hacim': {
                'threshold': 0,
                'description': 'Hacim ortalamanın altında',
                'impact': 'Zayıf fiyat hareketi beklenir',
                'interpretation': 'Düşük hacim, piyasada ilginin azaldığını gösterir'
            }
        }
        
        # Aktif senaryoyu belirle
        active_scenario = None
        for scenario, details in scenarios.items():
            if volume_ratio >= details['threshold']:
                active_scenario = scenario
                break
        
        # Hacim trendi
        volume_trend = df['volume'].tail(5).mean() > df['volume'].tail(20).mean()
        trend_text = "Yükseliş" if volume_trend else "Düşüş"
        
        analysis_text = f"""
        **📊 Hacim Analizi**
        - Güncel Hacim: {current_volume:,.0f}
        - 30 Günlük Ortalama: {avg_volume:,.0f}
        - Hacim/Ortalama Oranı: {volume_ratio:.2f}x
        - Hacim Trendi: {trend_text}
        
        **🔍 Aktif Senaryo: {active_scenario}**
        - Durum: {scenarios[active_scenario]['description']}
        - Beklenen Etki: {scenarios[active_scenario]['impact']}
        - Yorum: {scenarios[active_scenario]['interpretation']}
        """
        
        return analysis_text
        
    except Exception as e:
        st.error(f"Hacim analizi hatası: {str(e)}")
        return "Hacim analizi yapılamadı. Veri kalitesini kontrol edin."

def analyze_index_correlation(df, bist100_data):
    """BIST100 ile korelasyon analizi yapar"""
    try:
        # Veri kontrolü
        if bist100_data is None or not isinstance(bist100_data, pd.DataFrame):
            return "BIST100 verisi bulunamadı veya yanlış formatta. Korelasyon analizi yapılamadı."
            
        if 'Daily_Return' not in bist100_data.columns:
            return "BIST100 verisinde günlük getiri (Daily_Return) sütunu bulunamadı."  
        
        # Tarihleri indeks olarak ayarla
        df.index = pd.to_datetime(df.index)
        bist100_data.index = pd.to_datetime(bist100_data.index)
        
        # Ortak tarihleri bul
        common_dates = df.index.intersection(bist100_data.index)
        if len(common_dates) == 0:
            return "Hisse ve BIST100 verileri arasında ortak tarih bulunamadı."   
        
        # Ortak tarihlere göre verileri filtrele
        df_returns = df.loc[common_dates, 'Daily_Return']
        bist_returns = bist100_data.loc[common_dates, 'Daily_Return']
        
        if len(df_returns) < 2 or len(bist_returns) < 2:
            return "Korelasyon analizi için yeterli veri yok. En az 2 ortak tarih gerekli."
        
        # Korelasyon hesapla
        correlation = df_returns.corr(bist_returns)
        
        # Son 30 gün için korelasyon
        last_30_dates = common_dates[-30:] if len(common_dates) >= 30 else common_dates
        recent_correlation = df_returns.loc[last_30_dates].corr(bist_returns.loc[last_30_dates])
        
        # Korelasyon gücü ve yönü
        strength = 'Güçlü' if abs(correlation) > 0.7 else 'Orta' if abs(correlation) > 0.4 else 'Zayıf'
        direction = 'Pozitif' if correlation > 0 else 'Negatif'
        
        # Beta hesapla
        try:
            cov_matrix = np.cov(df_returns, bist_returns)
            if len(cov_matrix) > 1:
                beta = cov_matrix[0][1] / np.var(bist_returns)
            else:
                beta = 0
        except:
            beta = 0
            st.warning("Beta hesaplanamadı, varsayılan değer 0 kullanılıyor.")
        
        analysis_text = f"""
        **🔄 BIST100 Korelasyon Analizi**
        - Genel Korelasyon: {correlation:.2f}
        - Son {len(last_30_dates)} Gün Korelasyonu: {recent_correlation:.2f}
        - Korelasyon Gücü: {strength}
        - Korelasyon Yönü: {direction}
        - Beta Katsayısı: {beta:.2f}
        
        **📈 Olası Senaryolar:**
        - BIST100 Yükselirse: %{abs(correlation)*100:.1f} olasılıkla {direction} yönde hareket
        - BIST100 Düşerse: %{abs(correlation)*100:.1f} olasılıkla {direction} yönde hareket
        
        **💡 Yorum:**
        - {'Hisse, piyasa ile güçlü bir ilişki gösteriyor' if abs(correlation) > 0.7 else
          'Hisse, piyasa ile orta düzeyde ilişkili' if abs(correlation) > 0.4 else
          'Hisse, piyasadan bağımsız hareket ediyor'}
        - {'Hisse piyasadan daha oynak' if beta > 1 else 'Hisse piyasadan daha az oynak'} (Beta: {beta:.2f})
        """
        
        return analysis_text
        
    except Exception as e:
        st.error(f"Endeks korelasyonu analizi hatası: {str(e)}")
        return "Endeks korelasyonu analizi yapılamadı. Veri kalitesini kontrol edin."

def detect_patterns(df):
    """Teknik analiz örüntülerini tespit eder"""
    try:
        patterns = []
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Son 5 günlük veriler
        last_close = close[-5:]
        last_high = high[-5:]
        last_low = low[-5:]
        
        # Çift Tepe
        if (last_high[-5] > last_high[-4] and last_high[-3] > last_high[-4] and
            last_high[-3] < last_high[-2] and abs(last_high[-5] - last_high[-3]) < last_high[-5] * 0.02):
            patterns.append(("Çift Tepe", "Düşüş", "Güçlü bir düşüş sinyali"))
            
        # Çift Dip
        if (last_low[-5] < last_low[-4] and last_low[-3] < last_low[-4] and
            last_low[-3] > last_low[-2] and abs(last_low[-5] - last_low[-3]) < last_low[-5] * 0.02):
            patterns.append(("Çift Dip", "Yükseliş", "Güçlü bir yükseliş sinyali"))
            
        # Yükselen Üçgen
        if (max(last_high) - min(last_high) < max(last_low) - min(last_low) and
            last_close[-1] > last_close[-5]):
            patterns.append(("Yükselen Üçgen", "Yükseliş", "Trend devamı beklenebilir"))
            
        # Alçalan Üçgen
        if (max(last_high) - min(last_high) > max(last_low) - min(last_low) and
            last_close[-1] < last_close[-5]):
            patterns.append(("Alçalan Üçgen", "Düşüş", "Trend devamı beklenebilir"))
            
        if not patterns:
            patterns.append(("Belirgin Örüntü Yok", "Nötr", "Net bir sinyal bulunmuyor"))
        
        analysis_text = """
        **📊 Teknik Örüntü Analizi**
        """
        
        for pattern, direction, comment in patterns:
            analysis_text += f"""
            - Örüntü: {pattern}
            - Yön: {direction}
            - Yorum: {comment}
            """
            
        return analysis_text
        
    except Exception as e:
        st.error(f"Örüntü analizi hatası: {str(e)}")
        return "Örüntü analizi yapılamadı. Veri kalitesini kontrol edin."

def detect_anomalies(df, window=20, std_dev=2):
    """Anomalileri tespit eder ve analiz eder"""
    try:
        # Getiri ve hacim anomalileri
        returns = df['Daily_Return']
        volumes = df['volume']
        
        # Hareketli ortalama ve standart sapma
        returns_mean = returns.rolling(window=window).mean()
        returns_std = returns.rolling(window=window).std()
        volume_mean = volumes.rolling(window=window).mean()
        volume_std = volumes.rolling(window=window).std()
        
        # Anomali bantları
        upper_return = returns_mean + (std_dev * returns_std)
        lower_return = returns_mean - (std_dev * returns_std)
        upper_volume = volume_mean + (std_dev * volume_std)
        
        # Anomalileri tespit et
        return_anomalies = returns[(returns > upper_return) | (returns < lower_return)]
        volume_anomalies = volumes[volumes > upper_volume]
        
        # Son 30 gündeki anomaliler
        recent_return_anomalies = return_anomalies[-30:]
        recent_volume_anomalies = volume_anomalies[-30:]
        
        analysis_text = f"""
        **🔍 Anomali Analizi**
        
        **Getiri Anomalileri:**
        - Toplam Anomali Sayısı: {len(return_anomalies)}
        - Son 30 Gündeki Anomaliler: {len(recent_return_anomalies)}
        - Ortalama Anomali Büyüklüğü: %{abs(return_anomalies).mean():.2f}
        
        **Hacim Anomalileri:**
        - Toplam Anomali Sayısı: {len(volume_anomalies)}
        - Son 30 Gündeki Anomaliler: {len(recent_volume_anomalies)}
        - Ortalama Anomali Büyüklüğü: {(volume_anomalies / volume_mean).mean():.1f}x
        
        **💡 Yorum:**
        - {'Yüksek anomali aktivitesi' if len(recent_return_anomalies) > 3 else 'Normal anomali aktivitesi'}
        - {'Dikkat: Son dönemde artan anomaliler' if len(recent_return_anomalies) > len(return_anomalies)/6 else 'Anomali dağılımı normal'}
        """
        
        return analysis_text
        
    except Exception as e:
        st.error(f"Anomali analizi hatası: {str(e)}")
        return "Anomali analizi yapılamadı. Veri kalitesini kontrol edin."

def generate_analysis_summary(df, predictions, risk_metrics, stats_results):
    """Analiz özetini ve yorumları oluşturur"""
    try:
        # Son fiyat ve değişim
        current_price = df['close'].iloc[-1]
        price_change = df['Daily_Return'].iloc[-1] * 100
        
        # Trend analizi
        short_trend = df['close'].tail(5).mean() > df['close'].tail(20).mean()
        long_trend = df['close'].tail(20).mean() > df['close'].tail(50).mean()
        
        # RSI ve MACD durumu
        rsi = stats_results['RSI']
        macd = stats_results['MACD']
        signal = stats_results['Signal']
        
        # Tahmin yönü
        prediction_direction = "Yükseliş" if predictions['Değişim'] > 0 else "Düşüş"
        
        summary_text = f"""
        **🎯 Genel Görünüm ve Öneriler**
        
        **📊 Mevcut Durum:**
        - Güncel Fiyat: ₺{current_price:.2f}
        - Günlük Değişim: %{price_change:.2f}
        - {
            'Güçlü yükseliş trendi' if short_trend and long_trend else
            'Kısa vadeli yükseliş, uzun vadeli düşüş' if short_trend else
            'Kısa vadeli düşüş, uzun vadeli yükseliş' if long_trend else
            'Düşüş trendi'
        }
        
        **🔮 Teknik Görünüm:**
        - RSI Durumu: {
            'Aşırı alım bölgesinde' if rsi > 70 else
            'Aşırı satım bölgesinde' if rsi < 30 else
            'Normal bölgede'
        }
        - MACD Sinyali: {
            'Güçlü alım' if macd > signal and macd > 0 else
            'Zayıf alım' if macd > signal else
            'Güçlü satış' if macd < signal and macd < 0 else
            'Zayıf satış'
        }
        
        **📈 Tahmin ve Beklentiler:**
        - Beklenen Yön: {prediction_direction}
        - Hedef Fiyat: ₺{predictions['Tahmin Edilen Kapanış']:.2f}
        - Beklenen Değişim: %{predictions['Değişim']:.2f}
        
        **💡 Öneriler:**
        - {
            'Kısa vadeli kar realizasyonu düşünülebilir' if rsi > 70 and price_change > 2 else
            'Alım için uygun seviyeler' if rsi < 30 and price_change < -2 else
            'Mevcut pozisyonlar korunabilir' if 30 <= rsi <= 70 else
            'Temkinli yaklaşılmalı'
        }
        - {
            'Stop-loss seviyeleri yukarı çekilebilir' if short_trend and long_trend else
            'Yeni alımlar için düşüşler beklenebilir' if not short_trend and long_trend else
            'Kademeli alım stratejisi izlenebilir' if short_trend and not long_trend else
            'Risk yönetimine dikkat edilmeli'
        }
        
        ⚠️ Not: Bu analizler sadece bilgilendirme amaçlıdır ve kesin alım-satım önerisi içermez.
        """
        
        return summary_text
        
    except Exception as e:
        st.error(f"Analiz özeti oluşturma hatası: {str(e)}")
        return "Analiz özeti oluşturulamadı. Veri kalitesini kontrol edin."

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

def create_comprehensive_report(hisse_adi, df, summary, risk_metrics, stats_results, predictions, content_col):
    """Kapsamlı analiz raporu oluşturur"""
    try:
        with content_col:
            # Ana metrikler
            st.header(f"📊 {hisse_adi} Analiz Raporu")
            
            # Fiyat ve hacim bilgileri
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Son Kapanış", f"₺{df['close'].iloc[-1]:.2f}", 
                         f"%{df['Daily_Return'].iloc[-1]:.2f}")
            with col2:
                st.metric("Günlük Hacim", f"{df['volume'].iloc[-1]:,.0f}",
                         f"%{((df['volume'].iloc[-1] / df['volume'].iloc[-2]) - 1) * 100:.2f}")
            with col3:
                st.metric("Tahmin", 
                         f"₺{predictions['Tahmin Edilen Kapanış']:.2f}",
                         f"%{predictions['Değişim']:.2f}")
            
            # Hacim analizi
            st.subheader("📊 Hacim Analizi")
            hacim_analizi = analyze_volume_scenarios(df)
            st.write(hacim_analizi)
            
            # BIST100 korelasyonu
            if 'Endeks Korelasyonu' in predictions:
                st.subheader("🔄 BIST100 Korelasyonu")
                korelasyon_analizi = analyze_index_correlation(df, predictions['Endeks Korelasyonu'])
                st.write(korelasyon_analizi)
            
            # Örüntü analizi
            st.subheader("📈 Teknik Örüntüler")
            oruntu_analizi = detect_patterns(df)
            st.write(oruntu_analizi)
            
            # Anomali analizi
            st.subheader("🔍 Anomaliler")
            anomali_analizi = detect_anomalies(df)
            st.write(anomali_analizi)
            
            # Risk metrikleri
            st.subheader("⚠️ Risk Analizi")
            risk_col1, risk_col2 = st.columns(2)
            with risk_col1:
                st.write("**Temel Risk Metrikleri:**")
                st.write(f"- Volatilite: %{risk_metrics['Volatilite (%)']:.2f}")
                st.write(f"- VaR (95): %{risk_metrics['VaR_95 (%)']:.2f}")
                st.write(f"- Sharpe Oranı: {risk_metrics['Sharpe Oranı']:.2f}")
            with risk_col2:
                st.write("**İleri Risk Metrikleri:**")
                st.write(f"- Maximum Drawdown: %{risk_metrics['Max Drawdown (%)']:.2f}")
                st.write(f"- Ani Yükseliş Riski: %{risk_metrics['Ani Yükseliş Riski (%)']:.2f}")
                st.write(f"- Ani Düşüş Riski: %{risk_metrics['Ani Düşüş Riski (%)']:.2f}")
            risk_analysis = generate_risk_analysis(risk_metrics)
            st.write(risk_analysis)
            
            # İstatistiksel analiz
            st.subheader("📊 İstatistiksel Analiz")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.write("**Temel İstatistikler:**")
                st.write(f"- Ortalama Getiri: %{stats_results['Ortalama Getiri']:.2f}")
                st.write(f"- Standart Sapma: %{stats_results['Standart Sapma']:.2f}")
                st.write(f"- Çarpıklık: {stats_results['Çarpıklık']:.2f}")
            with stats_col2:
                st.write("**Trend Göstergeleri:**")
                st.write(f"- RSI: {stats_results['RSI']:.2f}")
                st.write(f"- MACD: {stats_results['MACD']:.2f}")
                st.write(f"- Signal: {stats_results['Signal']:.2f}")
            stats_analysis = generate_statistical_analysis(stats_results)
            st.write(stats_analysis)
            
            # Genel görünüm ve öneriler
            st.subheader("🎯 Genel Görünüm ve Öneriler")
            st.write(summary)
            summary = generate_analysis_summary(df, predictions, risk_metrics, stats_results)
            st.write(summary)
            
            # Uyarı notu
            st.warning("""
            ⚠️ **Önemli Not:** Bu analiz sadece bilgilendirme amaçlıdır ve kesin alım-satım önerisi içermez. 
            Yatırım kararlarınızı verirken profesyonel destek almanız önerilir.
            """)
            
    except Exception as e:
        st.error(f"Rapor oluşturma hatası: {str(e)}")
        raise Exception(f"Rapor oluşturma hatası: {str(e)}")

def create_technical_report(hisse_adi, df, technical_summary, risk_metrics, predictions, content_col):
    with content_col:  # Ana içerik sütununda göster
        st.header("Teknik Analiz Raporu")
        
        # 1. FİYAT GRAFİĞİ
        st.subheader("1. Fiyat Grafiği")
        fig_candlestick = create_candlestick_chart(df)
        st.plotly_chart(fig_candlestick, use_container_width=True)
        
        # 2. TEKNİK GÖSTERGELER
        st.subheader("2. Teknik Göstergeler")
        
        col1, col2 = st.columns(2)
        with col1:
            # RSI grafiği
            rsi_fig = create_technical_charts(df)[0]
            st.plotly_chart(rsi_fig, use_container_width=True)
            
            current_rsi = df['RSI'].iloc[-1]
            st.metric("RSI", f"{current_rsi:.2f}")
            
        with col2:
            # MACD grafiği
            macd_fig = create_technical_charts(df)[1]
            st.plotly_chart(macd_fig, use_container_width=True)
            
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
            st.metric("Volatilite", f"%{risk_metrics['Volatilite (%)']:.2f}")
        with risk_cols[1]:
            st.metric("VaR (%95)", f"%{abs(risk_metrics['VaR_95 (%)']):.2f}")
        with risk_cols[2]:
            st.metric("Max Drawdown", f"%{abs(risk_metrics['Max Drawdown (%)']):.2f}")
        
        # 5. TAHMİNLER
        st.subheader("5. Yarınki Tahminler")
        pred_cols = st.columns(2)
        with pred_cols[0]:
            st.metric("Tahmin Edilen Kapanış", f"₺{predictions['Tahmin Edilen Kapanış']:.2f}")
        with pred_cols[1]:
            st.metric("Beklenen Değişim", f"%{predictions['Değişim']:.2f}")

def create_statistical_report(hisse_adi, df, stats_results, predictions, content_col):
    with content_col:
        st.header("📊 İstatistiksel Analiz Raporu")
        
        # GÜNCEL FİYAT BİLGİLERİ
        st.subheader("📈 Güncel Fiyat Bilgileri")
        
        son_fiyat = df['close'].iloc[-1]
        gunluk_degisim = ((son_fiyat - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        haftalik_degisim = ((son_fiyat - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
        aylik_degisim = ((son_fiyat - df['close'].iloc[-22]) / df['close'].iloc[-22]) * 100
        
        price_cols = st.columns(4)
        with price_cols[0]:
            st.metric("Son Fiyat", f"₺{son_fiyat:.2f}", f"%{gunluk_degisim:.2f}")
        with price_cols[1]:
            st.metric("Günlük Değişim", f"%{gunluk_degisim:.2f}")
        with price_cols[2]:
            st.metric("Haftalık Değişim", f"%{haftalik_degisim:.2f}")
        with price_cols[3]:
            st.metric("Aylık Değişim", f"%{aylik_degisim:.2f}")
            
        # Son işlem günü değerleri
        st.info(f"""
        **📊 Son İşlem Günü Değerleri:**
        - Açılış: ₺{df['open'].iloc[-1]:.2f}
        - En Yüksek: ₺{df['high'].iloc[-1]:.2f}
        - En Düşük: ₺{df['low'].iloc[-1]:.2f}
        - Kapanış: ₺{df['close'].iloc[-1]:.2f}
        - Hacim: {df['volume'].iloc[-1]:,.0f}
        """)
        
        # TEMEL İSTATİSTİKLER VE GETİRİ ANALİZİ
        st.subheader("📈 Temel İstatistikler ve Getiri Analizi")
        
        # Temel istatistikler tablosu
        stats_df = df[['close', 'volume', 'Daily_Return']].describe()
        stats_df.index = ['Gözlem Sayısı', 'Ortalama', 'Standart Sapma', 'Minimum', '25%', 'Medyan', '75%', 'Maksimum']
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
        
        # Getiri ve volatilite analizi
        mean_return = df['Daily_Return'].mean() * 100
        volatility = df['Daily_Return'].std() * 100
        skewness = df['Daily_Return'].skew()
        kurtosis = df['Daily_Return'].kurtosis()
        
        st.info(f"""
        **📊 Getiri ve Volatilite Analizi:**
        - Ortalama Günlük Getiri: %{mean_return:.2f}
        - Günlük Volatilite: %{volatility:.2f}
        - Getiri Dağılımı: {'Sağa Çarpık (Büyük Kazanç Potansiyeli)' if skewness > 0 else 'Sola Çarpık (Büyük Kayıp Riski)'} (Çarpıklık: {skewness:.2f})
        - Basıklık: {kurtosis:.2f} ({'Yüksek Uç Değer Riski' if kurtosis > 3 else 'Normal Dağılım'})
        
        **💡 Yorum:**
        - {'✅ Pozitif ortalama getiri' if mean_return > 0 else '⚠️ Negatif ortalama getiri'}
        - {'⚠️ Yüksek volatilite - Dikkatli olunmalı' if volatility > 2 else '✅ Normal volatilite seviyesi' if volatility > 1 else '✅ Düşük volatilite - İstikrarlı seyir'}
        - {'🎯 Büyük kazanç fırsatları mevcut' if skewness > 0 else '⚠️ Büyük kayıp riski mevcut'} 
        """)
        
        # RİSK ANALİZİ
        st.subheader("⚠️ Detaylı Risk Analizi")
        
        # Risk metrikleri hesaplama
        risk_free_rate = 0.05  # Risksiz faiz oranı
        excess_returns = df['Daily_Return'] - risk_free_rate/252
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # VaR hesaplama
        var_95 = np.percentile(df['Daily_Return'], 5) * 100
        var_99 = np.percentile(df['Daily_Return'], 1) * 100
        
        # Maximum Drawdown hesaplama
        cumulative_returns = (1 + df['Daily_Return']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = ((cumulative_returns - rolling_max) / rolling_max) * 100
        max_drawdown = drawdowns.min() * 100
        
        risk_cols = st.columns(3)
        with risk_cols[0]:
            st.metric("Sharpe Oranı", f"{sharpe:.2f}")
        with risk_cols[1]:
            st.metric("VaR (%95)", f"%{abs(var_95):.2f}")
        with risk_cols[2]:
            st.metric("Maximum Drawdown", f"%{abs(max_drawdown):.2f}")
        
        st.info(f"""
        **📊 Risk Değerlendirmesi:**
        - Risk/Getiri Oranı: {'Çok İyi' if sharpe > 1 else 'İyi' if sharpe > 0 else 'Kötü'} (Sharpe: {sharpe:.2f})
        - %95 Güven Düzeyinde VaR: %{abs(var_95):.2f}
        - %99 Güven Düzeyinde VaR: %{abs(var_99):.2f}
        - En Büyük Düşüş: %{abs(max_drawdown):.2f}
        
        **💡 Risk Yönetimi Önerileri:**
        1. {'⚠️ Stop-loss kullanımı ZORUNLU' if abs(max_drawdown) > 10 else '✅ Normal stop-loss yeterli'}
        2. {'⚠️ Pozisyon büyüklüğü sınırlandırılmalı' if abs(var_95) > 3 else '✅ Normal pozisyon büyüklüğü'}
        3. {'🎯 Kademeli alım stratejisi önerilir' if sharpe < 0.5 else '✅ Normal alım stratejisi'}
        """)
        
        # ÖRÜNTÜ VE ANOMALİ ANALİZİ
        st.subheader("🔍 Örüntü ve Anomali Analizi")
        
        # Mevsimsellik analizi
        try:
            decomposition = seasonal_decompose(df['close'], period=30, model='additive')
            seasonal_pattern = decomposition.seasonal[-1]
            seasonal_strength = np.std(decomposition.seasonal) / np.std(decomposition.resid)
            has_seasonality = seasonal_strength > 0.1
            
            # Mevsimsel döngülerin analizi
            monthly_returns = df.groupby(df.index.month)['Daily_Return'].mean() * 100
            best_month = monthly_returns.idxmax()
            worst_month = monthly_returns.idxmin()
            
            # Haftalık analiz
            weekly_returns = df.groupby(df.index.dayofweek)['Daily_Return'].mean() * 100
            best_day = weekly_returns.idxmax()
            worst_day = weekly_returns.idxmin()
            
            # Günlük pattern
            hourly_pattern = seasonal_pattern.groupby(seasonal_pattern.index.day).mean()
            strong_days = hourly_pattern[abs(hourly_pattern) > hourly_pattern.std()].index
            
            day_names = {0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe', 4: 'Cuma'}
            month_names = {1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
                         7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'}
            
            if has_seasonality:
                st.info(f"""
                🔄 **Mevsimsel Örüntü Analizi**
                
                **Aylık Döngüler:**
                - En İyi Ay: {month_names[best_month]} (Ort. %{monthly_returns[best_month]:.2f})
                - En Kötü Ay: {month_names[worst_month]} (Ort. %{monthly_returns[worst_month]:.2f})
                
                **Haftalık Döngüler:**
                - En İyi Gün: {day_names[best_day]} (Ort. %{weekly_returns[best_day]:.2f})
                - En Kötü Gün: {day_names[worst_day]} (Ort. %{weekly_returns[worst_day]:.2f})
                
                **💡 Alım-Satım Önerileri:**
                1. Alım için en uygun dönem: {month_names[worst_month]} ayı, özellikle {day_names[worst_day]} günleri
                2. Satış için en uygun dönem: {month_names[best_month]} ayı, özellikle {day_names[best_day]} günleri
                3. Güçlü Fiyat Hareketleri: Ayın {', '.join(map(str, strong_days))}. günlerinde
                
                **⚠️ Not:** Bu örüntüler geçmiş veriye dayalıdır ve gelecekte değişebilir.
                """)
        except Exception as e:
            has_seasonality = False
            
        # Anomali tespiti
        returns_mean = df['Daily_Return'].mean()
        returns_std = df['Daily_Return'].std()
        outliers = df[abs(df['Daily_Return'] - returns_mean) > 2 * returns_std]
        
        if not outliers.empty:
            # Anomalileri sınıflandır
            positive_anomalies = outliers[outliers['Daily_Return'] > returns_mean]
            negative_anomalies = outliers[outliers['Daily_Return'] < returns_mean]
            
            # Son 30 gündeki anomaliler
            recent_outliers = outliers[outliers.index >= outliers.index[-1] - pd.Timedelta(days=30)]
            
            # Hacim anomalileri
            volume_mean = df['volume'].mean()
            volume_std = df['volume'].std()
            volume_outliers = df[df['volume'] > volume_mean + 2 * volume_std]
            
            # Fiyat ve hacim anomalilerinin kesişimi
            combined_anomalies = pd.merge(outliers, volume_outliers, left_index=True, right_index=True, how='inner')
            
            st.warning(f"""
            ⚠️ **Anomali Analizi**
            
            **📊 Tespit Edilen Anomaliler:**
            - Toplam Anomali Sayısı: {len(outliers)} adet
            - Pozitif Anomaliler: {len(positive_anomalies)} adet (Ortalama: %{positive_anomalies['Daily_Return'].mean()*100:.2f})
            - Negatif Anomaliler: {len(negative_anomalies)} adet (Ortalama: %{negative_anomalies['Daily_Return'].mean()*100:.2f})
            - Son 30 Günde: {len(recent_outliers)} adet
            
            **🔍 Önemli Anomali Tarihleri:**
            {outliers.sort_values('Daily_Return', ascending=False)[['Daily_Return']].head().apply(lambda x: f"- {x.name.strftime('%d/%m/%Y')}: %{x['Daily_Return']*100:.2f}", axis=1).str.cat(sep='\\n')}
            
            **📈 Hacim Anomalileri ile Kesişim:**
            - {len(combined_anomalies)} adet fiyat hareketi yüksek hacim ile destekleniyor
            
            **💡 Dikkat Edilmesi Gereken Durumlar:**
            1. {'⚠️ Son 30 günde yüksek anomali - Dikkatli olunmalı!' if len(recent_outliers) > 0 else '✅ Son 30 günde önemli anomali yok'}
            2. Ani Yükseliş Riski: %{positive_anomalies['Daily_Return'].max()*100:.2f}
            3. Ani Düşüş Riski: %{abs(negative_anomalies['Daily_Return'].min()*100):.2f}
            
            **🎯 Öneriler:**
            1. Stop-Loss Seviyeleri: ₺{df['close'].iloc[-1] * (1 - abs(negative_anomalies['Daily_Return'].mean())):.2f}
            2. Kar Al Seviyeleri: ₺{df['close'].iloc[-1] * (1 + positive_anomalies['Daily_Return'].mean()):.2f}
            3. {'⚠️ Yüksek hacimli işlemlerde dikkatli olun' if len(combined_anomalies) > 0 else '✅ Hacim seviyeleri normal'}
            """)
            
            # Anomali grafiği
            fig = go.Figure()
            
            # Normal fiyat hareketleri
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Fiyat',
                line=dict(color='blue', width=1)
            ))
            
            # Pozitif anomaliler
            fig.add_trace(go.Scatter(
                x=positive_anomalies.index,
                y=positive_anomalies['close'],
                mode='markers',
                name='Pozitif Anomaliler',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            # Negatif anomaliler
            fig.add_trace(go.Scatter(
                x=negative_anomalies.index,
                y=negative_anomalies['close'],
                mode='markers',
                name='Negatif Anomaliler',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            
            fig.update_layout(
                title='Anomali Tespiti Grafiği',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # SONUÇ VE ÖNERİLER
        st.subheader("💡 Sonuç ve Öneriler")
        
        # Son 20 günlük trend analizi
        last_20_change = ((df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]) * 100
        rsi = df['RSI'].iloc[-1]
        
        st.success(f"""
        **📈 Özet Bulgular:**
        1. Getiri Profili: {'Pozitif' if mean_return > 0 else 'Negatif'} (%{mean_return:.2f})
        2. Risk Seviyesi: {'Yüksek' if abs(var_95) > 3 else 'Orta' if abs(var_95) > 2 else 'Düşük'}
        3. Yatırım Kalitesi: {'Yüksek' if sharpe > 1 else 'Orta' if sharpe > 0 else 'Düşük'}
        
        **🎯 Yatırım Stratejisi:**
        1. {'💹 GÜÇLÜ AL' if mean_return > 0 and sharpe > 1 and rsi < 70 else '✅ AL' if mean_return > 0 and sharpe > 0 and rsi < 70 else '⛔ SAT' if mean_return < 0 and sharpe < 0 else '⚠️ TUT'}
        2. Stop-Loss: ₺{df['close'].iloc[-1] * (1 - abs(var_95/100)):.2f}
        3. Hedef Fiyat: ₺{predictions['Tahmin Edilen Kapanış']:.2f}
        
        **⚠️ Önemli Uyarılar:**
        1. {f'⚠️ Yüksek risk! Sıkı risk yönetimi şart!' if abs(var_95) > 3 else '✅ Normal risk yönetimi yeterli'}
        2. {f'⚠️ RSI aşırı {"alım" if rsi > 70 else "satım"} bölgesinde!' if rsi > 70 or rsi < 30 else '✅ Teknik göstergeler normal'}
        3. {'⚠️ Anormal fiyat hareketlerine dikkat!' if not outliers.empty else '✅ Fiyat hareketleri normal'}
        """)
        
        # YARIN İÇİN TAHMİNLER
        st.subheader("🎯 Yarın İçin Tahminler")
        
        # Son kapanış fiyatı
        son_fiyat = df['close'].iloc[-1]
        
        # Son 5 günlük fiyat değişim yüzdesi ortalaması
        son_5_gun_degisim = df['close'].pct_change().tail(5).mean()
        
        # Fiyat aralığı tahmini için volatilite
        price_std = df['close'].pct_change().std()
        
        # RSI ve Momentum bazlı düzeltme
        rsi = df['RSI'].iloc[-1]
        momentum = df['close'].diff(5).iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal'].iloc[-1]
        
        # Temel tahmin - ARIMA tahminini baz alalım
        base_prediction = predictions['Tahmin Edilen Kapanış']
        
        # Teknik göstergelere göre düzeltme faktörü
        adjustment = 1.0
        
        # RSI bazlı düzeltme
        if rsi > 70:
            adjustment *= 0.995  # Aşırı alım - hafif düşüş beklentisi
        elif rsi < 30:
            adjustment *= 1.005  # Aşırı satım - hafif yükseliş beklentisi
        
        # MACD bazlı düzeltme
        if macd > signal:
            adjustment *= 1.002  # Yükseliş sinyali
        else:
            adjustment *= 0.998  # Düşüş sinyali
        
        # Momentum bazlı düzeltme
        if momentum > 0:
            adjustment *= 1.001  # Pozitif momentum
        else:
            adjustment *= 0.999  # Negatif momentum
        
        # Son 5 günlük trend bazlı düzeltme
        if son_5_gun_degisim > 0:
            adjustment *= 1.001
        else:
            adjustment *= 0.999
        
        # Tahminleri hesapla
        predicted_close = base_prediction * adjustment
        
        # Gün içi değişim aralığını hesapla (son 20 günlük ortalama)
        avg_daily_range = (df['high'] - df['low']).tail(20).mean() / df['close'].tail(20).mean()
        expected_range = predicted_close * avg_daily_range
        
        # Açılış fiyatı tahmini (son kapanışa daha yakın olmalı)
        predicted_open = son_fiyat * (1 + (predicted_close/son_fiyat - 1) * 0.3)
        
        # Yüksek ve düşük tahminleri
        predicted_high = max(predicted_open, predicted_close) + (expected_range/2)
        predicted_low = min(predicted_open, predicted_close) - (expected_range/2)
        
        # Değişim yüzdesi
        predicted_change = ((predicted_close - son_fiyat) / son_fiyat) * 100
        
        # Tahmin güvenilirliği
        confidence = "Yüksek" if abs(predicted_change) < 2 else "Orta" if abs(predicted_change) < 5 else "Düşük"
        
        pred_cols = st.columns(2)
        with pred_cols[0]:
            st.metric("Tahmini Kapanış", f"₺{predicted_close:.2f}", f"%{predicted_change:.2f}")
            st.metric("Tahmini En Yüksek", f"₺{predicted_high:.2f}")
        with pred_cols[1]:
            st.metric("Tahmini Açılış", f"₺{predicted_open:.2f}")
            st.metric("Tahmini En Düşük", f"₺{predicted_low:.2f}")
            
        st.info(f"""
        **📊 Tahmin Detayları:**
        - Beklenen İşlem Aralığı: ₺{predicted_low:.2f} - ₺{predicted_high:.2f}
        - Tahmin Güvenilirliği: {confidence}
        
        **💡 Tahmin Faktörleri:**
        1. RSI Durumu: {'Aşırı Alım - Düşüş Baskısı' if rsi > 70 else 'Aşırı Satım - Yükseliş Potansiyeli' if rsi < 30 else 'Normal Seviye'} ({rsi:.0f})
        2. MACD Sinyali: {'Alış' if macd > signal else 'Satış'}
        3. Momentum: {'Pozitif' if momentum > 0 else 'Negatif'}
        4. Son 5 Günlük Trend: {'Yükseliş' if son_5_gun_degisim > 0 else 'Düşüş'} (%{son_5_gun_degisim*100:.2f})
        5. Volatilite: {'Yüksek' if price_std > 0.02 else 'Normal' if price_std > 0.01 else 'Düşük'}
        
        **⚠️ Not:** 
        - Bu tahminler teknik analiz ve istatistiksel modellere dayanmaktadır
        - Piyasa koşulları ve beklenmedik gelişmeler tahminleri etkileyebilir
        - Önemli bir haber akışı durumunda tahminler geçerliliğini yitirebilir
        """)



def generate_technical_analysis(df):
    # Teknik analiz sonuçları
    technical_summary = {
        'Teknik Analiz': "Teknik analiz sonuçları..."
    }
    return technical_summary

def perform_seasonality_analysis(df):
    # Mevsimsellik analizi
    seasonal_result = seasonal_decompose(df['close'], period=30, model='additive')
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

# Streamlit uygulaması
st.set_page_config(page_title="Hisse Senedi Analizi", page_icon="📈", layout="wide")

# Ana container
main_container = st.container()

def main():
    try:
        # Başlık
        st.title("📊 Hisse Senedi Analiz Platformu")
        st.markdown("""
        Bu uygulama ile hisse senetleri için detaylı teknik ve istatistiksel analizler yapabilirsiniz.
        """)
        
        # Yan panel ve ana içerik için sütunlar
        sidebar, main_content = st.columns([1, 3])
        
        with sidebar:
            st.header("📈 Analiz Parametreleri")
            
            # Hisse senedi verisi yükleme
            st.subheader("📊 Hisse Senedi Verisi")
            uploaded_file = st.file_uploader("Hisse CSV dosyası yükleyin", type=['csv'])
            
            # BIST100 analizi seçeneği
            use_bist100 = st.checkbox("🔄 BIST100 Analizi Yap", value=False)
            
            # BIST100 verisi yükleme (eğer seçenek aktifse)
            bist100_data = None
            if use_bist100:
                st.subheader("📈 BIST100 Verisi")
                bist100_file = st.file_uploader("BIST100 CSV dosyası yükleyin", type=['csv'], key="bist100")
                if bist100_file is not None:
                    try:
                        bist100_df = pd.read_csv(bist100_file)
                        bist100_data = prepare_data(bist100_df)
                        if bist100_data is not None:
                            st.success("✅ BIST100 verisi başarıyla yüklendi")
                        else:
                            st.error("❌ BIST100 verisi hazırlanamadı")
                    except Exception as e:
                        st.error(f"❌ BIST100 verisi okuma hatası: {str(e)}")
            
            if uploaded_file is not None:
                try:
                    # CSV dosyasını oku
                    df = pd.read_csv(uploaded_file)
                    
                    # Veriyi hazırla
                    df = prepare_data(df)
                    
                    if df is not None:
                        # Hisse adı
                        hisse_adi = uploaded_file.name.split('.')[0].upper()
                        st.success(f"✅ {hisse_adi} verisi başarıyla yüklendi")
                        
                        # Analiz butonu
                        if st.button("🔄 Analiz Et"):
                            with st.spinner('Analiz yapılıyor...'):
                                try:
                                    # Risk metriklerini hesapla
                                    risk_metrics = calculate_risk_metrics(df)
                                    
                                    # İstatistiksel analiz yap
                                    stats_results = perform_statistical_analysis(df)
                                    
                                    # Tahminleri yap
                                    predictions = {
                                        'Tahmin Edilen Kapanış': 0.0,
                                        'Değişim': 0.0,
                                        'Açıklama': ''
                                    }
                                    
                                    # BIST100 analizi (eğer seçenek aktif ve veri yüklendiyse)
                                    if use_bist100 and bist100_data is not None:
                                        # Korelasyon analizi
                                        correlation_analysis = analyze_index_correlation(df, bist100_data)
                                        predictions['BIST100 Analizi'] = correlation_analysis
                                        
                                        # Tahminleri güncelle
                                        predictions = predict_next_day(df, bist100_data)
                                    else:
                                        # BIST100 olmadan tahmin yap
                                        predictions = predict_next_day(df, None)
                                    
                                    if predictions:
                                        # Kapsamlı rapor oluştur
                                        with main_content:
                                            create_comprehensive_report(
                                                hisse_adi,
                                                df,
                                                generate_analysis_summary(df, predictions, risk_metrics, stats_results),
                                                risk_metrics,
                                                stats_results,
                                                predictions,
                                                st.container()
                                            )
                                    else:
                                        st.error("❌ Tahmin yapılamadı. Veri kalitesini kontrol edin.")
                                        
                                except Exception as e:
                                    st.error(f"❌ Analiz hatası: {str(e)}")
                    else:
                        st.error("❌ Veri hazırlama hatası. CSV dosyasını kontrol edin.")
                        
                except Exception as e:
                    st.error(f"❌ Dosya okuma hatası: {str(e)}")
            else:
                st.info("ℹ️ Lütfen bir hisse senedi CSV dosyası yükleyin.")
                
        # Ana içerik alanı
        with main_content:
            if uploaded_file is None:
                st.markdown("""
                ### 👋 Hoş Geldiniz!
                
                **📊 Hisse Senedi Analiz Platformu**'na hoş geldiniz. 
                
                #### 🚀 Özellikler:
                - ✨ Detaylı Teknik Analiz
                - 📈 Risk Metrikleri
                - 🔄 BIST100 Korelasyonu (Opsiyonel)
                - 🎯 Fiyat Tahminleri
                - 📊 Kapsamlı Raporlama
                
                #### 📝 Nasıl Kullanılır:
                1. Sol panelden hisse senedi CSV dosyanızı yükleyin
                2. BIST100 analizi yapmak istiyorsanız seçeneği aktif edin
                3. BIST100 analizi seçiliyse, BIST100 CSV dosyasını yükleyin
                4. "Analiz Et" butonuna tıklayın
                5. Detaylı analiz sonuçlarını inceleyin
                
                #### 📋 CSV Formatı:
                Dosyalarınızda şu sütunlar bulunmalıdır:
                - date: Tarih
                - open: Açılış fiyatı
                - high: En yüksek fiyat
                - low: En düşük fiyat
                - close: Kapanış fiyatı
                - volume: İşlem hacmi
                """)
                
    except Exception as e:
        st.error(f"❌ Uygulama hatası: {str(e)}")

if __name__ == "__main__":
    main()

def get_bist100_data(start_date, end_date):
    """BIST100 verilerini çeker"""
    try:
        # BIST100 verisini çek
        bist = yf.download('^XU100', start=start_date, end=end_date)
        
        # Günlük getiriyi hesapla
        bist['Daily_Return'] = bist['Close'].pct_change() * 100
        
        # Sütun isimlerini küçük harfe çevir
        bist.columns = bist.columns.str.lower()
        
        return bist
    except Exception as e:
        st.error(f"BIST100 verisi çekme hatası: {str(e)}")
        return None
