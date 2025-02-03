import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Türkçe font desteği için
pdfmetrics.registerFont(TTFont('DejaVuSans', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))

def predict_next_day_values(df, features):
    # Veri hazırlama
    scaler = StandardScaler()
    
    # Açılış tahmini için model
    df['Next_Open'] = df['open'].shift(-1)
    df_open = df.dropna()
    X_open = df_open[features]
    y_open = df_open['Next_Open']
    X_open_scaled = scaler.fit_transform(X_open)
    model_open = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_open.fit(X_open_scaled, y_open)
    
    # Yüksek tahmin için model
    df['Next_High'] = df['high'].shift(-1)
    df_high = df.dropna()
    X_high = df_high[features]
    y_high = df_high['Next_High']
    X_high_scaled = scaler.fit_transform(X_high)
    model_high = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_high.fit(X_high_scaled, y_high)
    
    # Düşük tahmin için model
    df['Next_Low'] = df['low'].shift(-1)
    df_low = df.dropna()
    X_low = df_low[features]
    y_low = df_low['Next_Low']
    X_low_scaled = scaler.fit_transform(X_low)
    model_low = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_low.fit(X_low_scaled, y_low)
    
    # Kapanış tahmini için model
    df['Next_Close'] = df['close'].shift(-1)
    df_close = df.dropna()
    X_close = df_close[features]
    y_close = df_close['Next_Close']
    X_close_scaled = scaler.fit_transform(X_close)
    model_close = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_close.fit(X_close_scaled, y_close)
    
    # Son verilerle tahmin
    last_data = df[features].iloc[-1:].copy()
    last_data_scaled = scaler.transform(last_data)
    
    next_open = model_open.predict(last_data_scaled)[0]
    next_high = model_high.predict(last_data_scaled)[0]
    next_low = model_low.predict(last_data_scaled)[0]
    next_close = model_close.predict(last_data_scaled)[0]
    
    # Volatilite tahmini (son 20 günlük volatilitenin ortalaması ve trendi)
    volatility_trend = df['Volatility'].tail(20).mean()
    if df['Volatility'].tail(5).mean() > df['Volatility'].tail(20).mean():
        volatility_expectation = "Yuksek"
    else:
        volatility_expectation = "Normal"
    
    # Hacim bazlı alternatif senaryolar
    avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
    high_volume_scenario = {
        'open': next_open * 1.01,  # %1 yukarı
        'high': next_high * 1.02,  # %2 yukarı
        'low': next_low * 1.005,   # %0.5 yukarı
        'close': next_close * 1.015 # %1.5 yukarı
    }
    
    low_volume_scenario = {
        'open': next_open * 0.995,  # %0.5 aşağı
        'high': next_high * 0.99,   # %1 aşağı
        'low': next_low * 0.98,     # %2 aşağı
        'close': next_close * 0.99   # %1 aşağı
    }
    
    # Hacim analizi
    current_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio > 1.5:
        volume_trend = "Yuksek Hacim (Ortalamadan %{:.1f} fazla)".format((volume_ratio - 1) * 100)
    elif volume_ratio < 0.5:
        volume_trend = "Dusuk Hacim (Ortalamadan %{:.1f} az)".format((1 - volume_ratio) * 100)
    else:
        volume_trend = "Normal Hacim"
    
    return (next_open, next_high, next_low, next_close, volatility_trend, 
            volatility_expectation, high_volume_scenario, low_volume_scenario, 
            volume_trend, volume_ratio)

def calculate_risk_metrics(returns):
    """Risk metriklerini hesaplar."""
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
    
    return var_95, var_99, sharpe_ratio, max_drawdown

def calculate_fibonacci_levels(high, low):
    """Fibonacci retracement seviyelerini hesaplar."""
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
    """İstatistiksel analizleri gerçekleştirir."""
    # Durağanlık testi (Augmented Dickey-Fuller)
    adf_result = adfuller(df['close'])
    
    # Normallik testi
    _, normality_pvalue = stats.normaltest(df['Daily_Return'].dropna())
    
    # Otokorelasyon
    autocorr = df['Daily_Return'].autocorr()
    
    # ARIMA modeli
    try:
        model = ARIMA(df['close'], order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=1)[0]
    except:
        forecast = None
    
    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'normality_pvalue': normality_pvalue,
        'autocorrelation': autocorr,
        'arima_forecast': forecast
    }

def create_analysis_report(hisse_adi, csv_dosyasi):
    # CSV dosyasının varlığını kontrol et
    if not os.path.exists(csv_dosyasi):
        print(f"Hata: {csv_dosyasi} dosyası bulunamadı!")
        return
    
    try:
        # CSV dosyasından verileri okuyalım
        df = pd.read_csv(csv_dosyasi)
        
        # Veri setinin boş olup olmadığını kontrol et
        if df.empty:
            print("Hata: CSV dosyası boş!")
            return
            
        # Gerekli sütunların varlığını kontrol et
        required_columns = ['time', 'open', 'high', 'low', 'close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Hata: CSV dosyasında şu sütunlar eksik: {missing_columns}")
            print("Gerekli sütunlar: time, open, high, low, close, Volume")
            return
            
        # Veri tiplerini kontrol et
        try:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        except:
            print("Hata: 'time' sütunu doğru formatta değil!")
            return
            
        # Sayısal sütunları kontrol et
        numeric_columns = ['open', 'high', 'low', 'close', 'Volume']
        for col in numeric_columns:
            if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                print(f"Hata: '{col}' sütununda sayısal olmayan değerler var!")
                return
                
        print("Veri seti başarıyla yüklendi.")
        print(f"Toplam kayıt sayısı: {len(df)}")
        print(f"Tarih aralığı: {df['time'].min()} - {df['time'].max()}")
        print("\nVeri seti önizleme:")
        print(df.head())
        print("\nVeri seti istatistikleri:")
        print(df.describe())
        
        # Tarih sütununu datetime formatına çevirelim
        df.set_index('time', inplace=True)
        
        # Temel hesaplamalar
        df['Daily_Return'] = df['close'].pct_change() * 100
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        
        # RSI hesaplama
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI'] = calculate_rsi(df['close'])
        
        # MACD hesaplama
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['close'].rolling(window=20).std()
        
        # Feature engineering
        df['High_Low_Diff'] = df['high'] - df['low']
        df['Open_Close_Diff'] = df['close'] - df['open']
        df['Price_Range'] = (df['high'] - df['low']) / df['close'] * 100
        df['Price_Momentum'] = df['close'].pct_change(5)
        df['Volume_Momentum'] = df['Volume'].pct_change(5) if 'Volume' in df.columns else 0
        
        # Trend Analysis
        df['Trend'] = np.where(df['MA20'] > df['MA50'], 1, -1)
        
        # Tahmin özellikleri
        features = ['open', 'high', 'low', 'close', 'Daily_Return', 'Volatility', 
                    'MA20', 'MA50', 'RSI', 'High_Low_Diff', 'Open_Close_Diff', 
                    'Price_Range', 'Price_Momentum']
        
        # Gelecek gün tahminleri
        (next_open, next_high, next_low, next_close, volatility_trend, 
         volatility_expectation, high_volume_scenario, low_volume_scenario,
         volume_trend, volume_ratio) = predict_next_day_values(df, features)
        
        # İstatistiksel analiz sonuçları
        stat_results = perform_statistical_analysis(df)
        
        # Risk metrikleri
        var_95, var_99, sharpe_ratio, max_drawdown = calculate_risk_metrics(df['Daily_Return'].dropna())
        
        # Fibonacci seviyeleri
        fib_levels = calculate_fibonacci_levels(df['high'].max(), df['low'].min())
        
        # Korelasyon analizi
        correlation_matrix = df[['close', 'Volume', 'Daily_Return', 'Volatility']].corr()
        
        # PDF Rapor Oluşturma
        doc = SimpleDocTemplate(f"{hisse_adi}_analiz_raporu.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        
        # PDF stil tanımlamaları
        styles.add(ParagraphStyle(
            name='TurkishStyle',
            fontName='DejaVuSans',
            fontSize=11,
            leading=16,  # Satır aralığı
            spaceAfter=6,
            bulletIndent=20,
            leftIndent=20
        ))
        
        styles.add(ParagraphStyle(
            name='TurkishHeading',
            fontName='DejaVuSans',  # Kalın font yerine normal font
            fontSize=14,
            leading=16,
            spaceAfter=20,
            spaceBefore=20,
            textColor=colors.HexColor('#2E5090'),
            underline=True  # Altı çizili
        ))
        
        styles.add(ParagraphStyle(
            name='TurkishSubHeading',
            fontName='DejaVuSans',  # Kalın font yerine normal font
            fontSize=12,
            leading=14,
            spaceAfter=10,
            spaceBefore=10,
            textColor=colors.HexColor('#2E5090')
        ))
        
        # Normal stili de Türkçe fontla güncelle
        styles['Normal'].fontName = 'DejaVuSans'
        styles['Normal'].fontSize = 11
        styles['Heading1'].fontName = 'DejaVuSans'
        styles['Heading1'].fontSize = 14
        
        story = []
        
        # Başlık
        story.append(Paragraph(f"{hisse_adi} Hisse Senedi Analiz Raporu", styles['TurkishHeading']))
        story.append(Spacer(1, 12))
        
        # Sonuç ve Tahmin Bölümü
        story.append(Paragraph("1. Sonuc ve Tahminler", styles['TurkishHeading']))
        prediction_info = [
            [Paragraph("Metrik", styles['TurkishHeading']), Paragraph("Tahmin", styles['TurkishHeading'])],
            ["Sonraki Gun Acilis", f"{next_open:.2f} TL"],
            ["Beklenen En Yuksek", f"{next_high:.2f} TL"],
            ["Beklenen En Dusuk", f"{next_low:.2f} TL"],
            ["Beklenen Kapanis", f"{next_close:.2f} TL"],
            ["Volatilite Beklentisi", f"{volatility_expectation} ({volatility_trend:.2f}%)"],
            ["Hacim Durumu", volume_trend],
        ]
        
        t = Table(prediction_info, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Alternatif Senaryolar
        story.append(Paragraph("1.1 Hacim Bazli Alternatif Senaryolar", styles['TurkishHeading']))
        
        scenario_info = [
            [Paragraph("Metrik", styles['TurkishHeading']), 
             Paragraph("Yuksek Hacim Senaryosu", styles['TurkishHeading']),
             Paragraph("Dusuk Hacim Senaryosu", styles['TurkishHeading'])],
            ["Acilis", f"{high_volume_scenario['open']:.2f} TL", f"{low_volume_scenario['open']:.2f} TL"],
            ["En Yuksek", f"{high_volume_scenario['high']:.2f} TL", f"{low_volume_scenario['high']:.2f} TL"],
            ["En Dusuk", f"{high_volume_scenario['low']:.2f} TL", f"{low_volume_scenario['low']:.2f} TL"],
            ["Kapanis", f"{high_volume_scenario['close']:.2f} TL", f"{low_volume_scenario['close']:.2f} TL"],
        ]
        
        t = Table(scenario_info, colWidths=[133, 133, 133])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        
        # Hacim Analizi Açıklaması
        volume_analysis = f"""
        Hacim Analizi:
        • Mevcut Hacim Durumu: {volume_trend}
        • Yuksek Hacim Senaryosu: Normal hacmin en az %50 uzerinde islem gerceklesmesi durumu
        • Dusuk Hacim Senaryosu: Normal hacmin %50'sinin altinda islem gerceklesmesi durumu
        
        Not: Hacim artisi genellikle fiyat hareketinin yonunu guclendirir. Yuksek hacimle gelen 
        yukselis daha guclu bir yukselis trendine, dusuk hacimle gelen yukselis ise zayif bir 
        yukselis trendine isaret edebilir.
        """
        story.append(Paragraph(volume_analysis, styles['TurkishStyle']))
        story.append(Spacer(1, 20))
        
        # Temel Analiz
        story.append(Paragraph("2. Temel Analiz", styles['TurkishHeading']))
        current_price = df['close'].iloc[-1]
        
        basic_info = [
            [Paragraph("Metrik", styles['TurkishHeading']), Paragraph("Deger", styles['TurkishHeading'])],
            ["Son Kapanis", f"{current_price:.2f} TL"],
            ["20 Gunluk Ortalama", f"{df['MA20'].iloc[-1]:.2f} TL"],
            ["50 Gunluk Ortalama", f"{df['MA50'].iloc[-1]:.2f} TL"],
            ["RSI", f"{df['RSI'].iloc[-1]:.2f}"],
            ["Volatilite (20 Gunluk)", f"{df['Volatility'].iloc[-1]:.2f}%"]
        ]
        
        t = Table(basic_info, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Grafikleri oluştur ve ekle
        plt.figure(figsize=(15, 10))
        plt.plot(df.index, df['close'], label='Kapanis Fiyati')
        plt.plot(df.index, df['MA20'], label='20 Gunluk MA')
        plt.plot(df.index, df['MA50'], label='50 Gunluk MA')
        plt.plot(df.index, df['MA200'], label='200 Gunluk MA')
        plt.title(f'{hisse_adi} Hisse Fiyat Grafigi')
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat (TL)')
        plt.legend()
        plt.savefig('price_analysis.png')
        plt.close()
        
        story.append(Paragraph("3. Teknik Analiz Grafikleri", styles['TurkishHeading']))
        story.append(Image('price_analysis.png', 6*inch, 4*inch))
        story.append(Spacer(1, 12))
        
        # İstatistiksel Analiz
        story.append(Paragraph("4. Istatistiksel Analiz", styles['TurkishHeading']))
        daily_returns = df['Daily_Return'].dropna()
        shapiro_stat, shapiro_p = stats.shapiro(daily_returns)
        
        stats_text = f"""
        • Gunluk Getiri Ortalamasi: {daily_returns.mean():.2f}%
        • Gunluk Getiri Std. Sapma: {daily_returns.std():.2f}%
        • Shapiro-Wilk Normallik Testi p-degeri: {shapiro_p:.4f}
        • Carpiklik: {daily_returns.skew():.2f}
        • Basiklik: {daily_returns.kurtosis():.2f}
        """
        story.append(Paragraph(stats_text, styles['TurkishStyle']))
        story.append(Spacer(1, 20))
        
        # Trend ve Momentum Analizi
        story.append(Paragraph("5. Trend ve Momentum Analizi", styles['TurkishHeading']))
        current_trend = "Yukselis" if df['close'].iloc[-1] > df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else \
                       "Yukselis" if df['close'].iloc[-1] > df['MA20'].iloc[-1] else \
                       "Dusus" if df['close'].iloc[-1] < df['MA20'].iloc[-1] < df['MA50'].iloc[-1] else \
                       "Dusus" if df['close'].iloc[-1] < df['MA20'].iloc[-1] else "Yatay"
        
        rsi_status = "Asiri Alim Bolgesi" if df['RSI'].iloc[-1] > 70 else \
                     "Asiri Satim Bolgesi" if df['RSI'].iloc[-1] < 30 else "Notr Bolge"
        
        trend_text = f"""
        • Mevcut Trend: {current_trend}
        • RSI Durumu: {rsi_status} ({df['RSI'].iloc[-1]:.2f})
        • Momentum: {df['Price_Momentum'].iloc[-1]:.2f}
        """
        story.append(Paragraph(trend_text, styles['TurkishStyle']))
        
        # İstatistiksel Analiz Sonuçları
        story.append(Paragraph("Istatistiksel Analiz Sonuclari ve Yorumlari", styles['TurkishHeading']))
        
        # Durağanlık testi yorumu
        durability_explanation = f"""
        <b>Duraganlik Testi (p-değeri: {stat_results['adf_pvalue']:.4f}):</b><br/>
        {'Fiyat serisi duragan degildir ve trend icerir.' if stat_results['adf_pvalue'] > 0.05 else 'Fiyat serisi duragandır.'}<br/>
        <i>Not: Duraganlik, fiyatların belirli bir ortalamaya etrafında dalgalandığını gösterir. 
        Duragan olmayan seriler genellikle trend icerir ve tahmin yapmak daha zordur.</i><br/><br/>
        """
        story.append(Paragraph(durability_explanation, styles['TurkishStyle']))
        
        # Normallik testi yorumu
        normality_explanation = f"""
        <b>Normallik Testi (p-değeri: {stat_results['normality_pvalue']:.4f}):</b><br/>
        {'Getiriler normal dagilimaya uymamaktadır.' if stat_results['normality_pvalue'] < 0.05 else 'Getiriler normal dagilimaya uymaktadır.'}<br/>
        <i>Not: Normal dagilimaya uygunluk, getirilerin tahmin edilebilirliğini ve risk hesaplamalarının guvenilirliğini etkiler.</i><br/><br/>
        """
        story.append(Paragraph(normality_explanation, styles['TurkishStyle']))
        
        # Otokorelasyon yorumu
        autocorr_explanation = f"""
        <b>Otokorelasyon ({stat_results['autocorrelation']:.4f}):</b><br/>
        {'Güçlü bir ardışık ilişki var.' if abs(stat_results['autocorrelation']) > 0.7 else 
        'Orta düzeyde ardışık ilişki var.' if abs(stat_results['autocorrelation']) > 0.3 else 
        'Zayıf bir ardışık ilişki var.'}<br/>
        <i>Not: Otokorelasyon, fiyat hareketlerinin birbirini ne kadar etkilediğini gösterir. 
        Yüksek otokorelasyon, trend takip stratejilerinin işe yarayabileceğine işaret eder.</i><br/><br/>
        """
        story.append(Paragraph(autocorr_explanation, styles['TurkishStyle']))
        
        # ARIMA tahmini yorumu
        if stat_results['arima_forecast']:
            arima_explanation = f"""
            <b>ARIMA Modeli Tahmini:</b><br/>
            Gelecek gün için tahmin edilen fiyat: {stat_results['arima_forecast']:.2f}<br/>
            <i>Not: Bu tahmin, geçmiş fiyat hareketlerinin istatistiksel analizi sonucu elde edilmiştir. 
            Sadece referans amaçlı kullanılmalıdır.</i><br/><br/>
            """
            story.append(Paragraph(arima_explanation, styles['TurkishStyle']))
        
        story.append(Spacer(1, 12))
        
        # Risk Metrikleri
        story.append(Paragraph("Risk Analizi ve Değerlendirmesi", styles['TurkishHeading']))
        
        # VaR açıklaması
        var_explanation = f"""
        <b>Value at Risk (VaR) Analizi:</b><br/>
        • %95 güven aralığında maksimum günlük kayıp: {var_95:.2%}<br/>
        • %99 güven aralığında maksimum günlük kayıp: {var_99:.2%}<br/>
        <i>Not: VaR, normal piyasa koşullarında karşılaşılabilecek maksimum kaybı gösterir. 
        Örneğin, %95 VaR değeri {var_95:.2%} ise, günlük kaybın bu değeri aşma olasılığı %5'tir.</i><br/><br/>
        """
        story.append(Paragraph(var_explanation, styles['TurkishStyle']))
        
        # Sharpe Ratio açıklaması
        sharpe_explanation = f"""
        <b>Sharpe Oranı: {sharpe_ratio:.2f}</b><br/>
        {'Yüksek getiri/risk oranı' if sharpe_ratio > 1 else 
        'Orta düzey getiri/risk oranı' if sharpe_ratio > 0 else 
        'Düşük getiri/risk oranı'}<br/>
        <i>Not: Sharpe oranı, alınan risk başına elde edilen getiriyi ölçer. 
        1'in üzerindeki değerler iyi performansı gösterir.</i><br/><br/>
        """
        story.append(Paragraph(sharpe_explanation, styles['TurkishStyle']))
        
        # Maximum Drawdown açıklaması
        drawdown_explanation = f"""
        <b>Maksimum Düşüş: {max_drawdown:.2%}</b><br/>
        {'Yüksek risk seviyesi' if abs(max_drawdown) > 0.3 else 
        'Orta risk seviyesi' if abs(max_drawdown) > 0.15 else 
        'Düşük risk seviyesi'}<br/>
        <i>Not: Maksimum düşüş, en yüksek noktadan en düşük noktaya kadar olan kayıp yüzdesini gösterir. 
        Bu değer, yatırımcının karşılaşabileceği en kötü senaryoyu temsil eder.</i><br/><br/>
        """
        story.append(Paragraph(drawdown_explanation, styles['TurkishStyle']))
        
        story.append(Spacer(1, 12))
        
        # Fibonacci Seviyeleri
        story.append(Paragraph("Teknik Analiz: Fibonacci Seviyeleri", styles['TurkishHeading']))
        fib_explanation = """
        <b>Fibonacci Seviyeleri Nedir?</b><br/>
        Fibonacci seviyeleri, fiyatın geri çekilme ve ilerleme noktalarını tahmin etmek için kullanılan teknik analiz aracıdır. 
        Bu seviyeler, destek ve direnç noktalarını belirlemede yardımcı olur.<br/><br/>
        <b>Nasıl Yorumlanır?</b><br/>
        • 0.236 ve 0.382: Zayıf destek/direnç seviyeleri<br/>
        • 0.500: Orta güçte destek/direnç seviyesi<br/>
        • 0.618 ve 0.786: Güçlü destek/direnç seviyeleri<br/><br/>
        """
        story.append(Paragraph(fib_explanation, styles['TurkishStyle']))
        
        fib_data = [[level, f"{value:.2f}"] for level, value in fib_levels.items()]
        fib_data.insert(0, ["Fibonacci Seviyesi", "Fiyat Seviyesi"])
        fib_table = Table(fib_data)
        fib_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(fib_table)
        story.append(Spacer(1, 12))
        
        # Trend yönünü belirle
        trend_direction = "Yukselis" if df['close'].iloc[-1] > df['close'].iloc[0] else "Dusus"
        trend_strength = abs(df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        
        # Döngüsellik analizi
        autocorr = df['close'].autocorr(lag=30)  # 30 gunluk otokorelasyon
        is_cyclical = abs(autocorr) > 0.7

        # Momentum gostergeleri
        df['RSI'] = calculate_rsi(df['close'])
        current_rsi = df['RSI'].iloc[-1]
        
        # Volatilite analizi
        volatility = df['close'].pct_change().std() * np.sqrt(252) * 100  # Yillik volatilite
        
        # Önemli bulgular metnini güncelle
        findings = [
            "<b>Önemli Bulgular:</b>",
            "",
            f"• Hisse senedi {trend_direction} trendinde olup, trend gücü %{trend_strength:.2f} seviyesindedir.",
            "",
            f"• Fiyat hareketleri {'döngüsel bir örüntü göstermektedir' if is_cyclical else 'döngüsel bir örüntü göstermemektedir'}.",
            "",
            f"• RSI göstergesi {current_rsi:.2f} seviyesinde olup, hisse {'aşırı alım' if current_rsi > 70 else 'aşırı satım' if current_rsi < 30 else 'nötr'} bölgesindedir.",
            "",
            f"• Yıllık volatilite %{volatility:.2f} seviyesindedir{', bu piyasa ortalamasının üzerindedir' if volatility > 30 else ', bu piyasa ortalamasının altındadır'}.",
            "",
            f"• Getiriler normal dağılıma {'uymaktadır' if stat_results['normality_pvalue'] > 0.05 else 'uymamaktadır'}, risk hesaplamaları {'daha güvenilirdir' if stat_results['normality_pvalue'] > 0.05 else 'ihtiyatla değerlendirilmelidir'}.",
            "",
            f"• Maksimum kayıp riski (VaR 95): %{var_95:.2f}",
            "",
            f"• Risk/getiri oranı (Sharpe): {sharpe_ratio:.2f}"
        ]

        recommendations = [
            "<b>Yatırımcı İçin Öneriler:</b>",
            "",
            f"• {'Güçlü ' if trend_strength > 20 else ''}{trend_direction.lower()}e uygun pozisyon alınabilir.",
            "",
            f"• Hisse {'yüksek' if volatility > 30 else 'orta' if volatility > 20 else 'düşük'} volatiliteye sahiptir. {'Portföyün en fazla %5-10\'luk kısmında pozisyon alınması' if volatility > 30 else 'Portföyün %10-20\'lik kısmında pozisyon alınması' if volatility > 20 else 'Portföyde daha yüksek ağırlıkta tutulabilir'} önerilir.",
            "",
            f"• Stop-loss emirleri için %{var_95:.2f}'lik maksimum kayıp seviyesi referans alınabilir.",
            "",
            f"• {'RSI aşırı alım bölgesinde, kar realizasyonu düşünülebilir.' if current_rsi > 70 else 'RSI aşırı satım bölgesinde, alım fırsatı olabilir.' if current_rsi < 30 else 'RSI nötr bölgede, trend yönünde pozisyon alınabilir.'}",
            "",
            "• Fibonacci seviyeleri ve destek/direnç noktaları alım-satım kararları için referans olarak kullanılabilir."
        ]

        # PDF raporuna ekle
        story.append(Paragraph("Genel Değerlendirme ve Özet", styles['TurkishHeading']))
        story.append(Paragraph(f"Risk Seviyesi: {'Yüksek' if volatility > 30 or abs(sharpe_ratio) > 2 else 'Orta' if volatility > 20 or abs(sharpe_ratio) > 1 else 'Düşük'}", styles['TurkishSubHeading']))
        story.append(Paragraph(f"Performans Değerlendirmesi: {'Güçlü' if sharpe_ratio > 1 else 'Zayıf' if sharpe_ratio < 0 else 'Orta'}", styles['TurkishSubHeading']))
        story.append(Spacer(1, 12))
        
        # Önemli bulgular ve önerileri ayrı ayrı paragraflar halinde ekle
        for line in findings:
            story.append(Paragraph(line, styles['TurkishStyle']))
        
        story.append(Spacer(1, 12))
        
        for line in recommendations:
            story.append(Paragraph(line, styles['TurkishStyle']))
        
        # Durağanlık analizi ve spesifik öneriler ekle
        story.append(Paragraph("Durağanlık Analizi ve Spesifik Öneriler", styles['TurkishHeading']))
        
        # ADX hesaplama
        def calculate_adx(high, low, close, period=14):
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (abs(minus_dm.rolling(window=period).mean()) / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            return adx
        
        # ROC hesaplama
        df['ROC'] = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100
        
        # ADX hesaplama
        df['ADX'] = calculate_adx(df['high'], df['low'], df['close'])
        current_adx = df['ADX'].iloc[-1]
        
        # MFI hesaplama
        def calculate_mfi(high, low, close, volume, period=14):
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
            
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))
            return mfi
        
        df['MFI'] = calculate_mfi(df['high'], df['low'], df['close'], df['Volume'])
        current_mfi = df['MFI'].iloc[-1]
        current_roc = df['ROC'].iloc[-1]
        
        # Hacim analizi
        volume_sma = df['Volume'].rolling(window=20).mean()
        volume_trend = (df['Volume'].iloc[-1] / volume_sma.iloc[-1] - 1) * 100
        
        # Volatilite analizi
        volatility_change = (df['Volatility'].iloc[-1] / df['Volatility'].mean() - 1) * 100
        
        recommendations = [
            "<b>1. Trend Bazli Oneriler (ADX):</b>",
            f"• {'Güçlü trend devam ediyor, mevcut pozisyonlar korunabilir.' if current_adx > 25 else 'Trend zayıf, pozisyon almak için trend güçlenene kadar beklenebilir.'}",
            "",
            "<b>2. Momentum Bazli Oneriler (ROC):</b>",
            f"• {'Momentum pozitif ve güçlü, kademeli alım yapılabilir.' if current_roc > 2 else 'Momentum zayıf, yeni pozisyon açmak için acele edilmemeli.' if current_roc < -2 else 'Momentum nötr, mevcut seviyelerde beklenebilir.'}",
            "",
            "<b>3. Hacim Bazli Oneriler:</b>",
            f"• {'Artan hacim trendi destekliyor, pozisyonlar güçlendirilebilir.' if volume_trend > 20 and current_roc > 0 else 'Hacim trendi desteklemiyor, temkinli olunmalı.'}",
            "",
            "<b>4. Volatilite Bazli Risk Yonetimi:</b>",
            f"• {'Yükselen volatilite nedeniyle stop-loss seviyeleri daraltılabilir.' if volatility_change > 20 else 'Volatilite normal seviyelerde, standart risk yönetimi uygulanabilir.'}",
            "",
            "<b>5. Teknik Seviye Onerileri:</b>",
            f"• Stop-loss için {fib_levels['Seviye 0.382']:.2f} seviyesi kullanılabilir.",
            f"• Kar realizasyonu için {fib_levels['Seviye 0.618']:.2f} - {fib_levels['Seviye 0.786']:.2f} aralığı hedeflenebilir.",
            "",
            "<b>6. Risk Yonetimi Onerileri:</b>",
            "• Portföy büyüklüğünün %2'sinden fazla risk alınmaması önerilir.",
            f"• {'Volatilite yüksek olduğundan pozisyon büyüklüğü küçük tutulmalı.' if volatility_change > 20 else 'Normal pozisyon büyüklükleri korunabilir.'}",
            "",
            "<b>7. MFI Bazli Alim/Satim Onerileri:</b>",
            f"• {'MFI aşırı alım bölgesinde, kar realizasyonu düşünülebilir.' if current_mfi > 80 else 'MFI aşırı satım bölgesinde, kademeli alım fırsatı olabilir.' if current_mfi < 20 else 'MFI nötr bölgede, mevcut strateji korunabilir.'}"
        ]
        
        for line in recommendations:
            story.append(Paragraph(line, styles['TurkishStyle']))
        
        # PDF oluştur
        doc.build(story)
        
        # Geçici dosyaları temizle
        if os.path.exists('price_analysis.png'):
            os.remove('price_analysis.png')
        
        print(f"\n{hisse_adi}_analiz_raporu.pdf dosyasi olusturuldu.")
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return

if __name__ == "__main__":
    hisse_adi = input("Analiz edilecek hisse adini girin (orn: PAPIL): ")
    csv_dosyasi = input("CSV dosyasinin adini girin: ")
    
    # Dosya uzantısını kontrol et
    if not csv_dosyasi.endswith('.csv'):
        csv_dosyasi += '.csv'
    
    # Dosya yolu kontrolü
    if not os.path.isabs(csv_dosyasi):
        csv_dosyasi = os.path.join(os.getcwd(), csv_dosyasi)
    
    create_analysis_report(hisse_adi, csv_dosyasi)
