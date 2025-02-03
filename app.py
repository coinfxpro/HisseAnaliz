import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import yfinance as yf
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

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
    # Özellik seçimi
    features = ['open', 'high', 'low', 'close', 'Volume', 'MA20', 'MA50', 'MA200', 'RSI', 'Daily_Return']
    X = df[features].values[:-1]  # Son günü tahmin için kullanacağız
    y_close = df['close'].values[1:]  # Bir sonraki günün kapanışı
    y_high = df['high'].values[1:]    # Bir sonraki günün en yükseği
    y_low = df['low'].values[1:]      # Bir sonraki günün en düşüğü
    
    # Veri setini bölme
    X_train, X_test, y_train_close, y_test_close = train_test_split(X[:-1], y_close[:-1], test_size=0.2, random_state=42)
    _, _, y_train_high, y_test_high = train_test_split(X[:-1], y_high[:-1], test_size=0.2, random_state=42)
    _, _, y_train_low, y_test_low = train_test_split(X[:-1], y_low[:-1], test_size=0.2, random_state=42)
    
    # Model eğitimi - Kapanış
    model_close = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_close.fit(X_train, y_train_close)
    
    # Model eğitimi - En Yüksek
    model_high = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_high.fit(X_train, y_train_high)
    
    # Model eğitimi - En Düşük
    model_low = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_low.fit(X_train, y_train_low)
    
    # Son günün verilerini kullanarak tahmin
    last_day = X[-1].reshape(1, -1)
    
    predictions = {
        'close': model_close.predict(last_day)[0],
        'high': model_high.predict(last_day)[0],
        'low': model_low.predict(last_day)[0],
        'open': df['close'].iloc[-1]  # Bir sonraki günün açılışı son kapanış olarak tahmin edilir
    }
    
    return predictions

def generate_alternative_scenarios(df, predictions):
    # Temel tahminler
    base_predictions = predictions.copy()
    
    # Yüksek hacim senaryosu (normal hacmin 1.5 katı)
    high_volume_scenario = {
        'open': base_predictions['open'] * 1.01,  # %1 daha yüksek
        'high': base_predictions['high'] * 1.02,  # %2 daha yüksek
        'low': base_predictions['low'] * 0.995,   # %0.5 daha düşük
        'close': base_predictions['close'] * 1.015 # %1.5 daha yüksek
    }
    
    # Düşük hacim senaryosu (normal hacmin yarısı)
    low_volume_scenario = {
        'open': base_predictions['open'] * 0.995,  # %0.5 daha düşük
        'high': base_predictions['high'] * 0.99,   # %1 daha düşük
        'low': base_predictions['low'] * 0.98,     # %2 daha düşük
        'close': base_predictions['close'] * 0.99   # %1 daha düşük
    }
    
    # Hacim durumu analizi
    avg_volume = df['Volume'].mean()
    current_volume = df['Volume'].iloc[-1]
    volume_change = ((current_volume - avg_volume) / avg_volume) * 100
    
    volume_status = "Düşük Hacim" if volume_change < -25 else "Yüksek Hacim" if volume_change > 25 else "Normal Hacim"
    
    scenarios = {
        'Temel': base_predictions,
        'Yüksek_Hacim': high_volume_scenario,
        'Düşük_Hacim': low_volume_scenario,
        'Hacim_Durumu': {
            'Durum': volume_status,
            'Değişim': volume_change
        }
    }
    
    return scenarios

def analyze_volume_scenarios(df, predictions):
    avg_volume = df['Volume'].mean()
    current_volume = df['Volume'].iloc[-1]
    volume_change = ((current_volume - avg_volume) / avg_volume) * 100
    
    # Hacim bazlı senaryolar
    scenarios = {
        'Yüksek Hacim': {
            'open': predictions['close'] * 1.01,  # %1 daha yüksek
            'high': predictions['high'] * 1.02,   # %2 daha yüksek
            'low': predictions['low'] * 0.995,    # %0.5 daha düşük
            'close': predictions['close'] * 1.015 # %1.5 daha yüksek
        },
        'Düşük Hacim': {
            'open': predictions['close'] * 0.995,  # %0.5 daha düşük
            'high': predictions['high'] * 0.99,    # %1 daha düşük
            'low': predictions['low'] * 0.98,      # %2 daha düşük
            'close': predictions['close'] * 0.99    # %1 daha düşük
        }
    }
    
    volume_status = "Düşük Hacim" if volume_change < -25 else "Yüksek Hacim" if volume_change > 25 else "Normal Hacim"
    volume_percentage = abs(volume_change)
    
    return scenarios, volume_status, volume_percentage

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
    volume_avg = df['Volume'].mean()
    current_volume = df['Volume'].iloc[-1]
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
        'Tahmin': f"{'YÜKSELİŞ 📈' if predictions['close'] > df['close'].iloc[-1] else 'DÜŞÜŞ 📉'} (₺{predictions['close']:.2f})",
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
        ('close', 'Volume'),
        ('close', 'RSI'),
        ('Volume', 'Daily_Return'),
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
    if var1 == 'close' and var2 == 'Volume':
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
    
    elif var1 == 'Volume' and var2 == 'Daily_Return':
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
        y=df['Volume'],
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

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="Hisse Senedi Analizi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sayfa stili
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

        # 2. TEKNİK ANALİZ GRAFİKLERİ
        st.header("2. TEKNİK ANALİZ GRAFİKLERİ")
        
        # 2.1 Mum Grafiği ve Hacim Analizi
        st.subheader("2.1 Mum Grafiği ve Hacim Analizi")
        
        # Mum grafiği
        fig_candlestick = create_candlestick_chart(df)
        st.plotly_chart(fig_candlestick)
        
        # Teknik analiz yorumları
        current_price = df['close'].iloc[-1]
        ma20_last = df['MA20'].iloc[-1]
        ma50_last = df['MA50'].iloc[-1]
        ma200_last = df['MA200'].iloc[-1]
        
        trend_analysis = f"""
        **Trend Analizi:**
        - **Kısa Vadeli (MA20):** {"Yükseliş" if current_price > ma20_last else "Düşüş"} 
          - Fiyat: ₺{current_price:.2f}, MA20: ₺{ma20_last:.2f}
          - {"%{:.1f} {} MA20'den" .format(abs((current_price/ma20_last-1)*100), "yukarıda" if current_price > ma20_last else "aşağıda")}
        
        - **Orta Vadeli (MA50):** {"Yükseliş" if current_price > ma50_last else "Düşüş"}
          - MA50: ₺{ma50_last:.2f}
          - {"%{:.1f} {} MA50'den" .format(abs((current_price/ma50_last-1)*100), "yukarıda" if current_price > ma50_last else "aşağıda")}
        
        - **Uzun Vadeli (MA200):** {"Yükseliş" if current_price > ma200_last else "Düşüş"}
          - MA200: ₺{ma200_last:.2f}
          - {"%{:.1f} {} MA200'den" .format(abs((current_price/ma200_last-1)*100), "yukarıda" if current_price > ma200_last else "aşağıda")}
        
        **Trend Gücü:** {trend_gucu}
        """.format(
            trend_gucu="GÜÇLÜ 💪" if all([current_price > ma20_last > ma50_last > ma200_last]) else 
                      "ORTA 👍" if current_price > ma20_last and current_price > ma50_last else 
                      "ZAYIF 👎" if current_price < ma20_last and current_price < ma50_last else 
                      "BELİRSİZ ⚠️"
        )

        st.markdown(trend_analysis)
        
        # Hacim grafiği ve analizi
        fig_volume = create_volume_chart(df)
        st.plotly_chart(fig_volume)
        
        # Hacim analizi
        avg_volume = df['Volume'].mean()
        current_volume = df['Volume'].iloc[-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        
        volume_analysis = f"""
        **Hacim Analizi:**
        - **Günlük Hacim:** {current_volume:,.0f}
        - **Ortalama Hacim:** {avg_volume:,.0f}
        - **Hacim Değişimi:** %{volume_change:.1f} ({current_volume/avg_volume:.1f}x)
        
        **Hacim Durumu:** {
            "🔥 ÇOK YÜKSEK - Güçlü alıcı/satıcı ilgisi" if volume_change > 100 else
            "📈 YÜKSEK - Artan ilgi" if volume_change > 50 else
            "➡️ NORMAL - Ortalama ilgi" if volume_change > -25 else
            "📉 DÜŞÜK - Azalan ilgi" if volume_change > -50 else
            "⚠️ ÇOK DÜŞÜK - İlgi kaybı"
        }
        
        **Yorum:** {
            "Çok yüksek hacim, fiyat hareketinin güvenilirliğini artırıyor." if volume_change > 100 else
            "Ortalamanın üzerinde hacim, trend yönünü destekliyor." if volume_change > 50 else
            "Normal hacim seviyeleri, standart piyasa aktivitesi." if volume_change > -25 else
            "Düşük hacim, trend gücünün zayıf olduğunu gösteriyor." if volume_change > -50 else
            "Çok düşük hacim, piyasa ilgisinin azaldığını gösteriyor."
        }
        """
        
        st.markdown(volume_analysis)
        
        # 2.2 Teknik Göstergeler ve Yorumları
        st.subheader("2.2 Teknik Göstergeler")
        
        # RSI ve MACD grafikleri
        rsi_fig, macd_fig = create_technical_charts(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(rsi_fig)
            
            # RSI yorumu
            current_rsi = df['RSI'].iloc[-1]
            rsi_analysis = f"""
            **RSI Analizi (14 günlük):**
            - **Mevcut RSI:** {current_rsi:.1f}
            - **Durum:** {
                "💹 AŞIRI ALIM - Satış fırsatı" if current_rsi > 70 else
                "📉 AŞIRI SATIM - Alım fırsatı" if current_rsi < 30 else
                "➡️ NÖTR - Normal seviyeler"
            }
            
            **Yorum:** {
                "Aşırı alım bölgesinde, düzeltme gelebilir." if current_rsi > 70 else
                "Aşırı satım bölgesinde, tepki yükselişi gelebilir." if current_rsi < 30 else
                "RSI nötr bölgede, trend yönünde hareket devam edebilir."
            }
            """
            st.markdown(rsi_analysis)
            
        with col2:
            st.plotly_chart(macd_fig)
            
            # MACD yorumu
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['Signal_Line'].iloc[-1]
            macd_cross = "AL" if current_macd > current_signal else "SAT"
            
            macd_analysis = f"""
            **MACD Analizi:**
            - **MACD:** {current_macd:.3f}
            - **Sinyal:** {current_signal:.3f}
            - **Sinyal:** {
                "🟢 AL - MACD, sinyal çizgisinin üzerinde" if macd_cross == "AL" else
                "🔴 SAT - MACD, sinyal çizgisinin altında"
            }
            
            **Yorum:** {
                "Yükseliş momentumu devam ediyor." if macd_cross == "AL" and current_macd > 0 else
                "Zayıf bir yükseliş sinyali var." if macd_cross == "AL" and current_macd < 0 else
                "Düşüş momentumu devam ediyor." if macd_cross == "SAT" and current_macd < 0 else
                "Zayıf bir düşüş sinyali var."
            }
            """
            st.markdown(macd_analysis)

        # 3. İSTATİSTİKSEL ANALİZ
        st.header("3. İSTATİSTİKSEL ANALİZ")
        
        # 3.1 Temel İstatistikler
        st.subheader("3.1 Temel İstatistikler")
        
        # Temel istatistikler
        basic_stats = df[['close', 'Volume', 'Daily_Return']].describe()
        st.dataframe(basic_stats)
        
        # İstatistik yorumları
        mean_price = df['close'].mean()
        std_price = df['close'].std()
        price_cv = std_price / mean_price  # Değişim katsayısı
        
        stats_analysis = f"""
        **Fiyat İstatistikleri:**
        - **Ortalama Fiyat:** ₺{mean_price:.2f}
        - **Standart Sapma:** ₺{std_price:.2f}
        - **Değişim Katsayısı:** {price_cv:.2f}
        - **Volatilite Seviyesi:** {
            "🔥 ÇOK YÜKSEK" if price_cv > 0.5 else
            "📈 YÜKSEK" if price_cv > 0.3 else
            "➡️ NORMAL" if price_cv > 0.1 else
            "📉 DÜŞÜK"
        }
        
        **Getiri İstatistikleri:**
        - **Ortalama Getiri:** %{df['Daily_Return'].mean():.2f}
        - **Maksimum Yükseliş:** %{df['Daily_Return'].max():.2f}
        - **Maksimum Düşüş:** %{df['Daily_Return'].min():.2f}
        - **Pozitif Getiri Günleri:** %{(df['Daily_Return'] > 0).mean()*100:.1f}
        
        **Hacim İstatistikleri:**
        - **Ortalama Hacim:** {df['Volume'].mean():,.0f}
        - **Maksimum Hacim:** {df['Volume'].max():,.0f}
        - **Minimum Hacim:** {df['Volume'].min():,.0f}
        """
        
        st.markdown(stats_analysis)
        
        # 3.2 Risk Metrikleri
        st.subheader("3.2 Risk Analizi")
        risk_metrics = calculate_risk_metrics(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Yıllık Volatilite", f"%{risk_metrics['Volatilite']*100:.2f}")
        with col2:
            st.metric("Sharpe Oranı", f"{risk_metrics['Sharpe Oranı']:.2f}")
        with col3:
            st.metric("Maximum Drawdown", f"%{risk_metrics['Max Drawdown']*100:.2f}")
        
        risk_analysis = f"""
        **Risk Analizi Sonuçları:**
        
        1. **Volatilite Analizi:**
           - Yıllık Volatilite: %{risk_metrics['Volatilite']*100:.2f}
           - Durum: {
               "🔥 ÇOK RİSKLİ - Yüksek oynaklık" if risk_metrics['Volatilite'] > 0.4 else
               "⚠️ RİSKLİ - Artan oynaklık" if risk_metrics['Volatilite'] > 0.25 else
               "ℹ️ NORMAL - Standart oynaklık" if risk_metrics['Volatilite'] > 0.15 else
               "✅ DÜŞÜK RİSK - Düşük oynaklık"
           }
        
        2. **Sharpe Oranı Analizi:**
           - Sharpe Oranı: {risk_metrics['Sharpe Oranı']:.2f}
           - Yorum: {
               "🌟 MÜKEMMEL - Risk/getiri oranı çok iyi" if risk_metrics['Sharpe Oranı'] > 2 else
               "✅ İYİ - Pozitif risk/getiri oranı" if risk_metrics['Sharpe Oranı'] > 1 else
               "ℹ️ NORMAL - Kabul edilebilir risk/getiri" if risk_metrics['Sharpe Oranı'] > 0 else
               "⚠️ ZAYIF - Negatif risk/getiri oranı"
           }
        
        3. **Value at Risk (VaR) Analizi:**
           - VaR (95%): %{risk_metrics['VaR_95']*100:.2f}
           - VaR (99%): %{risk_metrics['VaR_99']*100:.2f}
           - Yorum: Bir günde %95 olasılıkla maksimum %{abs(risk_metrics['VaR_95']*100):.1f} kayıp beklentisi
        
        4. **Maximum Drawdown Analizi:**
           - Maximum Drawdown: %{risk_metrics['Max Drawdown']*100:.2f}
           - Durum: {
               "⚠️ YÜKSEK KAYIP RİSKİ" if risk_metrics['Max Drawdown'] < -0.3 else
               "⚡ DİKKAT" if risk_metrics['Max Drawdown'] < -0.2 else
               "ℹ️ NORMAL" if risk_metrics['Max Drawdown'] < -0.1 else
               "✅ DÜŞÜK KAYIP"
           }
        
        **Risk Yönetimi Önerileri:**
        1. Stop-Loss: %{abs(risk_metrics['VaR_95']*100):.1f} altında belirlenmeli
        2. Position Sizing: {
            "Küçük pozisyonlar önerilir" if risk_metrics['Volatilite'] > 0.3 else
            "Orta büyüklükte pozisyonlar alınabilir" if risk_metrics['Volatilite'] > 0.2 else
            "Normal pozisyon büyüklüğü uygun"
        }
        3. Takip: {
            "Çok yakın takip gerekli" if risk_metrics['Volatilite'] > 0.3 else
            "Günlük takip önerilir" if risk_metrics['Volatilite'] > 0.2 else
            "Standart takip yeterli"
        }
        """
        
        st.markdown(risk_analysis)
        
        # 3.3 İstatistiksel Testler
        st.subheader("3.3 İstatistiksel Testler")
        stats_results = perform_statistical_analysis(df)
        
        # Test sonuçları
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Durağanlık Testi (ADF)", 
                     f"p-değeri: {stats_results['ADF p-değeri']:.4f}")
            st.metric("Normallik Testi", 
                     f"p-değeri: {stats_results['Normallik p-değeri']:.4f}")
        
        with col2:
            st.metric("Otokorelasyon", 
                     f"{stats_results['Otokorelasyon']:.4f}")
            if stats_results['ARIMA Tahmini'] is not None:
                st.metric("ARIMA Tahmini", 
                         f"₺{stats_results['ARIMA Tahmini']:.2f}")
        
        # Test yorumları
        test_analysis = """
        **İstatistiksel Test Sonuçları:**
        
        1. **Durağanlık Analizi (ADF Testi):**
           - p-değeri: {:.4f}
           - Sonuç: {}
           - Yorum: {}
        
        2. **Normallik Testi:**
           - p-değeri: {:.4f}
           - Sonuç: {}
           - Yorum: {}
        
        3. **Otokorelasyon Analizi:**
           - Katsayı: {:.4f}
           - Sonuç: {}
           - Yorum: {}
        
        4. **Mevsimsellik Analizi:**
           - Sonuç: {}
           - Yorum: {}
        """.format(
            stats_results['ADF p-değeri'],
            "❌ DURAĞAN DEĞİL" if stats_results['ADF p-değeri'] > 0.05 else "✅ DURAĞAN",
            "Fiyat serisi trend içeriyor, teknik analiz için fark alınmalı" if stats_results['ADF p-değeri'] > 0.05 else "Fiyat serisi durağan, doğrudan analiz edilebilir",
            
            stats_results['Normallik p-değeri'],
            "❌ NORMAL DAĞILIM DEĞİL" if stats_results['Normallik p-değeri'] < 0.05 else "✅ NORMAL DAĞILIM",
            "Ekstrem hareketler normalden fazla, risk yönetimi önemli" if stats_results['Normallik p-değeri'] < 0.05 else "Fiyat hareketleri normal dağılıma uyuyor",
            
            stats_results['Otokorelasyon'],
            "GÜÇLÜ İLİŞKİ" if abs(stats_results['Otokorelasyon']) > 0.7 else "ORTA İLİŞKİ" if abs(stats_results['Otokorelasyon']) > 0.3 else "ZAYIF İLİŞKİ",
            "Geçmiş fiyatlar gelecek tahmini için kullanılabilir" if abs(stats_results['Otokorelasyon']) > 0.5 else "Geçmiş fiyatlar zayıf gösterge",
            
            "MEVSİMSELLİK VAR" if stats_results['Mevsimsellik'] is not None and abs(stats_results['Mevsimsellik']) > 0.1 else "MEVSİMSELLİK YOK",
            "Belirli dönemlerde tekrarlayan hareketler mevcut" if stats_results['Mevsimsellik'] is not None and abs(stats_results['Mevsimsellik']) > 0.1 else "Belirgin dönemsel hareket yok"
        )
        
        st.markdown(test_analysis)

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

        # 5. GELECEK TAHMİNLERİ VE SENARYOLAR
        st.header("5. GELECEK TAHMİNLERİ VE SENARYOLAR")
        
        # 5.1 Temel Tahminler
        st.subheader("5.1 Temel Tahminler")
        predictions = predict_next_day_values(df)
        scenarios = generate_alternative_scenarios(df, predictions)
        
        # Tahmin özet tablosu
        pred_df = pd.DataFrame({
            'Metrik': ['Açılış', 'En Yüksek', 'En Düşük', 'Kapanış'],
            'Tahmin': [
                f"₺{predictions['open']:.2f}",
                f"₺{predictions['high']:.2f}",
                f"₺{predictions['low']:.2f}",
                f"₺{predictions['close']:.2f}"
            ],
            'Değişim (%)': [
                f"%{((predictions['open']/df['close'].iloc[-1])-1)*100:.1f}",
                f"%{((predictions['high']/df['close'].iloc[-1])-1)*100:.1f}",
                f"%{((predictions['low']/df['close'].iloc[-1])-1)*100:.1f}",
                f"%{((predictions['close']/df['close'].iloc[-1])-1)*100:.1f}"
            ]
        })
        
        st.table(pred_df)
        
        # Tahmin yorumları
        pred_change = ((predictions['close'] / df['close'].iloc[-1]) - 1) * 100
        pred_range = ((predictions['high'] - predictions['low']) / predictions['low']) * 100
        
        prediction_analysis = f"""
        **Tahmin Analizi:**
        
        1. **Genel Görünüm:**
           - Beklenen Yön: {"🟢 YÜKSELİŞ" if pred_change > 1 else "🔴 DÜŞÜŞ" if pred_change < -1 else "⚪ YATAY"}
           - Beklenen Değişim: %{pred_change:.1f}
           - Fiyat Aralığı: ₺{predictions['low']:.2f} - ₺{predictions['high']:.2f} (%{pred_range:.1f})
        
        2. **Güven Analizi:**
           - Trend Gücü: {
               "💪 GÜÇLÜ" if abs(pred_change) > 3 else
               "👍 ORTA" if abs(pred_change) > 1 else
               "👎 ZAYIF"
           }
           - Tahmin Güvenilirliği: {
               "⭐⭐⭐ YÜKSEK" if scenarios['Hacim_Durumu']['Durum'] == "Yüksek Hacim" and abs(pred_change) > 2 else
               "⭐⭐ ORTA" if scenarios['Hacim_Durumu']['Durum'] == "Normal Hacim" or abs(pred_change) > 1 else
               "⭐ DÜŞÜK"
           }
        
        3. **Destek/Direnç Seviyeleri:**
           - Güçlü Direnç: ₺{predictions['high']:.2f}
           - Zayıf Direnç: ₺{(predictions['high'] + predictions['close'])/2:.2f}
           - Zayıf Destek: ₺{(predictions['low'] + predictions['close'])/2:.2f}
           - Güçlü Destek: ₺{predictions['low']:.2f}
        """
        
        st.markdown(prediction_analysis)
        
        # 5.2 Alternatif Senaryolar
        st.subheader("5.2 Hacim Bazlı Senaryolar")
        
        # Hacim durumu analizi
        volume_status = scenarios['Hacim_Durumu']['Durum']
        volume_change = scenarios['Hacim_Durumu']['Değişim']
        
        st.info(f"Mevcut Hacim Durumu: {volume_status} (Ortalamadan %{volume_change:.1f} {'fazla' if volume_change > 0 else 'az'})")
        
        # Senaryo tablosu
        scenario_df = pd.DataFrame({
            'Metrik': ['Açılış', 'En Yüksek', 'En Düşük', 'Kapanış'],
            'Yüksek Hacim': [
                f"₺{scenarios['Yüksek_Hacim']['open']:.2f}",
                f"₺{scenarios['Yüksek_Hacim']['high']:.2f}",
                f"₺{scenarios['Yüksek_Hacim']['low']:.2f}",
                f"₺{scenarios['Yüksek_Hacim']['close']:.2f}"
            ],
            'Düşük Hacim': [
                f"₺{scenarios['Düşük_Hacim']['open']:.2f}",
                f"₺{scenarios['Düşük_Hacim']['high']:.2f}",
                f"₺{scenarios['Düşük_Hacim']['low']:.2f}",
                f"₺{scenarios['Düşük_Hacim']['close']:.2f}"
            ]
        })
        
        st.table(scenario_df)
        
        # Senaryo yorumları
        scenario_analysis = f"""
        **Senaryo Analizi:**
        
        1. **Yüksek Hacim Senaryosu:**
           - Beklenen Hareket: {"Güçlü Yükseliş 📈" if pred_change > 0 else "Güçlü Düşüş 📉"}
           - Hedef Fiyat: ₺{scenarios['Yüksek_Hacim']['close']:.2f} (%{((scenarios['Yüksek_Hacim']['close']/df['close'].iloc[-1])-1)*100:.1f})
           - Olasılık: {"Yüksek ⭐⭐⭐" if volume_status == "Yüksek Hacim" else "Düşük ⭐"}
        
        2. **Düşük Hacim Senaryosu:**
           - Beklenen Hareket: {"Zayıf Yükseliş ↗️" if pred_change > 0 else "Zayıf Düşüş ↘️"}
           - Hedef Fiyat: ₺{scenarios['Düşük_Hacim']['close']:.2f} (%{((scenarios['Düşük_Hacim']['close']/df['close'].iloc[-1])-1)*100:.1f})
           - Olasılık: {"Yüksek ⭐⭐⭐" if volume_status == "Düşük Hacim" else "Düşük ⭐"}
        
        **Pozisyon Önerileri:**
        1. Stop-Loss: ₺{scenarios['Düşük_Hacim']['low']:.2f} (%{((scenarios['Düşük_Hacim']['low']/df['close'].iloc[-1])-1)*100:.1f})
        2. İlk Hedef: ₺{scenarios['Yüksek_Hacim']['close']:.2f} (%{((scenarios['Yüksek_Hacim']['close']/df['close'].iloc[-1])-1)*100:.1f})
        3. Maksimum Hedef: ₺{scenarios['Yüksek_Hacim']['high']:.2f} (%{((scenarios['Yüksek_Hacim']['high']/df['close'].iloc[-1])-1)*100:.1f})
        
        **Risk/Getiri Oranı:** {abs(((scenarios['Yüksek_Hacim']['close']/df['close'].iloc[-1])-1) / ((scenarios['Düşük_Hacim']['low']/df['close'].iloc[-1])-1)):.1f}
        """
        
        st.markdown(scenario_analysis)
        
        # ARIMA tahmini varsa göster
        if stats_results['ARIMA Tahmini'] is not None:
            arima_change = ((stats_results['ARIMA Tahmini'] / df['close'].iloc[-1]) - 1) * 100
            st.info(f"""
            **ARIMA Modeli Tahmini:**
            - Fiyat: ₺{stats_results['ARIMA Tahmini']:.2f}
            - Değişim: %{arima_change:.1f}
            - Uyum: {"✅ Diğer tahminlerle uyumlu" if (arima_change > 0) == (pred_change > 0) else "⚠️ Diğer tahminlerle çelişiyor"}
            """)

        # 6. ANALİZ ÖZET VE YORUMLAR
        st.header("6. ANALİZ ÖZET VE YORUMLAR")
        
        summary = generate_analysis_summary(df, predictions, risk_metrics, stats_results)
        
        # Genel Durum
        st.subheader("6.1 Genel Durum")
        
        # Ana metrikler
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Genel Trend", summary['Genel Trend'])
        with col2:
            st.metric("Risk Durumu", summary['Risk Durumu'])
        with col3:
            st.metric("MACD Sinyali", summary['MACD Sinyali'])
        with col4:
            st.metric("Bollinger", summary['Bollinger'])
            
        # Teknik Göstergeler
        st.subheader("6.2 Teknik Göstergeler")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("RSI", summary['RSI Durumu'])
        with col2:
            st.metric("Volatilite", summary['Volatilite'])
        with col3:
            st.metric("Hacim", summary['Hacim Durumu'])
        
        # Hareketli Ortalamalar
        st.subheader("6.3 Hareketli Ortalamalar")
        ma_cols = st.columns(3)
        for i, (ma_name, ma_value) in enumerate(summary['Teknik Göstergeler'].items()):
            with ma_cols[i]:
                st.metric(ma_name, ma_value)
        
        # Tahmin ve Risk
        st.subheader("6.4 Tahmin ve Risk Analizi")
        pred_cols = st.columns(3)
        with pred_cols[0]:
            st.metric("Yarınki Tahmin", summary['Tahmin'])
        with pred_cols[1]:
            st.metric("Sharpe Oranı", summary['Sharpe'])
        with pred_cols[2]:
            var_value = f"%{abs(risk_metrics['VaR_95']*100):.1f} kayıp riski"
            st.metric("VaR (%95)", var_value)

        # 7. KORELASYON ANALİZİ
        st.header("7. KORELASYON ANALİZİ")
        
        # Korelasyon matrisi
        corr_matrix = df[['open', 'high', 'low', 'close', 'Volume', 'Daily_Return', 'RSI']].corr()
        
        # Korelasyon haritası
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Korelasyon Matrisi')
        st.pyplot(fig_corr)
        
        # Korelasyon analizi ve yorumları
        st.subheader("7.1 Korelasyon Analizi Yorumları")
        
        correlations = analyze_correlation_matrix(corr_matrix)
        
        for corr in correlations:
            st.write(f"**{corr['pair']}:** {corr['interpretation']} (Korelasyon: {corr['correlation']:.2f}, {corr['strength']})")
        
        st.markdown("""
        **Korelasyon Analizi Özeti:**
        1. **Hacim-Fiyat İlişkisi:** {}
        2. **Momentum Durumu:** {}
        3. **Volatilite Etkisi:** {}
        """.format(
            "Güçlü" if abs(corr_matrix.loc['close', 'Volume']) > 0.5 else "Zayıf",
            "Trend devam ediyor" if corr_matrix.loc['close', 'RSI'] > 0.7 else "Trend zayıflıyor",
            "Yüksek" if abs(corr_matrix.loc['Daily_Return', 'Volume']) > 0.3 else "Düşük"
        ))

        # 8. İSTATİSTİKSEL ANALİZ
        st.header("8. İSTATİSTİKSEL ANALİZ")
        
        # İstatistiksel test sonuçları
        st.subheader("8.1 İstatistiksel Test Sonuçları")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Durağanlık Testi (ADF)", 
                     f"p-değeri: {stats_results['ADF p-değeri']:.4f}")
            st.metric("Normallik Testi", 
                     f"p-değeri: {stats_results['Normallik p-değeri']:.4f}")
        
        with col2:
            st.metric("Otokorelasyon", 
                     f"{stats_results['Otokorelasyon']:.4f}")
            st.metric("Çarpıklık", 
                     f"{stats_results['Çarpıklık']:.4f}")
        
        # İstatistiksel analiz yorumları
        st.subheader("8.2 İstatistiksel Analiz Yorumları")
        
        # Durağanlık yorumu
        if stats_results['ADF p-değeri'] < 0.05:
            st.success("✓ Fiyat serisi durağan: Belirli bir ortalama etrafında dalgalanma eğiliminde")
        else:
            st.warning("⚠ Fiyat serisi durağan değil: Belirgin bir trend mevcut")
            
        # Normallik yorumu
        if stats_results['Normallik p-değeri'] < 0.05:
            st.warning("⚠ Getiriler normal dağılıma uymuyor: Ekstrem hareketler normalden fazla")
        else:
            st.success("✅ Getiriler normal dağılıma uyuyor: Fiyat hareketleri öngörülebilir aralıkta")
            
        # Otokorelasyon yorumu
        if abs(stats_results['Otokorelasyon']) > 0.2:
            st.info(f"ℹ Güçlü otokorelasyon ({stats_results['Otokorelasyon']:.2f}): Fiyat hareketleri birbirini takip ediyor")
        else:
            st.info("ℹ Zayıf otokorelasyon: Fiyat hareketleri bağımsız")
            
        # Çarpıklık yorumu
        if abs(stats_results['Çarpıklık']) > 1:
            st.warning(f"⚠ Yüksek çarpıklık ({stats_results['Çarpıklık']:.2f}): Asimetrik fiyat hareketleri")
        else:
            st.success("✅ Düşük çarpıklık: Simetrik fiyat hareketleri")
        
        # Örüntü analizi
        st.subheader("8.3 Zamansallık ve Örüntü Analizi")
        patterns = analyze_statistical_patterns(df)
        
        if patterns['Mevsimsellik']:
            st.info("ℹ Mevsimsel örüntü tespit edildi: Periyodik fiyat hareketleri mevcut")
        if patterns['Otokorelasyon']:
            st.info("ℹ Fiyat hareketlerinde süreklilik tespit edildi")
        if patterns['Trend Gücü'] > 1:
            st.warning(f"⚠ Güçlü trend (z-skor: {patterns['Trend Gücü']:.2f})")
        if patterns['Döngüsel Hareket'] > 0.2:
            st.info("ℹ Döngüsel hareket tespit edildi")
            
        st.markdown("""
        **Özet Değerlendirme:**
        1. **Trend Analizi:** {}
        2. **Volatilite:** {}
        3. **Örüntü:** {}
        4. **Öngörülebilirlik:** {}
        """.format(
            "Güçlü trend mevcut" if patterns['Trend Gücü'] > 1 else "Zayıf trend",
            "Yüksek" if risk_metrics['Volatilite'] > 0.3 or risk_metrics['VaR_99'] > 0.2 else "Orta" if risk_metrics['Volatilite'] > 0.15 else "Düşük",
            "Belirgin örüntüler mevcut" if patterns['Mevsimsellik'] or patterns['Otokorelasyon'] else "Belirgin örüntü yok",
            "Yüksek" if patterns['Otokorelasyon'] and stats_results['Normallik p-değeri'] > 0.05 else "Düşük"
        ))

        # 9. RİSK ANALİZİ
        st.header("9. RİSK ANALİZİ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Volatilite", f"%{risk_metrics['Volatilite']*100:.2f}")
            st.metric("Sharpe Oranı", f"{risk_metrics['Sharpe Oranı']:.2f}")
            
        with col2:
            st.metric("Value at Risk (%95)", f"₺{risk_metrics['VaR_95']:.2f}")
            st.metric("Maksimum Kayıp", f"%{risk_metrics['VaR_99']*100:.2f}")
        
        # Risk analizi yorumları
        st.subheader("9.1 Risk Analizi Yorumları")
        
        # Volatilite yorumu
        if risk_metrics['Volatilite'] > 0.3:
            st.warning("⚠ Yüksek volatilite: Riskli yatırım ortamı")
        elif risk_metrics['Volatilite'] > 0.15:
            st.info("ℹ️ Normal volatilite: Orta risk seviyesi")
        else:
            st.success("✅ Düşük volatilite: Düşük risk seviyesi")
            
        # Sharpe oranı yorumu
        if risk_metrics['Sharpe Oranı'] > 1:
            st.success("✅ Yüksek Sharpe oranı: Risk/getiri dengesi iyi")
        elif risk_metrics['Sharpe Oranı'] > 0:
            st.info("ℹ️ Orta Sharpe oranı: Risk/getiri dengesi normal")
        else:
            st.warning("⚠️ Düşük Sharpe oranı: Risk/getiri dengesi zayıf")
            
        # VaR yorumu
        var_pct = risk_metrics['VaR_95'] / df['close'].iloc[-1] * 100
        st.info(f"ℹ️ %95 güven aralığında maksimum %{var_pct:.2f} kayıp beklentisi")
        
        # Maksimum kayıp yorumu
        if risk_metrics['VaR_99'] > 0.2:
            st.warning("⚠️ Yüksek maksimum kayıp: Dikkatli pozisyon alınmalı")
        else:
            st.success("✅ Kabul edilebilir maksimum kayıp seviyesi")
            
        st.markdown("""
        **Risk Analizi Özeti:**
        1. **Genel Risk Seviyesi:** {}
        2. **Yatırım Potansiyeli:** {}
        3. **Pozisyon Önerisi:** {}
        4. **Risk Yönetimi:** Stop-loss seviyesi ₺{} olarak belirlenebilir
        """.format(
            "Yüksek" if risk_metrics['Volatilite'] > 0.3 or risk_metrics['VaR_99'] > 0.2 else "Orta" if risk_metrics['Volatilite'] > 0.15 else "Düşük",
            "İyi" if risk_metrics['Sharpe Oranı'] > 1 else "Orta" if risk_metrics['Sharpe Oranı'] > 0 else "Zayıf",
            "Küçük pozisyon" if risk_metrics['Volatilite'] > 0.3 else "Normal pozisyon",
            f"{df['close'].iloc[-1] * (1 - risk_metrics['VaR_95']):.2f}"
        ))

        # 10. PDF RAPORU
        st.header("10. PDF Raporu")
        
        try:
            # PDF oluştur
            pdf_buffer = create_pdf_report(hisse_adi, df, summary, risk_metrics, stats_results, predictions)
            
            if pdf_buffer:
                # PDF'i indir butonu
                st.download_button(
                    label="📥 PDF Raporu İndir",
                    data=pdf_buffer,
                    file_name=f"{hisse_adi}_analiz_raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key="download_pdf",
                    help="Analiz raporunu PDF formatında indirmek için tıklayın"
                )
                st.success("✅ PDF raporu başarıyla oluşturuldu! İndirmek için yukarıdaki butona tıklayın.")
            else:
                st.error("❌ PDF raporu oluşturulamadı. Lütfen tekrar deneyin.")
                
        except Exception as e:
            st.error(f"PDF oluşturulurken bir hata oluştu: {str(e)}")
            st.info("Lütfen tekrar deneyin veya destek ekibiyle iletişime geçin.")

def create_pdf_report(hisse_adi, df, summary, risk_metrics, stats_results, predictions):
    """PDF raporu oluşturur"""
    # PDF buffer oluştur
    buffer = io.BytesIO()
    
    try:
        # PDF dokümanı oluştur
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Başlık
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph(f"{hisse_adi} Hisse Analiz Raporu", title_style))
        story.append(Spacer(1, 12))
        
        # Tarih
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.gray
        )
        story.append(Paragraph(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}", date_style))
        story.append(Spacer(1, 20))
        
        # Genel Durum
        story.append(Paragraph("1. Genel Durum", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        general_data = [
            ["Metrik", "Değer"],
            ["Son Fiyat", f"₺{df['close'].iloc[-1]:.2f}"],
            ["Trend", summary['trend']],
            ["Risk Durumu", summary['risk_durumu']],
            ["MACD Sinyali", summary['macd_signal']],
            ["Bollinger", summary['bollinger_signal']]
        ]
        
        t = Table(general_data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Risk Analizi
        story.append(Paragraph("2. Risk Analizi", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        risk_data = [
            ["Metrik", "Değer"],
            ["Sharpe Oranı", f"{risk_metrics['Sharpe Oranı']:.2f}"],
            ["VaR (%95)", f"%{abs(risk_metrics['VaR_95']*100):.1f}"],
            ["Volatilite", f"%{risk_metrics['Volatilite']*100:.1f}"],
            ["Maximum Drawdown", f"%{risk_metrics['Max Drawdown']*100:.1f}"]
        ]
        
        t = Table(risk_data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # İstatistiksel Analiz
        story.append(Paragraph("3. İstatistiksel Analiz", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        stats_data = [
            ["Metrik", "Değer"],
            ["Ortalama Getiri", f"%{stats_results['Ortalama Getiri']*100:.2f}"],
            ["Standart Sapma", f"%{stats_results['Standart Sapma']*100:.2f}"],
            ["Çarpıklık", f"{stats_results['Çarpıklık']:.2f}"],
            ["Basıklık", f"{stats_results['Basıklık']:.2f}"]
        ]
        
        t = Table(stats_data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        
        # PDF oluştur
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"PDF oluşturulurken bir hata oluştu: {str(e)}")
        return None

# Ana uygulama
if uploaded_file is not None:
    # Dosya adını kontrol et
    if not uploaded_file.name.startswith(hisse_adi):
        st.error(f"Lütfen {hisse_adi} ile başlayan bir CSV dosyası yükleyin!")
    else:
        # CSV dosyasını oku ve analizleri yap
        df = pd.read_csv(uploaded_file)
        # ... diğer analizler ...
else:
    st.info(f"Lütfen önce hisse adını girin ve ardından {hisse_adi if hisse_adi else 'hisse adı'} ile başlayan CSV dosyasını yükleyin.")
